"""Projects, exams, mock interview, notes/journal, dashboard - service level."""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from app.models import (
    CodingTask,
    Exam,
    ExamAttempt,
    InterviewSession,
    LearningMemory,
    PracticeAttempt,
    Project,
    ProjectAttempt,
    ProjectEnrollment,
    ProjectTask,
    Question,
    Skill,
    UserNote,
    UserSkill,
)
from app.config import settings
from app.services import exams as exam_service
from app.services import interview as interview_service
from app.services import learning_engine, projects as project_service
from app.services.graph import SkillGraph
from app.services.user_knowledge import ensure_skill_rows


def test_projects_cover_all_levels_and_have_rubrics(db, fast_seeded):
    rows = list(db.query(Project).all())
    levels = {p.target_level for p in rows}
    assert {"beginner", "junior", "middle", "strong_middle", "senior"} <= levels, levels
    senior = next(p for p in rows if p.target_level == "senior")
    dims = " ".join(str(r.get("dimension", "")) for r in (senior.rubric or []))
    for required in ("Architecture", "Monitoring", "Scalability", "Cost", "Documentation", "Testing"):
        assert required in dims, f"senior rubric must cover {required}"
    assert senior.milestones and all(m.get("acceptance_criteria") for m in senior.milestones)


def test_enrolment_materialises_milestones_and_progress_moves(db, fast_seeded, user):
    project = db.query(Project).filter(Project.slug == "junior-churn-prediction").one()
    payload = project_service.enroll(db, user, project.id)
    assert payload["milestones_total"] >= 4
    assert payload["progress"] == 0.0
    task = db.get(ProjectTask, payload["milestones"][0]["id"])
    task.status = "doing"
    db.commit()
    project_service.set_milestone_status(db, user, task.id, "done", notes="protocol written, splits reviewed")
    db.commit()
    updated = project_service.enrollment_payload(db, db.get(ProjectEnrollment, payload["id"]))
    assert updated["milestones_done"] == 1 and updated["progress"] > 0
    assert "review" in json.dumps(updated["milestones"]) or updated["milestones"][0]["review_notes"]


def test_mentor_refuses_to_write_the_project_but_gives_structure(db, fast_seeded, user):
    project = db.query(Project).filter(Project.slug == "middle-rag-service").one()
    enrollment = project_service.enroll(db, user, project.id)
    reply = project_service.mentor_reply(db, user, "Just write the whole code for me, I do not want to implement it.", enrollment_id=enrollment["id"])
    text = reply["text"]
    assert "not write" in text.lower() or "will not" in text.lower(), text[:200]
    assert "```" in text, "a refusal without a concrete structure is useless"
    assert "What you should implement now" in text or "Next milestone" in text
    db.expunge_all()
    attempts = db.query(ProjectAttempt).filter(ProjectAttempt.enrollment_id == enrollment["id"]).all()
    kinds = {a.kind for a in attempts}
    assert ProjectAttempt.KIND_AI_SUGGESTION in kinds or ProjectAttempt.KIND_MESSAGE in kinds
    ai_rows = [a for a in attempts if a.author == "ai"]
    assert ai_rows and all("suggested" in (a.kind or "") or a.ai_suggested for a in ai_rows), "AI help must be attributed"


def test_submission_and_evaluation_update_the_engineering_model(db, fast_seeded, user):
    ensure_skill_rows(db, user.id)
    project = db.query(Project).filter(Project.slug == "beginner-tabular-analysis").one()
    enrollment = project_service.enroll(db, user, project.id)
    first_task = db.query(ProjectTask).filter(ProjectTask.enrollment_id == enrollment["id"]).order_by(ProjectTask.order_index).first()
    project_service.record_submission(
        db,
        user,
        enrollment_id=enrollment["id"],
        task_id=first_task.id,
        content="Data audit: 12 columns, 3 with >20% missing; excluded 'boat' (post-outcome). Tests: test_dtypes, test_no_nan_leakage pass.",
        user_implemented="audit script + column classification",
    )
    for task in db.query(ProjectTask).filter(ProjectTask.enrollment_id == enrollment["id"]).all():
        project_service.set_milestone_status(db, user, task.id, "done")
    db.commit()
    report = project_service.evaluate(db, user, enrollment["id"], allow_ai=False)
    assert report["engine"] == "local" and "checklist" in report["note"].lower()
    assert report["total"] > 60
    assert {d["dimension"] for d in report["dimensions"]} >= {"ML correctness", "Testing", "Documentation"}
    db.expunge_all()
    row = (
        db.query(UserSkill)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .filter(UserSkill.user_id == user.id, Skill.code == "test.unit")
        .first()
    )
    assert row is not None and row.engineering_score > 0


def test_exam_is_built_per_section_and_grades_objectively(db, fast_seeded, user):
    ensure_skill_rows(db, user.id)
    exam = db.query(Exam).filter(Exam.code == "junior_ml_engineer").one()
    started = exam_service.start(db, user, exam.code)
    assert started["items"], "exam must contain items"
    sections = {i["section"] for i in started["items"]}
    assert sections <= {s["name"] for s in exam.sections}

    answers = {}
    for item in started["items"]:
        question = db.get(Question, item["id"])
        if question.options and question.correct_option is not None:
            answers[str(item["id"])] = {"answer": str(question.correct_option), "selected_option": question.correct_option}
        else:
            answers[str(item["id"])] = {"answer": " ".join(question.expected_points or [question.expected_answer])}
    exam_service.save_answers(db, user, started["attempt_id"], answers)
    graded = exam_service.submit(db, user, started["attempt_id"])
    assert graded["report"]["score"] > 50
    assert graded["report"]["section_scores"], "per-section breakdown is mandatory"
    assert graded["state"] == "graded"
    db.expunge_all()
    attempt = db.get(ExamAttempt, started["attempt_id"])
    assert attempt.passed in (True, False)
    assert db.query(LearningMemory).filter(LearningMemory.user_id == user.id, LearningMemory.category == "exam").count() >= 1
    # exams must move the knowledge model
    assert db.query(UserSkill).filter(UserSkill.user_id == user.id, UserSkill.attempts > 0).count() >= 1


def test_senior_exam_focuses_on_design_tradeoffs_and_ops(db, fast_seeded):
    exam = db.query(Exam).filter(Exam.code == "senior_ml_engineer").one()
    text = json.dumps(exam.sections, ensure_ascii=False).lower() + exam.description.lower()
    for required in ("design", "reliability", "cost", "monitoring"):
        assert required in text, required


def test_readiness_is_computed_from_the_skill_model_not_attendance(db, fast_seeded, user):
    cold = exam_service.exam_readiness(db, user)
    assert cold and all(0 <= row["readiness"] <= 100 for row in cold)
    ensure_skill_rows(db, user.id)
    from app.services.user_knowledge import update_from_activity

    for _ in range(3):
        update_from_activity(
            db,
            user_id=user.id,
            skill_codes=[c for c, n in SkillGraph(db).nodes.items() if n.level <= 3],
            activity="concept",
            score=95,
            correct=True,
            difficulty=4,
        )
    db.commit()
    warm = {row["code"]: row for row in exam_service.exam_readiness(db, user)}
    assert warm["junior_ml_engineer"]["readiness"] > dict((r["code"], r["readiness"]) for r in cold)["junior_ml_engineer"]


def test_interview_runs_turns_and_reports_six_axes(db, fast_seeded, user):
    ensure_skill_rows(db, user.id)
    session = interview_service.start(db, user, level="middle", focus="production ml")
    sid = session["id"]
    assert session["turns"][0]["role"] == "interviewer"
    good = (
        "I would split point-in-time rather than randomly, because a random split leaks future aggregates. "
        "I would use a group-aware split by account, monitor the score distribution, and calibrate because the "
        "threshold decision depends on probabilities, not ranking. The failure mode I fear most is silent upstream "
        "schema change, so I would add contract checks and alert on null-rate and PSI."
    )
    for _ in range(3):
        result = interview_service.answer(db, user, sid, good, seconds=90)
        assert result["snapshot"]["turns"]
    finished = interview_service.finish(db, user, sid)
    report = finished["report"]
    assert report["engine"] == "local"
    for axis in ("technical_knowledge", "depth", "reasoning", "communication", "system_design", "production_thinking"):
        assert axis in finished["scores"], axis
        assert 0 <= finished["scores"][axis] <= 100
    assert report["per_turn"] and report["narrative"]
    db.expunge_all()
    stored = db.get(InterviewSession, sid)
    assert stored.state == "finished" and stored.finished_at is not None
    assert db.query(LearningMemory).filter(LearningMemory.user_id == user.id, LearningMemory.category == "exam").count() >= 1


def test_short_weak_answers_score_lower_in_the_interview(db, fast_seeded, user):
    ensure_skill_rows(db, user.id)
    strong = interview_service.start(db, user, level="senior", focus="system design")
    weak = interview_service.start(db, user, level="senior", focus="system design")
    for _ in range(3):
        interview_service.answer(db, user, strong["id"], "Capacity math first: 50k RPS with 20ms of CPU work per request means about 1000 cores; I would shard the ANN index, cache top-k per user with stampede protection, degrade to a cached popularity list on timeout, and alert on error budget burn.", seconds=120)
        interview_service.answer(db, user, weak["id"], "use redis and kubernetes", seconds=15)
    strong_report = interview_service.finish(db, user, strong["id"])
    weak_report = interview_service.finish(db, user, weak["id"])
    assert strong_report["scores"]["overall"] > weak_report["scores"]["overall"]
    assert weak_report["scores"]["communication"] < strong_report["scores"]["communication"]


def test_notes_bookmarks_and_mistake_capturing(db, seeded, user):
    from app.services import notes as notes_service
    from app.models import DocumentChunk

    chunk = db.query(DocumentChunk).first()
    note = notes_service.create_note(
        db, user, kind="concept", title="softmax stability", body="subtract max", chunk_id=chunk.id, tags=["numerical"]
    )
    assert note["citation"].get("document"), "a note attached to a chunk keeps its provenance"
    assert note["excerpt"]
    updated = notes_service.update_note(db, user, note["id"], pinned=True, body="subtract row max before exp; shift-invariance makes it exact")
    assert updated["pinned"] is True
    listed = notes_service.list_notes(db, user, kind="concept")
    assert listed and listed[0]["id"] == note["id"]
    searched = notes_service.list_notes(db, user, q="shift-invariance")
    assert [n["id"] for n in searched] == [note["id"]]
    assert notes_service.delete_note(db, user, note["id"]) is True
    db.expunge_all()
    assert db.query(UserNote).filter(UserNote.id == note["id"]).first() is None


def test_mistake_becomes_a_note_with_its_source(db, seeded, user):
    from app.services import notes as notes_service
    from app.services.questions import submit_answer

    question = db.query(Question).filter(Question.source_chunk_id.isnot(None)).first() or db.query(Question).first()
    result = submit_answer(db, user, question_id=question.id, answer="nonsense", allow_ai=False)
    if not result["verdict"]["correct"]:
        note = notes_service.note_from_mistake(db, user, result["attempt_id"])
        assert note is not None and note["kind"] == "mistake"
        assert "Q:" in note["body"] and "My answer:" in note["body"]


def test_recommendation_changes_after_results(db, seeded, user):
    """§51: results must change what the engine serves next."""
    ensure_skill_rows(db, user.id)
    graph = SkillGraph(db)
    before = learning_engine.recommend_next(db, user)
    plan_before = learning_engine.build_daily_plan(db, user, plan_date=learning_engine.date.today(), reason="test")
    db.commit()
    learn_before = [i for i in plan_before.items if i["kind"] == "learn"]
    assert learn_before, "the engine must always propose what to learn"
    focus = learn_before[0]["code"]

    from app.services.user_knowledge import update_from_activity

    for _ in range(3):
        update_from_activity(
            db,
            user_id=user.id,
            skill_codes=graph.topic_skills.get(focus, [])[:3] or [focus],
            activity="code",
            score=97,
            correct=True,
            difficulty=5,
        )
    db.commit()
    plan_after = learning_engine.build_daily_plan(db, user, plan_date=learning_engine.date.today(), reason="test")
    db.commit()
    learn_after = [i for i in plan_after.items if i["kind"] == "learn" and not i.get("payload", {}).get("remediation")]
    assert focus not in [i["code"] for i in learn_after], (focus, [i["code"] for i in learn_after])
    after = learning_engine.recommend_next(db, user)
    assert (after.get("topic") or {}).get("code") != focus
    skipped = {t["code"] for t in learning_engine.skipped_topics(db, user, limit=68)}
    assert focus in skipped


def test_practice_and_coding_feed_study_time_and_xp(db, fast_seeded, user):
    from app.services import practice as practice_service
    from app.services import progress
    from app.services.coding import submit as coding_submit

    task = practice_service.generate_for_topic(db, user, topic_code="ml.linear_models", level="beginner", count=1, allow_ai=False)
    assert task
    result = practice_service.submit(db, user, task_id=task[0].id, answer=str(task[0].expected_value or "answer"), allow_ai=False)
    assert result["attempt_id"]
    db.expunge_all()
    assert db.query(PracticeAttempt).filter(PracticeAttempt.user_id == user.id).count() == 1

    session_row = progress.start_session(db, user, activity="practice", topic_code="ml.linear_models")
    db.commit()
    coding_task = db.query(CodingTask).filter(CodingTask.slug == "numpy.moving_average").one()
    out = coding_submit(db, user, task_id=coding_task.id, code=coding_task.solution, seconds=600, session_id=session_row.id)
    assert out["solved"] is True
    progress.end_session(db, user, session_row.id, xp=5)
    db.commit()
    velocity = progress.velocity(db, user, days=7)
    assert velocity["coding_tasks_per_week"] >= 0.1
    assert velocity["study_minutes_total"] >= 0
    assert velocity["skills_improved"] >= 1
    assert velocity["projects_per_month"] == 0


def test_ai_usage_counters_and_limits_are_real(db, fast_seeded, user, monkeypatch):
    from app.ai.base import AIResult, AIProvider, Usage
    from app.ai.manager import AIManager
    from app.models import AIRequest, AIUsage

    class Counting(AIProvider):
        name = "counter"
        calls = 0

        def complete(self, messages, **kwargs):
            type(self).calls += 1
            return AIResult(ok=True, text='{"ok": true}', data={"ok": True}, provider="counter", model="counter", usage=Usage(input_tokens=40, output_tokens=10))

        def health(self):
            return {"provider": "counter", "configured": True, "model": "counter"}

    import app.ai.manager as manager_module

    manager_module.manager._provider = Counting()
    from app.services import teacher as teacher_service

    for i in range(2):
        teacher_service.respond(db, user, text=f"What is attention {i}", mode="explain")
    db.expunge_all()
    rows = db.query(AIRequest).all()
    assert rows and all(r.provider in {"counter", "none"} for r in rows)
    usage = db.query(AIUsage).first()
    assert usage is not None and usage.requests >= 1

    monkeypatch.setattr(settings, "max_ai_requests_per_day", 0)
    blocked = manager_module.manager.invoke("explain", user_id=user.id, kwargs={"topic": "x", "level": 1}, session=db)
    assert blocked.ok is False, blocked
    assert blocked.error_code == "limited", blocked
    db.expunge_all()
    assert db.query(AIUsage).first().limited >= 1
    manager_module.manager.reset()
