"""Learning engine, skill model, spaced repetition, progress and XP."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest

from app.models import (
    CodingAttempt,
    User,
    CodingTask,
    DailyPlan,
    Question,
    QuestionAttempt,
    ReviewSchedule,
    Skill,
    StudySession,
    UserSkill,
)
from app.services import learning_engine, progress
from app.services.graph import SkillGraph
from app.services.srs import compute_next, due_rows, schedule_event
from app.services.user_knowledge import apply_forgetting, ensure_skill_rows, update_from_activity


def _answer(db, user, question: Question, *, correct: bool) -> None:
    from app.services.questions import submit_answer

    answer = question.expected_answer or " ".join(question.expected_points or [])
    if not correct:
        answer = "completely unrelated nonsense about cooking pasta"
    submit_answer(db, user, question_id=question.id, answer=answer, allow_ai=False)


def test_curriculum_graph_is_loaded_and_acyclic(db, seeded):
    graph = SkillGraph(db)
    assert len(graph.nodes) >= 100
    assert "opt.gradient_descent" in graph.nodes
    prereqs = graph.prerequisites("opt.gradient_descent")
    assert "calc.partial_gradient" in prereqs, "gradient descent must require partial derivatives"
    assert "linalg.vectors" in prereqs, "and vector/matrix foundations"
    # transitive closure must be finite and must not contain itself
    assert "opt.gradient_descent" not in prereqs
    deeper = graph.prerequisites("ml.logistic_regression")
    assert set(graph.prerequisites("calc.derivatives")).issubset(set(deeper))


def test_readiness_reflects_prerequisites(db, user, seeded):
    ensure_skill_rows(db, user.id)
    graph = SkillGraph(db)
    cold = graph.readiness(user.id, "ml.gradient_descent")
    assert cold["readiness"] == pytest.approx(0.0, abs=1.0)
    assert cold["unmet"], "an unknown learner must have unmet prerequisites"

    update_from_activity(
        db,
        user_id=user.id,
        skill_codes=["calc.partial_gradient", "linalg.vectors", "np.linalg_ops", "opt.gradient_descent", "opt.optimization_mindset"],
        activity="concept",
        score=95,
        correct=True,
        difficulty=2,
    )
    db.commit()
    warm = graph.readiness(user.id, "ml.gradient_descent")
    assert warm["readiness"] > cold["readiness"] + 15
    assert warm["unmet"], "unmet list must stay concrete so the UI can act on it"
    assert all(u["score"] < 55 for u in warm["unmet"])


def test_strong_python_skips_beginner_topics(db, user, seeded):
    """§52: demonstrated fundamentals must not be re-taught - the frontier moves forward."""
    ensure_skill_rows(db, user.id)
    demonstrated = [
        "py.idioms", "py.oo", "py.errors", "py.async",           # all of python.core
        "np.arrays", "np.broadcast", "np.linalg_ops",            # all of python.numpy
        "code.control_flow", "code.types_values", "algo.complexity",
    ]
    for _ in range(2):
        update_from_activity(
            db, user_id=user.id, skill_codes=demonstrated, activity="code", score=97, correct=True, difficulty=5
        )
    db.commit()

    candidates = learning_engine.candidate_topics(db, user, limit=68)
    codes = [c.code for c in candidates]
    for already_known in ("python.core", "python.numpy"):
        assert already_known not in codes, f"must not re-teach demonstrated basics: {codes[:6]}"
    skipped = {t["code"]: t for t in learning_engine.skipped_topics(db, user, limit=68)}
    assert "python.core" in skipped and "python.numpy" in skipped
    assert candidates, "the queue must still contain the next real step"
    assert candidates[0].level >= 1

    # ...and a learner who is weak in Python must get Python reinforced first
    user2 = User(username="weak-py", daily_minutes=45)
    db.add(user2)
    db.commit()
    ensure_skill_rows(db, user2.id)
    for _ in range(3):
        update_from_activity(db, user_id=user2.id, skill_codes=["py.idioms"], activity="code", score=25, correct=False, difficulty=1)
    db.commit()
    weak_candidates = learning_engine.candidate_topics(db, user2, limit=20)
    assert any(c.code in {"python.core", "foundations.code_literacy"} for c in weak_candidates[:5]), [
        c.code for c in weak_candidates[:5]
    ]
    rec = learning_engine.recommend_next(db, user2)
    assert rec.get("topic") or rec.get("remediation")


def test_daily_plan_shape_and_budget(db, user):
    plan = learning_engine.build_daily_plan(db, user, reason="test")
    db.commit()
    kinds = [i["kind"] for i in plan.items]
    assert "learn" in kinds and "practice" in kinds and "coding" in kinds and "reflection" in kinds
    assert sum(i["minutes"] for i in plan.items) <= plan.minutes_budget * 1.35 + 1
    assert all(i["why"] for i in plan.items), "every plan item must justify itself"

    learning = [i for i in plan.items if i["kind"] == "learn"][0]
    learning_engine.mark_item_done(db, user, learning["id"])
    db.commit()
    refreshed = learning_engine.get_or_build_plan(db, user, plan_date=plan.plan_date)
    done = [i for i in refreshed.items if i.get("done")]
    assert [i["id"] for i in done] == [learning["id"]]


def test_review_queue_contains_failed_skill_earlier_than_succeeded(db, user):
    ensure_skill_rows(db, user.id)
    now = datetime.utcnow()
    failed = update_from_activity(db, user_id=user.id, skill_codes=["ml.trees"], activity="concept", score=20, correct=False, difficulty=3)
    ok = update_from_activity(db, user_id=user.id, skill_codes=["ml.metrics"], activity="concept", score=96, correct=True, difficulty=3)
    db.commit()
    assert failed[0].next_review <= now + timedelta(hours=2), "a failure returns the item the same day"
    assert ok[0].next_review >= now + timedelta(hours=23), "a success pushes it out by ~a day"
    due = due_rows(db, user.id, now=datetime.utcnow() + timedelta(hours=3), item_type="skill")
    codes = [r.item_code for r in due]
    assert "ml.trees" in codes and "ml.metrics" not in codes


def test_spaced_repetition_math():
    class Fake:
        ease_factor, interval_days, repetitions, lapses, streak, difficulty = 2.5, 0.0, 0, 0, 0, 5.0

    now = datetime(2026, 1, 1, 12, 0, 0)
    first = compute_next(Fake(), 4, now=now)
    assert first["repetitions"] == 1 and first["interval_days"] == 1.0, "first success -> one day"

    class Reps2:
        ease_factor, interval_days, repetitions, lapses, streak, difficulty = 2.5, 3.0, 1, 0, 1, 5.0

    second = compute_next(Reps2(), 4, now=now)
    assert second["interval_days"] > 1.0, "second success must lengthen the interval"

    class Failed:
        ease_factor, interval_days, repetitions, lapses, streak, difficulty = 2.6, 30.0, 5, 1, 4, 6.0

    after_fail = compute_next(Failed(), 1, now=now)
    assert after_fail["repetitions"] == 0
    assert after_fail["lapses"] == 2
    assert after_fail["interval_days"] == 0.0
    assert after_fail["ease_factor"] < 2.6
    assert after_fail["next_review"] - now == timedelta(minutes=45), "failure must return the item the same day"

    class Again:
        ease_factor, interval_days, repetitions, lapses, streak, difficulty = 2.9, 60.0, 8, 0, 8, 3.0

    big = compute_next(Again(), 5, now=now)
    assert big["interval_days"] > 60.0, "sustained success must grow the interval"
    assert 1.3 <= big["ease_factor"] <= 3.1


def test_scores_move_across_dimensions(db, user, seeded):
    ensure_skill_rows(db, user.id)
    skill = db.query(Skill).filter(Skill.code == "ml.linear_regression_impl").one()
    (update,) = update_from_activity(db, user_id=user.id, skill_codes=["ml.linear_regression_impl"], activity="code", score=90, correct=True, difficulty=3)
    row = db.query(UserSkill).filter(UserSkill.user_id == user.id, UserSkill.skill_id == skill.id).one()
    db.commit()
    assert update.after > 0
    assert row.coding_score > row.math_score, "a coding activity must lift the coding dimension more"
    assert row.confidence > 0 and row.attempts == 1 and row.streak == 1

    update_from_activity(db, user_id=user.id, skill_codes=["ml.linear_regression_impl"], activity="math", score=85, correct=True, difficulty=3)
    db.commit()
    db.refresh(row)
    assert row.math_score > 0 and row.attempts == 2 and row.streak == 2
    assert row.mastery_state in {"learning", "developing", "solid", "strong"}


def test_repeated_failure_triggers_prerequisite_repair(db, user):
    ensure_skill_rows(db, user.id)
    question = db.query(Question).filter(Question.topic_code == "ml.gradient_descent").first() or db.query(Question).first()
    for _ in range(3):
        _answer(db, user, question, correct=False)
    db.commit()
    state = learning_engine.learner_state(db, user)
    assert state["weaknesses"], "the learner should now have recorded weak skills"
    rec = learning_engine.recommend_next(db, user)
    assert rec.get("topic") or rec.get("remediation")
    due = due_rows(db, user.id, item_type="skill", limit=200, now=datetime.utcnow() + timedelta(hours=2))
    assert due, "failed skills must be scheduled for review"


def test_forgetting_decay_is_applied(db, user, seeded):
    ensure_skill_rows(db, user.id)
    update_from_activity(db, user_id=user.id, skill_codes=["ml.trees"], activity="concept", score=80, correct=True, difficulty=3)
    db.commit()
    skill = db.query(Skill).filter(Skill.code == "ml.trees").one()
    row = db.query(UserSkill).filter(UserSkill.user_id == user.id, UserSkill.skill_id == skill.id).one()
    before = row.knowledge_score
    row.last_review = datetime.utcnow() - timedelta(days=240)
    db.commit()
    touched = apply_forgetting(db, user.id)
    db.commit()
    assert touched >= 1
    db.refresh(row)
    assert row.knowledge_score < before


def test_progress_is_not_pages_read_and_velocity_counts_work(db, user, seeded):
    ensure_skill_rows(db, user.id)
    update_from_activity(db, user_id=user.id, skill_codes=["np.arrays"], activity="math", score=88, correct=True, difficulty=2, xp=12, minutes=9)
    db.add(StudySession(user_id=user.id, activity="practice", started_at=datetime.utcnow() - timedelta(hours=2), ended_at=datetime.utcnow(), active_seconds=900, xp=30, items_done=3))
    db.add(CodingTask(slug="tmp-task", title="tmp", test_code="", starter_code=""))
    db.commit()
    task = db.query(CodingTask).filter(CodingTask.slug == "tmp-task").one()
    db.add(CodingAttempt(user_id=user.id, task_id=task.id, code="x=1", passed=True, total_tests=3, passed_tests=3))
    db.commit()

    summary = progress.progress_summary(db, user)
    assert set(summary["dimensions"]) == {"theory", "math", "coding", "problem_solving", "engineering"}
    assert summary["skills_assessed"] >= 1
    assert summary["xp"] >= 12
    assert summary["ml_engineer_score"] <= summary["weighted_dimension_score"]

    velocity = progress.velocity(db, user, days=7)
    assert velocity["coding_tasks_per_week"] >= 0.1
    assert velocity["study_minutes_total"] >= 15
    assert velocity["days_active"] >= 1
    assert progress.streak(db, user.id)["current"] >= 1


def test_journal_gaps_feed_the_engine(db, user, seeded):
    from app.services import notes as notes_service

    ensure_skill_rows(db, user.id)
    before_due = len(due_rows(db, user.id, item_type="skill", limit=500))
    entry = notes_service.write_journal(
        db,
        user,
        content="Today I finally understood ridge regression. But I don't understand the gradient of cross entropy and I keep forgetting what batchnorm does.",
        minutes=30,
    )
    codes = [g["code"] for g in entry["analysis"]["gaps"]]
    assert any("cross" in c or "entropy" in c or "backprop" in c or "info" in c for c in codes), codes
    after_due = len(due_rows(db, user.id, item_type="skill", limit=500))
    assert after_due > before_due, "journal-detected gaps must enter the review queue"

    memory = progress.memory_bundle(db, user.id)
    assert memory["weaknesses"], "journal gaps must be stored in learning memory"
