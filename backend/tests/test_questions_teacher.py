"""
Question engine + grading + AI Teacher, with and without an AI provider.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from app.config import settings
from app.models import Question, QuestionAttempt, ReviewSchedule, TeacherConversation, TeacherMessage, User
from app.services import questions as qservice
from app.services import teacher as teacher_service


def _user(db, username="q-learner"):
    from app.models import User

    user = db.query(User).filter(User.username == username).one()
    return user


def test_select_next_respects_topic_and_difficulty_ladder(db, fast_seeded, user):
    first = qservice.select_next(db, user, topic_code="ml.linear_models")
    assert first and first["question"]["stem"]
    assert first["origin"] in {"bank", "generated"}
    db.commit()
    # answering it must change what is served next (no repeats within the recent window)
    qid = first["question"]["id"]
    qservice.submit_answer(db, user, question_id=qid, answer="x", allow_ai=False)
    nxt = qservice.select_next(db, user, topic_code="ml.linear_models")
    assert nxt and nxt["question"]["id"] != qid


def test_grading_mcq_math_and_open(db, fast_seeded, user):
    mcq = db.query(Question).filter(Question.question_type == "mcq", Question.options != []).first()
    assert mcq is not None
    good = qservice.grade_locally(question=mcq, answer="", selected_option=mcq.correct_option)
    assert good["correct"] is True and good["score"] >= 60
    bad = qservice.grade_locally(question=mcq, answer="", selected_option=(mcq.correct_option + 1) % len(mcq.options))
    assert bad["correct"] is False

    math_q = db.query(Question).filter(Question.expected_value != "", Question.expected_value.isnot(None)).first()
    assert math_q is not None
    exact = qservice.grade_locally(question=math_q, answer="working out ... final 4.0")
    value = float(math_q.expected_value)
    exact = qservice.grade_locally(question=math_q, answer=f"I get {value}")
    assert exact["correct"] is True, (math_q.expected_value, exact)
    wrong = qservice.grade_locally(question=math_q, answer="I get 999")
    assert wrong["correct"] is False and wrong["error_type"] == "math"

    open_q = db.query(Question).filter(Question.question_type == "open").first()
    model = " ".join(open_q.expected_points or [])
    strong = qservice.grade_locally(question=open_q, answer=model)
    thin = qservice.grade_locally(question=open_q, answer="not sure, something about it")
    assert strong["score"] > thin["score"]
    assert thin["dimensions"]["depth"] < strong["dimensions"]["depth"]


def test_wrong_answer_produces_correction_and_a_retest_question(db, fast_seeded, user):
    from app.services.user_knowledge import ensure_skill_rows

    ensure_skill_rows(db, user.id)
    question = db.query(Question).filter(Question.question_type == "mcq").first()
    result = qservice.submit_answer(
        db, user, question_id=question.id, answer="wrong", selected_option=(question.correct_option + 1) % len(question.options), allow_ai=False
    )
    verdict = result["verdict"]
    assert verdict["correct"] is False
    assert verdict.get("correction") or verdict.get("feedback")
    assert result["retry_question"], "a wrong answer must be followed by a similar question"
    assert result["skill_updates"], "the knowledge model must move"
    db.refresh(question)
    assert question.answer_count == 1 and question.correct_count == 0

    db.expunge_all()
    attempt = db.query(QuestionAttempt).filter(QuestionAttempt.user_id == user.id).order_by(QuestionAttempt.id.desc()).first()
    assert attempt is not None and attempt.evaluated_by == "local"
    retry = db.get(Question, result["retry_question"]["id"])
    assert retry is not None and retry.topic_code == question.topic_code


def test_generated_questions_are_deduplicated_and_cited(db, seeded, user):
    from app.services.user_knowledge import ensure_skill_rows

    ensure_skill_rows(db, user.id)
    first = qservice.generate_questions(db, user, topic_code="math.calculus", count=3, types=["conceptual", "open"], difficulty=2)
    db.commit()
    assert first, "offline generator must produce questions from the indexed library"
    hashes = {q.content_hash for q in first}
    second = qservice.generate_questions(db, user, topic_code="math.calculus", count=3, types=["conceptual", "open"], difficulty=2)
    db.commit()
    assert not (hashes & {q.content_hash for q in second}), "duplicates must be refused"
    grounded = [q for q in first if q.source_chunk_id]
    for q in first:
        assert q.stem and q.question_type
        if q.citation:
            assert q.citation.get("document"), "a citation must name a real document"
    db.expunge_all()
    assert db.query(Question).filter(Question.generated_by == "template").count() >= len(first)


def test_teacher_explain_uses_the_library_and_cites_it(db, seeded, user):
    payload = teacher_service.respond(db, user, text="What is gradient descent?", mode="explain")
    db.commit()
    assert payload["answer"]
    assert payload["grounded"] is True
    assert payload["sources"], "an answer built from the library must show sources"
    assert payload["engine"] in {"local", "ai"}
    assert "Check your understanding" in payload["answer"] or "check" in payload["answer"].lower()
    for source in payload["sources"]:
        assert source.get("document")
    db.expunge_all()
    conv = db.get(TeacherConversation, payload["conversation_id"])
    assert conv is not None and conv.message_count >= 2
    msgs = db.query(TeacherMessage).filter(TeacherMessage.conversation_id == conv.id).all()
    assert {m.role for m in msgs} == {"user", "assistant"}
    assistant = [m for m in msgs if m.role == "assistant"][0]
    assert assistant.sources, "sources must be persisted with the message"


def test_teacher_admits_when_the_library_has_nothing(db, seeded, user):
    payload = teacher_service.respond(db, user, text="Explain the 1997 Albanian monetary policy transmission mechanism")
    assert payload["sources"] == [], "weak matches must not be presented as citations"
    assert "not covered" in payload["answer"].lower() or "does not contain" in payload["answer"].lower()


def test_teacher_quiz_mode_creates_and_returns_questions(db, seeded, user):
    from app.services.user_knowledge import ensure_skill_rows

    ensure_skill_rows(db, user.id)
    payload = teacher_service.respond(db, user, text="Quiz me on gradient descent", mode="quiz", topic_code="ml.gradient_descent")
    db.commit()
    assert payload["attachments"], "quiz mode must attach real questions"
    assert payload["pending"]["kind"] == "quiz"
    question = db.get(Question, payload["pending"]["question_ids"][0])
    graded = qservice.submit_answer(db, user, question_id=question.id, answer="the model does the thing", allow_ai=False)
    assert graded["verdict"]["score"] >= 0 and graded["skill_updates"] is not None


def test_teacher_socratic_mode_asks_instead_of_answering(db, seeded, user):
    payload = teacher_service.respond(db, user, text="How does the softmax gradient work?", mode="socratic", topic_code="math.info_theory")
    answer = payload["answer"].lower()
    assert "?" in payload["answer"], "socratic mode must ask a question"
    assert "step 1" in answer or "q1" in answer
    assert "the answer is" not in answer


def test_review_my_answer_updates_plan_and_schedule(db, seeded, user):
    from app.services import learning_engine
    from app.services.user_knowledge import ensure_skill_rows

    ensure_skill_rows(db, user.id)
    question = db.query(Question).filter(Question.topic_code == "math.calculus").first() or db.query(Question).first()
    payload = teacher_service.respond(
        db, user, text="y = 3x^2 so y' = 6x, at x=2 that is 12", mode="review_answer", topic_code=question.topic_code, answer_to_question_id=question.id
    )
    assert "Verdict" in payload["answer"]
    due = db.query(ReviewSchedule).filter(ReviewSchedule.user_id == user.id, ReviewSchedule.item_type == "skill").count()
    assert due >= 1, "reviewing an answer must schedule a review"


def test_ai_grading_is_used_when_configured_and_marked(db, seeded, user, monkeypatch):
    from app.ai.base import AIResult, AIProvider, Usage

    class Stub(AIProvider):
        name = "stub"

        def complete(self, messages, **kwargs):
            return AIResult(
                ok=True,
                text=json.dumps(
                    {
                        "correct": True,
                        "score": 88,
                        "dimensions": {"correctness": 90, "depth": 80, "reasoning": 85, "precision": 92, "confidence": 70},
                        "error_type": "none",
                        "what_is_wrong": "",
                        "correction": "",
                        "feedback": "Solid answer; you could add the log-sum-exp detail.",
                        "missing_points": [],
                        "check_question": None,
                    }
                ),
                data={
                    "correct": True,
                    "score": 88,
                    "dimensions": {"correctness": 90, "depth": 80, "reasoning": 85, "precision": 92, "confidence": 70},
                    "error_type": "none",
                    "what_is_wrong": "",
                    "correction": "",
                    "feedback": "Solid answer; you could add the log-sum-exp detail.",
                    "missing_points": [],
                    "check_question": None,
                },
                provider="stub",
                model="stub",
                usage=Usage(input_tokens=100, output_tokens=40),
            )

        def health(self):
            return {"provider": "stub", "configured": True, "model": "stub"}

    import app.ai.manager as manager_module

    monkeypatch.setattr(manager_module.manager, "_provider", Stub())
    question = db.query(Question).filter(Question.question_type == "open").first()
    result = qservice.submit_answer(db, user, question_id=question.id, answer="because the log-likelihood gradient is p minus y", allow_ai=True)
    assert result["engine"] == "ai", result["verdict"]
    assert result["verdict"]["score"] == 88
    attempt = db.get(__import__("app.models", fromlist=["QuestionAttempt"]).QuestionAttempt, result["attempt_id"])
    assert attempt.evaluated_by == "ai"


def test_ai_failures_fall_back_to_local_grading(db, seeded, user, monkeypatch):
    from app.ai.base import AIResult, AIProvider

    class Broken(AIProvider):
        name = "broken"

        def complete(self, messages, **kwargs):
            return AIResult(ok=False, error_code="http", error="502 from gateway", provider="broken")

        def health(self):
            return {"provider": "broken", "configured": True, "model": "broken"}

    import app.ai.manager as manager_module

    monkeypatch.setattr(manager_module.manager, "_provider", Broken())
    question = db.query(Question).filter(Question.question_type == "open").first()
    result = qservice.submit_answer(db, user, question_id=question.id, answer="ridge shrinks coefficients", allow_ai=True)
    assert result["engine"] == "local"
    assert result["verdict"]["ai_error_code"] == "http" or "offline" in json.dumps(result["verdict"]).lower()


def test_offline_mode_message_is_actionable(client, auth_client, db, seeded, user):
    from app.config import settings as live

    assert live.ai_available() is False, "tests run with no AI configured"
    payload = teacher_service.respond(db, user, text="Explain regularisation", mode="explain")
    text = payload["answer"] + json.dumps(payload.get("ai", {}))
    assert ".env" in text or "offline" in text.lower()
