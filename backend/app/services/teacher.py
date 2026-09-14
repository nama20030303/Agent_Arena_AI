"""
AI Teacher (§8, §9, §30, §31).

The Teacher is not a chatbot with a RAG prefix. It is a state machine that:
  * reads the learner's position (skill graph, weak topics, previous mistakes,
    current plan/topic) before composing anything;
  * answers from the knowledge base when the library covers the question - and says
    so explicitly when it does not, instead of inventing a citation;
  * runs a teaching loop for the chosen mode: explain -> example -> check -> store;
  * works without AI: the offline engine composes a grounded explanation and generates
    real questions from the retrieved text.

Modes: explain, practice, quiz, deep_dive, socratic, code_review, project_mentor,
interview, exam_preparation.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import asdict
from datetime import datetime
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.knowledge.retrieval import (
    Retriever,
    build_context,
    detect_conflicts,
    format_sources_for_user,
)
from app.models import (
    Document,
    LearningMemory,
    Question,
    TeacherConversation,
    TeacherMessage,
    Topic,
    User,
)
from app.services.graph import SkillGraph
from app.services.learning_engine import learner_state

MODES = (
    "explain",
    "practice",
    "quiz",
    "deep_dive",
    "socratic",
    "code_review",
    "project_mentor",
    "interview",
    "exam_prep",
    "review_answer",
)

MODE_LABELS = {
    "explain": "Explain",
    "practice": "Practice",
    "quiz": "Quiz me",
    "deep_dive": "Deep Dive",
    "socratic": "Socratic Mode",
    "code_review": "Code Review",
    "project_mentor": "Project Mentor",
    "interview": "Interview",
    "exam_prep": "Exam Preparation",
    "review_answer": "Review My Answer",
}

DO_WORLDS_RE = re.compile(
    r"\b(explain|what(?:'s| is| are)|define|how (?:does|do|to)|why|difference between|tell me about|объясн|что такое|как работает|зачем)\b",
    re.IGNORECASE,
)


def _new_topic_guess(text: str, graph: SkillGraph, session: Session) -> tuple[str, float]:
    """Map a free-text question to the closest curriculum topic (skills resolve to their topic)."""
    lowered = (text or "").lower()
    scores: dict[str, float] = {}
    for code, node in graph.nodes.items():
        hits = sum(1 for kw in node.keywords if kw and kw.lower() in lowered)
        if node.name and node.name.lower() in lowered:
            hits += 5
        if hits:
            topic = node.topic_code or code
            scores[topic] = scores.get(topic, 0) + hits
    for topic in graph.topics.values():
        hits = sum(1 for kw in (topic.keywords or []) if kw and kw.lower() in lowered)
        if topic.name and topic.name.lower() in lowered:
            hits += 6
        if hits:
            scores[topic.code] = scores.get(topic.code, 0) + hits * 1.5
    if not scores:
        return "", 0.0
    best = max(scores, key=lambda k: scores[k])
    return best, float(scores[best])


def get_or_create_conversation(
    session: Session, user: User, *, conversation_id: int | None, mode: str, topic_code: str = ""
) -> TeacherConversation:
    conv: TeacherConversation | None = None
    if conversation_id:
        conv = session.get(TeacherConversation, conversation_id)
        if conv is not None and conv.user_id != user.id:
            conv = None
    if conv is None:
        conv = TeacherConversation(
            user_id=user.id,
            mode=mode,
            topic_code=topic_code,
            level=user.level_index,
            title="AI Teacher session",
            context={},
        )
        session.add(conv)
        session.flush()
    if mode and conv.mode != mode:
        conv.mode = mode
    if topic_code and conv.topic_code != topic_code:
        conv.topic_code = topic_code
    conv.level = user.level_index
    conv.updated_at = datetime.utcnow()
    if conv.title in {"", "New session", "AI Teacher session"}:
        conv.title = f"{MODE_LABELS.get(mode, mode.title)} · {(topic_code or 'free chat')}"[:120]
    session.flush()
    return conv


def conversation_messages(session: Session, conversation: TeacherConversation, *, limit: int = 40) -> list[dict[str, str]]:
    rows = session.execute(
        select(TeacherMessage).where(TeacherMessage.conversation_id == conversation.id).order_by(TeacherMessage.id.asc())
    ).scalars()
    return [{"role": m.role, "content": m.content, "mode": m.mode} for m in list(rows)[-limit:]]


# --------------------------------------------------------------------------- #
#  Main entry point
# --------------------------------------------------------------------------- #
def respond(
    session: Session,
    user: User,
    *,
    text: str,
    mode: str = "explain",
    conversation_id: int | None = None,
    topic_code: str = "",
    allow_web: bool | None = None,
    code: str = "",
    answer_to_question_id: int | None = None,
) -> dict[str, Any]:
    mode = mode if mode in MODES else "explain"
    graph = SkillGraph(session)
    conversation = get_or_create_conversation(session, user, conversation_id=conversation_id, mode=mode, topic_code=topic_code)
    state = learner_state(session, user, graph=graph)
    guessed_topic, guess_hits = _new_topic_guess(text, graph, session)
    topic = topic_code or guessed_topic or (conversation.topic_code or "") or state.get("current_topic", "")

    # ---- retrieve from the user's library -----------------------------------
    query = text if DO_WORLDS_RE.search(text) or len(text) > 60 else f"{topic} {text}"
    retriever = Retriever(session)
    sufficient, coverage, results = retriever.sufficient(query, top_k=settings.retrieval_top_k)
    web_used = False
    web_note = ""
    want_web = user.web_search_allowed if allow_web is None else allow_web
    if want_web and settings.web_ingestion_enabled and not sufficient:
        web_note = (
            "Your library does not cover this (coverage "
            f"{round(coverage * 100)}%). Web retrieval is not automatic: add the page as a source "
            "(Library -> Add URL) and I will use it as a citable tier-2..5 source."
        )

    context, citations = build_context(results, max_chars=settings.max_context_chars)
    conflicts = detect_conflicts(results, term=_key_term(text, topic)) if len(results) >= 2 else []
    grounded = sufficient and bool(results)

    # ---- learner question row for this turn ---------------------------------
    session.add(
        TeacherMessage(conversation_id=conversation.id, role="user", content=text[:20000], mode=mode)
    )
    conversation.message_count = (conversation.message_count or 0) + 1

    payload: dict[str, Any]
    pending: dict[str, Any] = {}
    engine = "local"
    ai_usage: dict[str, Any] = {}
    attachments: list[dict[str, Any]] = []

    if mode == "quiz":
        payload, attachments, pending, engine, ai_usage = _mode_quiz(session, user, topic=topic, state=state)
    elif mode == "practice":
        payload, attachments, pending, engine, ai_usage = _mode_practice(session, user, topic=topic, state=state)
    elif mode == "socratic":
        payload, engine, ai_usage = _mode_socratic(session, user, text=text, topic=topic, context=context, state=state, conversation=conversation)
    elif mode == "review_answer":
        payload, pending, engine, ai_usage = _mode_review(session, user, text=text, question_id=answer_to_question_id, context=context)
    elif mode == "code_review":
        payload, engine, ai_usage = _mode_code_review(session, user, text=text, code=code, topic=topic)
    elif mode == "project_mentor":
        payload, engine, ai_usage = _mode_mentor(session, user, text=text)
    elif mode == "interview":
        payload, pending, engine, ai_usage = _mode_interview_turn(session, user, text=text, level=state["level_label"], conversation=conversation)
    elif mode == "exam_prep":
        payload, attachments, pending, engine, ai_usage = _mode_exam_prep(session, user, topic=topic, state=state)
    else:
        payload, engine, ai_usage = _mode_explain(
            session, user, text=text, topic=topic, context=context, citations=citations, state=state, grounded=grounded, style="explain"
        )

    # §5: sources are shown only when the library genuinely supports the answer. Weak matches are
    # reported separately instead of being dressed up as citations.
    sources = format_sources_for_user(citations) if grounded else []
    weak_matches = [] if grounded else [
        {"document": c.get("document", ""), "label": c.get("label", ""), "chapter": c.get("chapter", ""), "page": c.get("page")} for c in citations[:3]
    ]

    assistant = TeacherMessage(
        conversation_id=conversation.id,
        role="assistant",
        content=payload.get("text", "")[:60000],
        mode=mode,
        sources=sources,
        attachments=attachments,
        pending_item=pending,
        engine=engine,
        tokens=int(ai_usage.get("total_tokens") or 0),
    )
    session.add(assistant)
    conversation.message_count = (conversation.message_count or 0) + 1
    conversation.context = {
        **(conversation.context or {}),
        "last_engine": engine,
        "grounded": grounded,
        "kb_coverage": round(coverage, 3),
        "web_note": web_note,
    }
    session.flush()

    _remember_turn(session, user, topic=topic, mode=mode, engine=engine, grounded=grounded)
    session.commit()

    return {
        "conversation_id": conversation.id,
        "mode": mode,
        "topic": topic,
        "answer": payload.get("text", ""),
        "sections": payload.get("sections", []),
        "sources": sources,
        "weak_matches": weak_matches,
        "grounded": grounded,
        "kb_coverage": round(coverage, 3),
        "web_note": web_note,
        "conflicts": conflicts,
        "engine": engine,
        "ai": {"usage": ai_usage, "message": payload.get("ai_note", "")},
        "attachments": attachments,
        "pending": pending,
        "suggestions": payload.get("suggestions", []) or _default_suggestions(mode, topic),
        "message_id": assistant.id,
    }


def _key_term(text: str, topic: str) -> str:
    for token in reversed(re.findall(r"[A-Za-z][A-Za-z\-]{5,}", text or "")):
        return token.lower()
    return (topic or "").replace("_", " ").split(".")[-1]


# --------------------------------------------------------------------------- #
#  Modes
# --------------------------------------------------------------------------- #
def _mode_explain(
    session: Session,
    user: User,
    *,
    text: str,
    topic: str,
    context: str,
    citations: list[dict[str, Any]],
    state: dict[str, Any],
    grounded: bool,
    style: str = "explain",
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from app.ai.fallback import local_explanation
    from app.ai.manager import manager as ai_manager

    topic_label = _topic_label(session, topic) or text[:120]
    retriever = Retriever(session)
    enough, _cov, hits = retriever.sufficient(f"{topic_label} {text}", top_k=6)
    results = hits or retriever.search(f"{topic_label} {text}", top_k=6)
    local = local_explanation(topic=topic_label, level=user.level_index, results=results, style=style, ai_hint="")
    if not enough:
        local = {
            "text": _no_material_text(topic_label, results),
            "sources": [],
            "engine": "local",
            "grounded": False,
        }

    if not ai_manager.is_configured() or not user.ai_enabled:
        payload = dict(local)
        payload["ai_note"] = "Offline engine (no AI provider configured): built from your library."
        payload["suggestions"] = _default_suggestions("explain", topic)
        return payload, "local", {}

    result = ai_manager.invoke(
        "explain",
        user_id=user.id,
        kwargs={
            "topic": f"{topic_label}: {text[:900]}",
            "level": user.level_index,
            "context": context,
            "learner_state": json.dumps(state, ensure_ascii=False)[:2600],
            "style": style,
            "language": user.reply_language or "auto",
        },
        session=session,
    )
    if not result.ok:
        payload = dict(local)
        payload["ai_note"] = result.user_message()
        payload["ai_error_code"] = result.error_code
        return payload, "local", {}

    body = result.text.strip()
    if not context:
        body += "\n\n> Not grounded in your library - the knowledge base has no material on this topic, so the answer above is general knowledge."
    return (
        {"text": body, "sections": ["model", "example", "check"], "sources_used": bool(citations)},
        "ai",
        asdict(result.usage),
    )


def _mode_quiz(session: Session, user: User, *, topic: str, state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str, dict[str, Any]]:
    from app.services.learning_engine import difficulty_for
    from app.services.questions import generate_questions, question_payload

    graph = SkillGraph(session)
    skills = graph.topic_skills.get(topic, [])
    difficulty = difficulty_for(session, user.id, skills or state.get("weaknesses", []))
    questions = generate_questions(
        session,
        user,
        topic_code=topic,
        skill_codes=skills or state.get("weaknesses", []),
        count=3,
        difficulty=difficulty,
        types=["conceptual", "mcq", "open", "math"],
    )
    if not questions:
        return (
            {
                "text": "I could not build a quiz for this topic: the library has no material on it and the bank has no "
                f"question for '{topic or 'the current topic'}'. Ask about a topic you have uploaded material for, or run the diagnostic first.",
                "ai_note": "",
            },
            [],
            {},
            "local",
            {},
        )
    payload_questions = [question_payload(q) for q in questions]
    for q in questions:
        q.use_count = (q.use_count or 0) + 1
    session.flush()
    intro = (
        f"Quiz on {_topic_label(session, topic) or 'the current topic'} - {len(questions)} question(s), "
        f"difficulty {difficulty}/5. Answer them one by one with *Review my answer* (or submit via the Quiz API); "
        "I will grade correctness, depth, reasoning, precision and confidence."
    )
    return (
        {"text": intro, "questions": payload_questions, "suggestions": ["Review my answer", "Give Practice", "Deep Dive"]},
        payload_questions,
        {"kind": "quiz", "question_ids": [q.id for q in questions], "index": 0},
        "local",
        {},
    )


def _mode_practice(session: Session, user: User, *, topic: str, state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str, dict[str, Any]]:
    from app.services.practice import generate_for_topic, task_payload

    level = {1: "beginner", 2: "beginner", 3: "intermediate", 4: "intermediate", 5: "advanced"}.get(
        min(5, user.level_index), "advanced"
    ) if user.level_index < 7 else "production"
    tasks = generate_for_topic(session, user, topic_code=topic, level=level, count=2)
    if not tasks:
        return (
            {"text": "No practice available for this topic yet (no library material and no bank entry). Ask me to explain the concept first, then request practice."},
            [],
            {},
            "local",
            {},
        )
    items = [task_payload(t) for t in tasks]
    text = (
        f"Practice for {_topic_label(session, topic) or 'the current topic'} at **{level}** level:\n\n"
        + "\n\n".join(f"**{i + 1}. {item['title']}** (~{item['est_minutes']} min)\n{item['statement'][:700]}" for i, item in enumerate(items))
        + "\n\nSubmit your answer or code in Practice; I check the steps, not only the final number."
    )
    return (
        {"text": text, "suggestions": ["Review my answer", "Quiz me", "Deep Dive"]},
        items,
        {"kind": "practice", "task_ids": [t.id for t in tasks]},
        "local",
        {},
    )


def _mode_socratic(
    session: Session,
    user: User,
    *,
    text: str,
    topic: str,
    context: str,
    state: dict[str, Any],
    conversation: TeacherConversation,
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    """Never hand over the answer: ask a leading question, one step at a time."""
    from app.ai.manager import manager as ai_manager
    from app.ai.prompts import socratic_opening
    from app.knowledge.retrieval import Retriever, build_context
    from app.services.graph import SkillGraph

    history = conversation_messages(session, conversation)
    prior_turns = sum(1 for m in history if m["role"] == "assistant" and m.get("mode") == "socratic")

    results = Retriever(session).search(f"{_topic_label(session, topic)} {text}", top_k=4)
    local_scaffold = _socratic_local_steps(topic, results, state)

    if ai_manager.is_configured() and user.ai_enabled:
        result = ai_manager.invoke(
            "generate",
            user_id=user.id,
            kwargs={
                "prompt": (
                    "You are in SOCRATIC mode. Do not give the answer. Based on the learner's last message and the "
                    "history, ask exactly ONE question that moves them forward, then one short sentence on why you ask it. "
                    "If their last answer was partly right, name the piece that is right, then ask for the missing piece.\n\n"
                    f"Topic: {_topic_label(session, topic)}\nPrior socratic turns in this conversation: {prior_turns}\n"
                    f"Hidden answer key you must steer toward (do not reveal it):\n{local_scaffold}"
                ),
                "system": "You are a Socratic tutor. One question per message. Never state the target answer.",
                "context": context,
                "max_tokens": 500,
            },
            session=session,
        )
        if result.ok:
            return {"text": result.text.strip(), "suggestions": ["Review my answer", "Explain it now", "Give Practice"]}, "ai", asdict(result.usage)
        note = result.user_message()
    else:
        note = "Offline engine: the scaffolding questions below are generated from your library."

    step = local_scaffold[min(prior_turns, len(local_scaffold) - 1)] if local_scaffold else {"question": socratic_opening(_topic_label(session, topic) or topic, user.level_index), "hint": ""}
    body = f"**Step {prior_turns + 1}**\n\n{step['question']}"
    if step.get("hint"):
        body += f"\n\n_Why I ask: {step['hint']}_"
    if note:
        body += f"\n\n> {note}"
    return {"text": body, "suggestions": ["Review my answer", "Explain it now", "Give Practice"]}, "local", {}


def _socratic_local_steps(topic: str, results: list[Any], state: dict[str, Any]) -> list[dict[str, str]]:
    """Derive a question ladder from the retrieved definition/property sentences."""
    from app.knowledge.textutil import split_sentences

    topic_label = topic.replace("_", " ") if topic else "this concept"
    steps: list[dict[str, str]] = [
        {
            "question": f"Before any formula: what problem would {topic_label} solve, and what goes wrong if you skip it?",
            "hint": "If you cannot answer this one, the mechanics will not stick.",
        }
    ]
    for chunk in results[:2]:
        for sentence in split_sentences(chunk.text):
            low = sentence.lower()
            if len(steps) >= 4:
                break
            if len(sentence) > 180:
                continue
            if any(marker in low for marker in ("because", "therefore", "so that", "allows", "ensures", "if ", "only when")):
                steps.append(
                    {
                        "question": f"Your library states: “{sentence.strip()[:220]}” — what would break if that condition did not hold?",
                        "hint": "Reason from the mechanism, not from memory of the wording.",
                    }
                )
            elif "=" in sentence and len(sentence) < 150:
                steps.append(
                    {
                        "question": f"Given this relation from your material — `{sentence.strip()[:160]}` — what happens to each side when the input is scaled 10x, and why?",
                        "hint": "Scaling arguments expose whether you understand the formula or merely recognise it.",
                    }
                )
    if len(steps) < 4:
        weak = (state.get("weaknesses") or ["the prerequisite"])[0].replace("_", " ")
        steps.append({"question": f"Where exactly does your reasoning stop: is the uncertainty in {weak}, or in how it feeds into {topic_label}?", "hint": "Naming the gap is half of closing it."})
    return steps


def _mode_review(
    session: Session, user: User, *, text: str, question_id: int | None, context: str
) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    """'Review my answer' - grade + update the knowledge model through the same loop as Learn."""
    from app.services.questions import submit_answer

    question = session.get(Question, int(question_id)) if question_id else None
    if question is None:
        return (
            {
                "text": "To review an answer I need the question it belongs to. Ask 'Quiz me' first, or send the answer "
                "from the question card in Learn/Practice (that path passes the question id automatically).",
            },
            {},
            "local",
            {},
        )
    result = submit_answer(session, user, question_id=question.id, answer=text, source="teacher_review")
    verdict = result["verdict"]
    lines = [
        f"**Verdict:** {'correct' if verdict['correct'] else 'not correct yet'} — score {round(float(verdict['score']))}/100",
        "",
        f"- correctness {round(verdict['dimensions'].get('correctness', 0))} · depth {round(verdict['dimensions'].get('depth', 0))}"
        f" · reasoning {round(verdict['dimensions'].get('reasoning', 0))} · precision {round(verdict['dimensions'].get('precision', 0))}"
        f" · confidence {round(verdict['dimensions'].get('confidence', 0))}",
    ]
    if verdict.get("what_is_wrong"):
        lines += ["", f"**What is off:** {verdict['what_is_wrong']}"]
    if verdict.get("correction"):
        lines += ["", f"**Correction:** {verdict['correction']}"]
    if verdict.get("feedback"):
        lines += ["", f"**Feedback:** {verdict['feedback']}"]
    if verdict.get("missing_points"):
        lines += ["", "**Missing:** " + "; ".join(str(m)[:160] for m in verdict["missing_points"][:4])]
    actions = (result.get("learning_engine") or {}).get("actions") or []
    if actions:
        lines += ["", "**Learning Engine:**"] + [f"- {a.get('message') or a.get('type')}" for a in actions[:3]]
    if result.get("retry_question"):
        lines += ["", "**Re-check (answer this):**", result["retry_question"]["stem"]]
    lines += ["", f"_graded by: {result['engine']}_"]
    return (
        {"text": "\n".join(lines), "suggestions": ["Give Practice", "Deep Dive", "Quiz me"]},
        {"kind": "review", "question_id": question.id, "verdict": verdict},
        result["engine"],
        verdict.get("usage", {}) or {},
    )


def _mode_code_review(session: Session, user: User, *, text: str, code: str, topic: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from app.ai.manager import manager as ai_manager
    from app.services.sandbox import screen_imports

    blob = code or text
    violations = screen_imports(blob)
    lint = _python_smells(blob)
    if not ai_manager.is_configured() or not user.ai_enabled:
        body = "**Static review (offline engine - no AI configured)**\n\n"
        body += ("- Banned/blocked imports detected: " + ", ".join(violations) + "\n") if violations else "- No blocked imports.\n"
        for item in lint:
            body += f"- {item}\n"
        body += (
            "\nThe Coding Lab can still *execute* your tests locally (`/api/coding/{id}/run`), which is the strongest "
            "signal. For a reasoned review of style and structure, configure an AI provider."
        )
        return {"text": body, "suggestions": ["Run my tests", "Explain it now"]}, "local", {}
    result = ai_manager.invoke(
        "review_code",
        user_id=user.id,
        kwargs={"code": blob, "task": {"title": _topic_label(session, topic) or topic, "description": text[:1500]}, "context": ""},
        session=session,
    )
    if not result.ok:
        return {"text": f"Code review unavailable: {result.user_message()}"}, "local", {}
    data = result.data or {}
    body = f"**Verdict:** {data.get('verdict', 'n/a')} (quality {data.get('quality_score', '?')}/100)\n\n"
    for issue in (data.get("issues") or [])[:6]:
        body += f"- **{issue.get('severity')}** {issue.get('where', '')}: {issue.get('problem', '')} → {issue.get('fix', '')}\n"
    if data.get("strengths"):
        body += "\n**Good:** " + "; ".join(data["strengths"][:3]) + "\n"
    if data.get("next_step"):
        body += f"\n**Do this next:** {data['next_step']}\n"
    return {"text": body, "suggestions": ["Run my tests", "Deep Dive"]}, "ai", asdict(result.usage)


def _python_smells(code: str) -> list[str]:
    out: list[str] = []
    if "fit_transform" in code and "fit(" in code and "Pipeline" not in code:
        out.append("`fit_transform` on the full dataset before splitting leaks: fit the transform inside a Pipeline per fold.")
    if re.search(r"\.mean\(\)(?!\s*\.\s*values)", code) and "axis" not in code:
        out.append("A `mean()` without an explicit axis is ambiguous in NumPy - state which axis you reduce over.")
    if "for i in range" in code and "np." in code:
        out.append("Python loop mixed with NumPy calls - usually vectorisable; check whether it is on the hot path.")
    if ".iloc[" in code and "reset_index" not in code:
        out.append("Positional slicing plus label-based indexing later can produce alignment bugs; reset or use .loc consistently.")
    if "np.random.rand" in code or "random.random" in code:
        out.append("Unseeded randomness - pin a generator (np.random.default_rng(seed)) so the run is reproducible.")
    if "torch.rand" in code and "manual_seed" not in code:
        out.append("No torch.manual_seed: results will differ between runs.")
    if not out:
        out.append("No mechanical issues detected by the static pass.")
    return out


def _mode_mentor(session: Session, user: User, *, text: str) -> tuple[dict[str, Any], str, dict[str, Any]]:
    from app.ai.manager import manager as ai_manager
    from app.services.projects import mentor_reply

    reply = mentor_reply(session, user, text)
    return {"text": reply["text"], "suggestions": ["Review my implementation", "What are the acceptance criteria?"]}, reply.get("engine", "local"), reply.get("usage", {})


def _mode_interview_turn(session: Session, user: User, *, text: str, level: str, conversation: TeacherConversation) -> tuple[dict[str, Any], dict[str, Any], str, dict[str, Any]]:
    from app.ai.manager import manager as ai_manager

    history = conversation_messages(session, conversation, limit=24)
    if not ai_manager.is_configured() or not user.ai_enabled:
        from app.services.interview import next_bank_question

        item = next_bank_question(session, level="middle", used=[int(q) for q in (conversation.context or {}).get("asked", [])])
        if item is None:
            return {"text": "Interview mode needs either an AI provider or questions in the bank. Run the diagnostic or configure AI."}, {}, "local", {}
        asked = list((conversation.context or {}).get("asked", [])) + [item.id]
        conversation.context = {**(conversation.context or {}), "asked": asked}
        session.flush()
        return (
            {
                "text": f"**Interviewer (offline, question bank · {level} track):**\n\n{item.stem}\n\nAnswer, and I will probe the weakest part of your reply.",
                "suggestions": ["Answer", "Ask a harder one", "Finish interview"],
            },
            {"kind": "interview_turn", "question_id": item.id},
            "local",
            {},
        )
    result = ai_manager.invoke(
        "interview",
        user_id=user.id,
        kwargs={"transcript": history, "stage": "turn", "level": level, "context": ""},
        session=session,
        use_cache=False,
    )
    if not result.ok:
        return {"text": f"Interview turn failed: {result.user_message()}"}, {}, "local", {}
    return {"text": result.text.strip(), "suggestions": ["Answer", "Finish interview"]}, {"kind": "interview_turn"}, "ai", asdict(result.usage)


def _mode_exam_prep(session: Session, user: User, *, topic: str, state: dict[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any], str, dict[str, Any]]:
    from app.services.exams import exam_readiness, list_exams

    readiness = exam_readiness(session, user)
    lines = ["**Exam readiness** (from your skill model, not from attendance):\n"]
    for row in readiness:
        lines.append(
            f"- {row['exam']}: {row['readiness']}/100 readiness, {row['covered']}/{row['total']} required skills at or above the bar"
            + (f" — {row['advice']}" if row.get("advice") else "")
        )
    if not ai_available():
        lines.append("\n_Offline engine: full mock exams with AI-graded open answers require an AI provider; objective sections work now._")
    weak = state.get("weaknesses") or []
    if weak:
        lines.append("\n**Attack these first:** " + ", ".join(weak[:5]))
    exam_list = list_exams(session)
    lines.append("\nAvailable: " + ", ".join(f"`{e['code']}`" for e in exam_list))
    return (
        {"text": "\n".join(lines), "suggestions": ["Quiz me", "Give Practice", "Start the Middle exam"]},
        [],
        {"kind": "exam_prep"},
        "local",
        {},
    )


def ai_available(session: Session | None = None) -> bool:
    from app.ai.manager import manager as ai_manager

    return ai_manager.is_configured()


# --------------------------------------------------------------------------- #
#  helpers
# --------------------------------------------------------------------------- #
def _no_material_text(topic: str, results: list[Any]) -> str:
    from app.ai.fallback import _no_material_message

    return _no_material_message(topic, results)


def _topic_label(session: Session, topic_code: str) -> str:
    if not topic_code:
        return ""
    topic = session.execute(select(Topic).where(Topic.code == topic_code)).scalar_one_or_none()
    if topic is not None:
        return topic.name
    graph = SkillGraph(session)
    node = graph.nodes.get(topic_code)
    return node.name if node else ""


def _remember_turn(session: Session, user: User, *, topic: str, mode: str, engine: str, grounded: bool) -> None:
    """Long-term learning memory (§34) - compact, no raw secrets, no book contents."""
    if not topic:
        return
    key = f"topic:{topic}"
    row = session.execute(
        select(LearningMemory).where(LearningMemory.user_id == user.id, LearningMemory.category == "topic", LearningMemory.key == key)
    ).scalar_one_or_none()
    stamp = datetime.utcnow().isoformat(timespec="seconds")
    if row is None:
        session.add(
            LearningMemory(
                user_id=user.id,
                category="topic",
                key=key,
                value=f"studied {topic}",
                payload={"touches": 1, "modes": [mode], "engines": [engine], "grounded": grounded, "last": stamp},
                weight=1.0,
            )
        )
    else:
        payload = dict(row.payload or {})
        payload["touches"] = int(payload.get("touches", 0)) + 1
        modes = list(payload.get("modes", []))
        if mode not in modes:
            modes.append(mode)
        payload["modes"] = modes[-6:]
        payload["last"] = stamp
        payload["grounded"] = bool(payload.get("grounded")) or grounded
        row.payload = payload
        row.weight = round(float(row.weight or 1.0) + 0.2, 2)
        row.updated_at = datetime.utcnow()
    session.flush()


def _default_suggestions(mode: str, topic: str) -> list[str]:
    if mode == "explain":
        return ["Quiz me", "Give Practice", "Deep Dive", "Socratic Mode"]
    if mode == "socratic":
        return ["Review my answer", "Explain it now"]
    return ["Explain", "Quiz me", "Give Practice", "Deep Dive"]


def list_conversations(session: Session, user: User, *, limit: int = 30) -> list[dict[str, Any]]:
    rows = session.execute(
        select(TeacherConversation).where(TeacherConversation.user_id == user.id).order_by(TeacherConversation.updated_at.desc()).limit(limit)
    ).scalars()
    return [
        {
            "id": c.id,
            "title": c.title,
            "mode": c.mode,
            "topic": c.topic_code,
            "messages": c.message_count,
            "updated": c.updated_at.isoformat() if c.updated_at else None,
            "context": c.context or {},
        }
        for c in rows
    ]


def get_conversation(session: Session, user: User, conversation_id: int) -> dict[str, Any] | None:
    conv = session.get(TeacherConversation, conversation_id)
    if conv is None or conv.user_id != user.id:
        return None
    messages = session.execute(
        select(TeacherMessage).where(TeacherMessage.conversation_id == conv.id).order_by(TeacherMessage.id.asc())
    ).scalars()
    return {
        "id": conv.id,
        "title": conv.title,
        "mode": conv.mode,
        "topic": conv.topic_code,
        "context": conv.context or {},
        "messages": [
            {
                "id": m.id,
                "role": m.role,
                "content": m.content,
                "mode": m.mode,
                "sources": m.sources or [],
                "attachments": m.attachments or [],
                "pending": m.pending_item or {},
                "engine": m.engine,
                "created": m.created_at.isoformat() if m.created_at else None,
            }
            for m in messages
        ],
    }
