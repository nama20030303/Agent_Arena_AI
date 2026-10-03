# -*- coding: utf-8 -*-
"""
AI Teacher for the Knowledge Universe.

Uses the provider-agnostic AI gateway (app.ai.manager). When no provider is
configured the teacher degrades to a deterministic Russian engine built on the
seeded node theory — the product stays fully usable offline (§61, §128).
"""

from __future__ import annotations

import re
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.ai.manager import get_ai_manager
from app.game.models import GameChatMessage, GameNode, GameQuestion, GameWorld, PlayerNode, PlayerState

MODES = {
    "explain": "Объясни концепцию просто и точно, с примером.",
    "socratic": "Не давай готовый ответ: веди ученика наводящими вопросами.",
    "practice": "Предложи ученику задачу по теме и жди его решения.",
    "challenge": "Брось вызов: задай вопрос чуть выше текущего уровня ученика.",
    "review": "Помоги повторить материал: кратко напомни суть и проверь понимание.",
}

SYSTEM_PROMPT = """Ты — Учитель в образовательной вселенной знаний. Твой ученик — {rank} {level} уровня.
Правила:
- Отвечай ТОЛЬКО на русском языке.
- Будь точным, доброжелательным и интеллектуально честным. Если ученик ошибается — объясни, почему, никогда не соглашайся из вежливости.
- Пиши компактно: короткие абзацы, списки, примеры. Не лей воду.
- Режим: {mode_instruction}
{context}"""


def _paragraphs(text: str) -> list[str]:
    return [p.strip() for p in re.split(r"\n\s*\n", text or "") if p.strip()]


def _node_context(session: Session, node_code: str) -> tuple[GameNode | None, str]:
    if not node_code:
        return None, ""
    node = session.execute(select(GameNode).where(GameNode.code == node_code)).scalar_one_or_none()
    if node is None:
        return None, ""
    world = session.execute(select(GameWorld).where(GameWorld.code == node.world_code)).scalar_one_or_none()
    ctx = (
        f"- Текущая тема: «{node.name_ru}» (мир: {world.name_ru if world else node.world_code}).\n"
        f"- Материал темы:\n{node.theory_ru.strip()[:2600]}"
    )
    return node, "Контекст:\n" + ctx


# --------------------------------------------------------------------------- #
#  Deterministic offline engine
# --------------------------------------------------------------------------- #
_WORD_RE = re.compile(r"[а-яёa-z0-9]{3,}", re.IGNORECASE)


def _tokens(text: str) -> set[str]:
    return {w.lower().replace("ё", "е") for w in _WORD_RE.findall(text or "")}


def _find_best_node(session: Session, message: str) -> GameNode | None:
    msg = _tokens(message)
    if not msg:
        return None
    best, best_score = None, 0.0
    for node in session.execute(select(GameNode)).scalars():
        hay = _tokens(node.name_ru + " " + node.short_ru + " " + node.theory_ru[:800])
        score = len(msg & hay)
        # name match is worth more
        score += 2 * len(msg & _tokens(node.name_ru))
        if score > best_score:
            best, best_score = node, score
    return best if best_score >= 2 else best


def _offline_reply(session: Session, ps: PlayerState, message: str, node_code: str, mode: str) -> str:
    node, _ = _node_context(session, node_code)
    if node is None:
        node = _find_best_node(session, message)

    if node is None:
        return (
            "Хороший вопрос! Пока я работаю в автономном режиме и опираюсь на карту знаний.\n\n"
            "Выбери узел на карте — и я разберу его с тобой по шагам: объясню теорию, "
            "дам подсказки и проверю понимание. Или спроси меня о конкретной теме: "
            "например, «циклы», «второй закон Ньютона» или «проценты»."
        )

    paras = _paragraphs(node.theory_ru)
    if mode == "socratic":
        q = session.execute(select(GameQuestion).where(GameQuestion.node_code == node.code)).scalars().first()
        lead = paras[0] if paras else node.short_ru
        question = q.hints[0] if (q and q.hints) else "С чего, по-твоему, стоит начать рассуждение?"
        return f"Давай подумаем вместе над темой «{node.name_ru}».\n\n{lead}\n\nМой вопрос к тебе: {question}"
    if mode == "practice":
        q = session.execute(select(GameQuestion).where(GameQuestion.node_code == node.code)).scalars().first()
        if q is not None:
            return (
                f"Задача по теме «{node.name_ru}»:\n\n{q.prompt_ru}\n\n"
                "Напиши свой ответ — а проверить его можно в самом узле на карте знаний."
            )
    if mode == "challenge":
        tail = paras[-1] if paras else node.short_ru
        return (
            f"Испытание по теме «{node.name_ru}».\n\n{tail}\n\n"
            "Попробуй объяснить это своими словами — как будто рассказываешь другу. "
            "Затем открой узел на карте и докажи понимание на задачах."
        )

    body = "\n\n".join(paras[:3]) if paras else node.short_ru
    return f"Разберём тему «{node.name_ru}».\n\n{body}\n\nЕсли что-то осталось неясным — спроси, объясню иначе."


# --------------------------------------------------------------------------- #
#  Public entrypoint
# --------------------------------------------------------------------------- #
def teacher_reply(
    session: Session,
    ps: PlayerState,
    *,
    message: str,
    node_code: str = "",
    mode: str = "explain",
    rank: str = "Новичок",
    level: int = 1,
) -> dict[str, Any]:
    mode = mode if mode in MODES else "explain"
    session.add(GameChatMessage(user_id=ps.user_id, role="user", mode=mode, node_code=node_code, text=message))

    manager = get_ai_manager()
    text, via_ai = "", False
    if manager.is_configured():
        _, ctx = _node_context(session, node_code)
        system = SYSTEM_PROMPT.format(rank=rank, level=level, mode_instruction=MODES[mode], context=ctx)
        history = list(
            session.execute(
                select(GameChatMessage)
                .where(GameChatMessage.user_id == ps.user_id)
                .order_by(GameChatMessage.id.desc())
                .limit(8)
            ).scalars()
        )[::-1]
        messages = [{"role": "system", "content": system}]
        for m in history:
            messages.append({"role": "user" if m.role == "user" else "assistant", "content": m.text[:2000]})
        result = manager.invoke(
            "complete",
            user_id=ps.user_id,
            kwargs={"messages": messages, "max_tokens": 700},
            session=session,
            use_cache=False,
        )
        if result.ok and result.text.strip():
            text, via_ai = result.text.strip(), True

    if not text:
        text = _offline_reply(session, ps, message, node_code, mode)

    session.add(GameChatMessage(user_id=ps.user_id, role="teacher", mode=mode, node_code=node_code, text=text, via_ai=via_ai))
    session.flush()
    return {"text": text, "via_ai": via_ai, "mode": mode}
