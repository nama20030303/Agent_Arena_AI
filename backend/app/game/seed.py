"""Idempotent seeding of the Knowledge Universe content."""

from __future__ import annotations

import hashlib
import logging

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.game import content
from app.game.models import GameAchievement, GameEdge, GameItem, GameNode, GameQuestion, GameWorld

logger = logging.getLogger(__name__)


def _hash(*parts: str) -> str:
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()


def seed_game(session: Session) -> dict[str, int]:
    stats = {"worlds": 0, "nodes": 0, "edges": 0, "questions": 0, "items": 0, "achievements": 0}

    existing_worlds = {w.code: w for w in session.execute(select(GameWorld)).scalars()}
    for code, name, tagline, color, glyph, order in content.WORLDS:
        w = existing_worlds.get(code)
        if w is None:
            session.add(GameWorld(code=code, name_ru=name, tagline_ru=tagline, color=color, glyph=glyph, order_index=order))
            stats["worlds"] += 1
        else:
            w.name_ru, w.tagline_ru, w.color, w.glyph, w.order_index = name, tagline, color, glyph, order

    existing_nodes = {n.code: n for n in session.execute(select(GameNode)).scalars()}
    for idx, (world, code, name, short, diff, xp, boss, root, (x, y), theory) in enumerate(content.NODES):
        n = existing_nodes.get(code)
        if n is None:
            session.add(
                GameNode(
                    world_code=world, code=code, name_ru=name, short_ru=short, theory_ru=theory.strip(),
                    difficulty=diff, xp_reward=xp, is_boss=boss, is_root=root, pos_x=x, pos_y=y, order_index=idx,
                )
            )
            stats["nodes"] += 1
        else:
            n.world_code, n.name_ru, n.short_ru, n.theory_ru = world, name, short, theory.strip()
            n.difficulty, n.xp_reward, n.is_boss, n.is_root = diff, xp, boss, root
            n.pos_x, n.pos_y, n.order_index = x, y, idx

    existing_edges = {(e.from_code, e.to_code) for e in session.execute(select(GameEdge)).scalars()}
    for frm, to in content.EDGES:
        if (frm, to) not in existing_edges:
            session.add(GameEdge(from_code=frm, to_code=to))
            stats["edges"] += 1

    existing_q = {q.content_hash for q in session.execute(select(GameQuestion)).scalars()}
    for node_code, questions in content.QUESTIONS.items():
        for q in questions:
            h = _hash(node_code, q["q"])
            if h in existing_q:
                continue
            session.add(
                GameQuestion(
                    node_code=node_code,
                    qtype=q["t"],
                    prompt_ru=q["q"],
                    options=q.get("o", []),
                    correct_index=q.get("c"),
                    expected=str(q.get("a", "")),
                    tolerance=float(q.get("tol", 0.0)),
                    keywords=q.get("kw", []),
                    explanation_ru=q.get("e", ""),
                    hints=q.get("h", []),
                    difficulty=int(q.get("d", 1)),
                    content_hash=h,
                )
            )
            stats["questions"] += 1

    existing_items = {i.code for i in session.execute(select(GameItem)).scalars()}
    for code, name, desc, rarity, kind, world, glyph in content.ITEMS:
        if code not in existing_items:
            session.add(GameItem(code=code, name_ru=name, description_ru=desc, rarity=rarity, kind=kind, world_code=world, glyph=glyph))
            stats["items"] += 1

    existing_ach = {a.code for a in session.execute(select(GameAchievement)).scalars()}
    for code, name, desc, glyph, xp, secret in content.ACHIEVEMENTS:
        if code not in existing_ach:
            session.add(GameAchievement(code=code, name_ru=name, description_ru=desc, glyph=glyph, xp_reward=xp, secret=secret))
            stats["achievements"] += 1

    session.flush()
    if any(stats.values()):
        logger.info("game seed: %s", stats)
    return stats
