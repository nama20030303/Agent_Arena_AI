"""
Game engine: progression, evaluation, economy. Single place where XP, levels,
chests, quests and achievements are produced — always on the server (§74).

Every grant goes through `_grant_xp` / `_grant_chest` / `_unlock_achievement`
and leaves a GameEvent row, so the whole economy is auditable (§73).
"""

from __future__ import annotations

import random
import re
from datetime import date, datetime, timedelta
from typing import Any

from sqlalchemy import func, select
from sqlalchemy.orm import Session

from app.game import content
from app.game.models import (
    GameAchievement,
    GameEdge,
    GameEvent,
    GameItem,
    GameNode,
    GameQuestion,
    GameWorld,
    PlayerAchievement,
    PlayerChest,
    PlayerItem,
    PlayerNode,
    PlayerQuest,
    PlayerQuestionState,
    PlayerState,
)

RARITY_XP = {"common": 10, "uncommon": 15, "rare": 25, "epic": 40, "legendary": 70, "mythic": 120}


# --------------------------------------------------------------------------- #
#  Levels & ranks
# --------------------------------------------------------------------------- #
def xp_for_level(level: int) -> int:
    """Total XP required to *reach* `level` (level 1 = 0 XP)."""
    if level <= 1:
        return 0
    return int(sum(60 + 28 * (n - 1) ** 1.35 for n in range(2, level + 1)))


def level_from_xp(xp: int) -> int:
    level = 1
    while xp >= xp_for_level(level + 1):
        level += 1
        if level > 99:
            break
    return level


def rank_for_level(level: int) -> str:
    rank = content.RANKS[0][1]
    for threshold, title in content.RANKS:
        if level >= threshold:
            rank = title
    return rank


def level_payload(xp: int) -> dict[str, Any]:
    level = level_from_xp(xp)
    cur, nxt = xp_for_level(level), xp_for_level(level + 1)
    return {
        "level": level,
        "rank": rank_for_level(level),
        "xp": xp,
        "xp_into_level": xp - cur,
        "xp_for_next": nxt - cur,
        "next_rank": next((t for lv, t in content.RANKS if lv > level), None),
    }


# --------------------------------------------------------------------------- #
#  Player bootstrap
# --------------------------------------------------------------------------- #
def ensure_player(session: Session, user_id: int) -> PlayerState:
    ps = session.execute(select(PlayerState).where(PlayerState.user_id == user_id)).scalar_one_or_none()
    if ps is None:
        ps = PlayerState(user_id=user_id)
        session.add(ps)
        session.flush()
    _ensure_player_nodes(session, user_id)
    return ps


def _ensure_player_nodes(session: Session, user_id: int) -> None:
    existing = {
        pn.node_code for pn in session.execute(select(PlayerNode).where(PlayerNode.user_id == user_id)).scalars()
    }
    for node in session.execute(select(GameNode)).scalars():
        if node.code in existing:
            continue
        status = PlayerNode.ST_AVAILABLE if node.is_root else PlayerNode.ST_LOCKED
        session.add(PlayerNode(user_id=user_id, node_code=node.code, status=status))
    session.flush()


def _event(session: Session, user_id: int, etype: str, payload: dict[str, Any]) -> None:
    session.add(GameEvent(user_id=user_id, etype=etype, payload=payload))


# --------------------------------------------------------------------------- #
#  Streak
# --------------------------------------------------------------------------- #
def touch_streak(session: Session, ps: PlayerState) -> dict[str, Any] | None:
    """Call on any meaningful learning action. Returns info when streak changes."""
    today = date.today()
    if ps.last_active_day == today:
        return None
    changed: dict[str, Any] | None = None
    if ps.last_active_day == today - timedelta(days=1):
        ps.streak_days += 1
        changed = {"streak": ps.streak_days, "kind": "extended"}
    elif ps.last_active_day == today - timedelta(days=2) and ps.streak_shields > 0:
        ps.streak_shields -= 1
        ps.streak_days += 1
        changed = {"streak": ps.streak_days, "kind": "shield_used"}
    else:
        new_streak = 1
        changed = {"streak": new_streak, "kind": "reset" if ps.streak_days > 1 else "started"}
        ps.streak_days = new_streak
    ps.best_streak = max(ps.best_streak, ps.streak_days)
    ps.last_active_day = today
    _event(session, ps.user_id, "streak.updated", changed or {})
    return changed


# --------------------------------------------------------------------------- #
#  XP / levels
# --------------------------------------------------------------------------- #
def _grant_xp(session: Session, ps: PlayerState, amount: int, reason: str, fx: dict[str, Any]) -> None:
    if amount <= 0:
        return
    before = level_from_xp(ps.xp)
    ps.xp += amount
    after = level_from_xp(ps.xp)
    ps.level = after
    fx.setdefault("xp_gained", 0)
    fx["xp_gained"] += amount
    fx.setdefault("xp_reasons", []).append({"reason": reason, "amount": amount})
    _event(session, ps.user_id, "xp.granted", {"amount": amount, "reason": reason, "total": ps.xp})
    if after > before:
        fx["level_up"] = {"from": before, "to": after, "rank": rank_for_level(after)}
        _event(session, ps.user_id, "level.up", {"from": before, "to": after})
        # every level-up grants a chest at levels 2,3 then every 2 levels
        if after <= 3 or after % 2 == 0:
            _grant_chest(session, ps, source="level", world_code="", fx=fx)


# --------------------------------------------------------------------------- #
#  Chests
# --------------------------------------------------------------------------- #
def _grant_chest(session: Session, ps: PlayerState, *, source: str, world_code: str, fx: dict[str, Any]) -> PlayerChest:
    chest = PlayerChest(user_id=ps.user_id, source=source, world_code=world_code)
    session.add(chest)
    session.flush()
    fx.setdefault("chests_granted", []).append({"id": chest.id, "source": source, "world": world_code})
    _event(session, ps.user_id, "chest.received", {"chest_id": chest.id, "source": source})
    return chest


def _roll_rarity(source: str, rng: random.Random) -> str:
    weights = dict(content.RARITY_WEIGHTS)
    if source in {"boss", "quest"}:
        # boss & quest chests: shift probability mass upward
        weights = {"common": 8, "uncommon": 22, "rare": 32, "epic": 24, "legendary": 11, "mythic": 3}
    if source == "onboarding":
        weights = {"common": 0, "uncommon": 55, "rare": 35, "epic": 10, "legendary": 0, "mythic": 0}
    order = content.RARITY_ORDER
    total = sum(weights.get(r, 0) for r in order)
    pick = rng.uniform(0, total)
    acc = 0.0
    for r in order:
        acc += weights.get(r, 0)
        if pick <= acc:
            return r
    return "common"


def open_chest(session: Session, ps: PlayerState, chest_id: int) -> dict[str, Any]:
    chest = session.get(PlayerChest, chest_id)
    if chest is None or chest.user_id != ps.user_id:
        raise ValueError("Сундук не найден.")
    if chest.opened:
        raise ValueError("Этот сундук уже открыт.")
    rng = random.SystemRandom()
    rarity = _roll_rarity(chest.source, rng)

    items = list(session.execute(select(GameItem).where(GameItem.rarity == rarity)).scalars())
    preferred = [i for i in items if i.world_code == chest.world_code] or items
    owned = {
        pi.item_code for pi in session.execute(select(PlayerItem).where(PlayerItem.user_id == ps.user_id)).scalars()
    }
    fresh = [i for i in preferred if i.code not in owned]
    item = rng.choice(fresh or preferred or items)

    xp_bonus = RARITY_XP.get(rarity, 10)
    chest.opened = True
    chest.rarity = rarity
    chest.item_code = item.code
    chest.xp_bonus = xp_bonus
    chest.opened_at = datetime.utcnow()
    ps.chests_opened += 1

    inv = session.execute(
        select(PlayerItem).where(PlayerItem.user_id == ps.user_id, PlayerItem.item_code == item.code)
    ).scalar_one_or_none()
    duplicate = inv is not None
    if inv is None:
        session.add(PlayerItem(user_id=ps.user_id, item_code=item.code))
    else:
        inv.count += 1

    fx: dict[str, Any] = {}
    _grant_xp(session, ps, xp_bonus, f"chest:{rarity}", fx)
    _event(session, ps.user_id, "chest.opened", {"chest_id": chest.id, "rarity": rarity, "item": item.code})
    _check_achievements(session, ps, fx)
    session.flush()
    return {
        "rarity": rarity,
        "item": _item_payload(item),
        "duplicate": duplicate,
        "xp_bonus": xp_bonus,
        "fx": fx,
    }


def _item_payload(item: GameItem) -> dict[str, Any]:
    return {
        "code": item.code,
        "name": item.name_ru,
        "description": item.description_ru,
        "rarity": item.rarity,
        "kind": item.kind,
        "world": item.world_code,
        "glyph": item.glyph,
    }


# --------------------------------------------------------------------------- #
#  Daily quests
# --------------------------------------------------------------------------- #
def ensure_daily_quests(session: Session, ps: PlayerState) -> list[PlayerQuest]:
    today = date.today()
    quests = list(
        session.execute(select(PlayerQuest).where(PlayerQuest.user_id == ps.user_id, PlayerQuest.day == today)).scalars()
    )
    if quests:
        return quests
    rng = random.Random(f"{ps.user_id}:{today.isoformat()}")
    templates = list(content.QUEST_TEMPLATES)
    picked = rng.sample(templates, 3)
    intensity = ps.intensity if ps.intensity in {"light", "normal", "deep"} else "normal"
    for code, title_tpl, targets, xp, chest in picked:
        target = targets[intensity]
        session.add(
            PlayerQuest(
                user_id=ps.user_id,
                day=today,
                code=code,
                title_ru=title_tpl.format(n=target),
                target=target,
                xp_reward=xp,
                chest_on_complete=chest,
            )
        )
    session.flush()
    return list(
        session.execute(select(PlayerQuest).where(PlayerQuest.user_id == ps.user_id, PlayerQuest.day == today)).scalars()
    )


def _bump_quest(session: Session, ps: PlayerState, code: str, fx: dict[str, Any], inc: int = 1, *, set_to: int | None = None) -> None:
    today = date.today()
    quest = session.execute(
        select(PlayerQuest).where(PlayerQuest.user_id == ps.user_id, PlayerQuest.day == today, PlayerQuest.code == code)
    ).scalar_one_or_none()
    if quest is None or quest.completed:
        return
    quest.progress = max(quest.progress, set_to) if set_to is not None else quest.progress + inc
    if quest.progress >= quest.target:
        quest.progress = quest.target
        quest.completed = True
        quest.claimed = True
        fx.setdefault("quests_completed", []).append({"code": quest.code, "title": quest.title_ru})
        _event(session, ps.user_id, "quest.completed", {"code": quest.code})
        _grant_xp(session, ps, quest.xp_reward, f"quest:{quest.code}", fx)
        if quest.chest_on_complete:
            _grant_chest(session, ps, source="quest", world_code="", fx=fx)


# --------------------------------------------------------------------------- #
#  Achievements
# --------------------------------------------------------------------------- #
def _unlock_achievement(session: Session, ps: PlayerState, code: str, fx: dict[str, Any]) -> None:
    exists = session.execute(
        select(PlayerAchievement).where(PlayerAchievement.user_id == ps.user_id, PlayerAchievement.achievement_code == code)
    ).scalar_one_or_none()
    if exists:
        return
    ach = session.execute(select(GameAchievement).where(GameAchievement.code == code)).scalar_one_or_none()
    if ach is None:
        return
    session.add(PlayerAchievement(user_id=ps.user_id, achievement_code=code))
    fx.setdefault("achievements", []).append({"code": ach.code, "name": ach.name_ru, "glyph": ach.glyph, "xp": ach.xp_reward})
    _event(session, ps.user_id, "achievement.unlocked", {"code": code})
    _grant_xp(session, ps, ach.xp_reward, f"achievement:{code}", fx)


def _check_achievements(session: Session, ps: PlayerState, fx: dict[str, Any]) -> None:
    session.flush()  # autoflush=False: make pending node/item changes visible to counts
    if ps.answers_correct >= 1:
        _unlock_achievement(session, ps, "ach-first-step", fx)
    if ps.answers_correct >= 10:
        _unlock_achievement(session, ps, "ach-correct-10", fx)
    if ps.answers_correct >= 50:
        _unlock_achievement(session, ps, "ach-correct-50", fx)
    if ps.chests_opened >= 1:
        _unlock_achievement(session, ps, "ach-first-chest", fx)
    if ps.bosses_defeated >= 1:
        _unlock_achievement(session, ps, "ach-first-boss", fx)
    if ps.streak_days >= 3:
        _unlock_achievement(session, ps, "ach-streak-3", fx)
    if ps.streak_days >= 7:
        _unlock_achievement(session, ps, "ach-streak-7", fx)
    if ps.level >= 5:
        _unlock_achievement(session, ps, "ach-level-5", fx)
    if ps.level >= 10:
        _unlock_achievement(session, ps, "ach-level-10", fx)

    mastered = session.execute(
        select(func.count()).select_from(PlayerNode).where(PlayerNode.user_id == ps.user_id, PlayerNode.status == PlayerNode.ST_MASTERED)
    ).scalar_one()
    if mastered >= 1:
        _unlock_achievement(session, ps, "ach-first-node", fx)

    items_count = session.execute(
        select(func.count()).select_from(PlayerItem).where(PlayerItem.user_id == ps.user_id)
    ).scalar_one()
    if items_count >= 5:
        _unlock_achievement(session, ps, "ach-collector-5", fx)

    # world coverage
    nodes_by_world: dict[str, list[GameNode]] = {}
    for node in session.execute(select(GameNode)).scalars():
        nodes_by_world.setdefault(node.world_code, []).append(node)
    mastered_codes = {
        pn.node_code
        for pn in session.execute(
            select(PlayerNode).where(PlayerNode.user_id == ps.user_id, PlayerNode.status == PlayerNode.ST_MASTERED)
        ).scalars()
    }
    for world, nodes in nodes_by_world.items():
        done = sum(1 for n in nodes if n.code in mastered_codes)
        if done * 2 >= len(nodes) and done > 0:
            _unlock_achievement(session, ps, "ach-world-half", fx)
        if done == len(nodes):
            _unlock_achievement(session, ps, "ach-world-full", fx)

    teacher_msgs = session.execute(
        select(func.count()).select_from(GameEvent).where(GameEvent.user_id == ps.user_id, GameEvent.etype == "teacher.message")
    ).scalar_one()
    if teacher_msgs >= 10:
        _unlock_achievement(session, ps, "ach-teacher-10", fx)


# --------------------------------------------------------------------------- #
#  Answer evaluation (server-side, no trust in the client)
# --------------------------------------------------------------------------- #
_NUM_RE = re.compile(r"^-?\d+(?:[.,]\d+)?$")


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip().lower()).replace("ё", "е")


def _check_answer(q: GameQuestion, answer: str, option: int | None) -> bool:
    if q.qtype == "mcq":
        return option is not None and option == q.correct_index
    expected = (q.expected or "").strip()
    got = (answer or "").strip()
    # numeric comparison with tolerance
    exp_norm = expected.replace(",", ".")
    got_norm = got.replace(",", ".")
    if _NUM_RE.match(exp_norm) and _NUM_RE.match(got_norm):
        try:
            return abs(float(exp_norm) - float(got_norm)) <= max(q.tolerance, 1e-9)
        except ValueError:
            pass
    # fraction like "5/6"
    if "/" in expected and "/" in got:
        return _normalize(got).replace(" ", "") == _normalize(expected).replace(" ", "")
    # text comparison
    norm_expected = _normalize(expected).replace(" ", "")
    norm_got = _normalize(got).replace(" ", "")
    if norm_expected == norm_got:
        return True
    if q.keywords:
        hits = sum(1 for kw in q.keywords if _normalize(kw) in _normalize(got))
        return hits >= max(1, len(q.keywords) // 2)
    return False


def submit_answer(
    session: Session,
    ps: PlayerState,
    question_id: int,
    *,
    answer: str = "",
    option: int | None = None,
    hints_used: int = 0,
) -> dict[str, Any]:
    q = session.get(GameQuestion, question_id)
    if q is None:
        raise ValueError("Задание не найдено.")
    node = session.execute(select(GameNode).where(GameNode.code == q.node_code)).scalar_one()
    pn = session.execute(
        select(PlayerNode).where(PlayerNode.user_id == ps.user_id, PlayerNode.node_code == q.node_code)
    ).scalar_one_or_none()
    if pn is None or pn.status == PlayerNode.ST_LOCKED:
        raise ValueError("Этот узел знаний ещё не открыт.")

    qs = session.execute(
        select(PlayerQuestionState).where(
            PlayerQuestionState.user_id == ps.user_id, PlayerQuestionState.question_id == q.id
        )
    ).scalar_one_or_none()
    if qs is None:
        qs = PlayerQuestionState(user_id=ps.user_id, question_id=q.id)
        session.add(qs)
        session.flush()

    already_solved = qs.solved
    correct = _check_answer(q, answer, option)
    qs.attempts += 1
    qs.hints_used = max(qs.hints_used, min(hints_used, len(q.hints)))
    qs.last_answer = answer if answer else (str(option) if option is not None else "")

    fx: dict[str, Any] = {}
    streak_info = touch_streak(session, ps)
    if streak_info:
        fx["streak"] = streak_info

    ps.answers_total += 1
    pn.attempts += 1
    if pn.status == PlayerNode.ST_AVAILABLE:
        pn.status = PlayerNode.ST_ACTIVE

    node_result: dict[str, Any] = {}
    if correct:
        ps.answers_correct += 1
        pn.correct += 1
        if not already_solved:
            qs.solved = True
            base = 8 + 4 * max(1, q.difficulty)
            penalty = 2 * min(hints_used, 3)
            _grant_xp(session, ps, max(4, base - penalty), "answer.correct", fx)
            _bump_quest(session, ps, "solve", fx)
            _bump_quest(session, ps, "perfect", fx)
            node_result = _update_node_progress(session, ps, node, pn, fx)
    else:
        # perfect-run quest resets on a wrong answer
        today = date.today()
        quest = session.execute(
            select(PlayerQuest).where(
                PlayerQuest.user_id == ps.user_id, PlayerQuest.day == today, PlayerQuest.code == "perfect"
            )
        ).scalar_one_or_none()
        if quest is not None and not quest.completed:
            quest.progress = 0

    _check_achievements(session, ps, fx)
    _recompute_power(session, ps)
    session.flush()

    return {
        "correct": correct,
        "already_solved": already_solved,
        "explanation": q.explanation_ru if (correct or qs.attempts >= 2) else "",
        "fx": fx,
        "node": node_result,
        "state": public_state(session, ps),
    }


def _update_node_progress(session: Session, ps: PlayerState, node: GameNode, pn: PlayerNode, fx: dict[str, Any]) -> dict[str, Any]:
    session.flush()  # the app session runs with autoflush=False
    q_ids = [q.id for q in session.execute(select(GameQuestion).where(GameQuestion.node_code == node.code)).scalars()]
    solved = session.execute(
        select(func.count())
        .select_from(PlayerQuestionState)
        .where(
            PlayerQuestionState.user_id == ps.user_id,
            PlayerQuestionState.question_id.in_(q_ids),
            PlayerQuestionState.solved.is_(True),
        )
    ).scalar_one()
    total = max(1, len(q_ids))
    pn.progress = round(100.0 * solved / total, 1)
    result: dict[str, Any] = {"code": node.code, "progress": pn.progress, "mastered": False, "unlocked": []}

    if solved >= total and pn.status != PlayerNode.ST_MASTERED:
        pn.status = PlayerNode.ST_MASTERED
        pn.mastered_at = datetime.utcnow()
        result["mastered"] = True
        _event(session, ps.user_id, "concept.mastered", {"node": node.code})
        _grant_xp(session, ps, node.xp_reward, f"node:{node.code}", fx)
        _bump_quest(session, ps, "master", fx)
        if node.is_boss:
            ps.bosses_defeated += 1
            _event(session, ps.user_id, "boss.defeated", {"node": node.code})
            _grant_chest(session, ps, source="boss", world_code=node.world_code, fx=fx)
        else:
            _grant_chest(session, ps, source="node", world_code=node.world_code, fx=fx)
        # perfect node: no wrong attempts at all
        if pn.attempts == pn.correct:
            _unlock_achievement(session, ps, "ach-perfect-node", fx)
        result["unlocked"] = _unlock_children(session, ps, node.code)
        for code in result["unlocked"]:
            _event(session, ps.user_id, "node.unlocked", {"node": code})
    return result


def _unlock_children(session: Session, ps: PlayerState, mastered_code: str) -> list[str]:
    unlocked: list[str] = []
    mastered = {
        pn.node_code
        for pn in session.execute(
            select(PlayerNode).where(PlayerNode.user_id == ps.user_id, PlayerNode.status == PlayerNode.ST_MASTERED)
        ).scalars()
    }
    children = [e.to_code for e in session.execute(select(GameEdge).where(GameEdge.from_code == mastered_code)).scalars()]
    for child in children:
        prereqs = [e.from_code for e in session.execute(select(GameEdge).where(GameEdge.to_code == child)).scalars()]
        if all(p in mastered for p in prereqs):
            pn = session.execute(
                select(PlayerNode).where(PlayerNode.user_id == ps.user_id, PlayerNode.node_code == child)
            ).scalar_one_or_none()
            if pn is not None and pn.status == PlayerNode.ST_LOCKED:
                pn.status = PlayerNode.ST_AVAILABLE
                unlocked.append(child)
    return unlocked


# --------------------------------------------------------------------------- #
#  Knowledge power & public state
# --------------------------------------------------------------------------- #
def _recompute_power(session: Session, ps: PlayerState) -> None:
    nodes = {n.code: n for n in session.execute(select(GameNode)).scalars()}
    power = 0.0
    for pn in session.execute(select(PlayerNode).where(PlayerNode.user_id == ps.user_id)).scalars():
        node = nodes.get(pn.node_code)
        if node is None:
            continue
        if pn.status == PlayerNode.ST_MASTERED:
            power += node.difficulty * 12 + (30 if node.is_boss else 0)
        else:
            power += pn.progress / 100.0 * node.difficulty * 6
    power += min(ps.streak_days, 30) * 2
    ps.knowledge_power = int(power)


def world_powers(session: Session, user_id: int) -> list[dict[str, Any]]:
    worlds = list(session.execute(select(GameWorld).order_by(GameWorld.order_index)).scalars())
    nodes = list(session.execute(select(GameNode)).scalars())
    pns = {
        pn.node_code: pn
        for pn in session.execute(select(PlayerNode).where(PlayerNode.user_id == user_id)).scalars()
    }
    out = []
    for w in worlds:
        wnodes = [n for n in nodes if n.world_code == w.code]
        total = sum(n.difficulty for n in wnodes) or 1
        got = 0.0
        mastered = 0
        for n in wnodes:
            pn = pns.get(n.code)
            if pn is None:
                continue
            if pn.status == PlayerNode.ST_MASTERED:
                got += n.difficulty
                mastered += 1
            else:
                got += n.difficulty * pn.progress / 100.0
        out.append(
            {
                "code": w.code,
                "name": w.name_ru,
                "color": w.color,
                "glyph": w.glyph,
                "power": round(100.0 * got / total),
                "mastered": mastered,
                "total": len(wnodes),
            }
        )
    return out


def public_state(session: Session, ps: PlayerState) -> dict[str, Any]:
    lp = level_payload(ps.xp)
    unopened = session.execute(
        select(func.count()).select_from(PlayerChest).where(PlayerChest.user_id == ps.user_id, PlayerChest.opened.is_(False))
    ).scalar_one()
    return {
        **lp,
        "knowledge_power": ps.knowledge_power,
        "streak": ps.streak_days,
        "best_streak": ps.best_streak,
        "streak_shields": ps.streak_shields,
        "chests_unopened": unopened,
        "answers_total": ps.answers_total,
        "answers_correct": ps.answers_correct,
        "bosses_defeated": ps.bosses_defeated,
        "onboarded": ps.onboarded,
        "interests": ps.interests,
        "goal": ps.goal_ru,
        "intensity": ps.intensity,
        "selected_title": ps.selected_title,
        "selected_frame": ps.selected_frame,
    }


# --------------------------------------------------------------------------- #
#  Current objective (home screen)
# --------------------------------------------------------------------------- #
def current_objective(session: Session, ps: PlayerState) -> dict[str, Any] | None:
    nodes = {n.code: n for n in session.execute(select(GameNode)).scalars()}
    pns = list(session.execute(select(PlayerNode).where(PlayerNode.user_id == ps.user_id)).scalars())
    interests = set(ps.interests or [])

    def sort_key(pn: PlayerNode):
        node = nodes[pn.node_code]
        pref = 0 if (not interests or node.world_code in interests) else 1
        return (pref, -pn.progress, node.difficulty, node.order_index, node.code)

    active = sorted([p for p in pns if p.status == PlayerNode.ST_ACTIVE and p.node_code in nodes], key=sort_key)
    available = sorted([p for p in pns if p.status == PlayerNode.ST_AVAILABLE and p.node_code in nodes], key=sort_key)
    pick = (active or available or [None])[0]
    if pick is None:
        return None
    node = nodes[pick.node_code]
    world = session.execute(select(GameWorld).where(GameWorld.code == node.world_code)).scalar_one()
    return {
        "node": node.code,
        "name": node.name_ru,
        "short": node.short_ru,
        "world": world.code,
        "world_name": world.name_ru,
        "color": world.color,
        "progress": pick.progress,
        "is_boss": node.is_boss,
        "status": pick.status,
    }
