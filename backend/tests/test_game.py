"""AI Knowledge RPG game layer: progression, economy, quests, teacher."""

from __future__ import annotations

import pytest
from sqlalchemy import select


@pytest.fixture()
def player(user, fast_seeded):
    from app.game.engine import ensure_player

    ps = ensure_player(fast_seeded, user.id)
    fast_seeded.commit()
    return ps


def test_seed_is_idempotent(fast_seeded):
    from app.game.seed import seed_game

    stats = seed_game(fast_seeded)
    assert all(v == 0 for v in stats.values())


def test_graph_is_consistent(fast_seeded):
    from app.game import content
    from app.game.models import GameNode, GameQuestion

    nodes = {n.code for n in fast_seeded.execute(select(GameNode)).scalars()}
    for frm, to in content.EDGES:
        assert frm in nodes and to in nodes
    # every node has questions
    for code in nodes:
        qs = list(fast_seeded.execute(select(GameQuestion).where(GameQuestion.node_code == code)).scalars())
        assert qs, f"node {code} has no questions"
        for q in qs:
            if q.qtype == "mcq":
                assert q.correct_index is not None and 0 <= q.correct_index < len(q.options)
            else:
                assert q.expected


def test_root_nodes_available(player, fast_seeded):
    from app.game.models import PlayerNode

    rows = list(fast_seeded.execute(select(PlayerNode).where(PlayerNode.user_id == player.user_id)).scalars())
    assert rows
    available = {r.node_code for r in rows if r.status == PlayerNode.ST_AVAILABLE}
    assert {"py-intro", "phys-motion", "math-numbers"} <= available


def _solve_node(session, ps, node_code):
    from app.game.engine import submit_answer
    from app.game.models import GameQuestion

    last = None
    for q in session.execute(select(GameQuestion).where(GameQuestion.node_code == node_code)).scalars():
        if q.qtype == "mcq":
            last = submit_answer(session, ps, q.id, option=q.correct_index)
        else:
            last = submit_answer(session, ps, q.id, answer=q.expected)
        assert last["correct"] is True
    return last


def test_mastery_unlocks_children_and_grants_rewards(player, fast_seeded):
    from app.game.models import PlayerChest, PlayerNode

    result = _solve_node(fast_seeded, player, "py-intro")
    assert result["node"]["mastered"] is True
    assert "py-vars" in result["node"]["unlocked"]
    assert player.xp > 0

    pn = fast_seeded.execute(
        select(PlayerNode).where(PlayerNode.user_id == player.user_id, PlayerNode.node_code == "py-vars")
    ).scalar_one()
    assert pn.status == PlayerNode.ST_AVAILABLE

    chests = list(fast_seeded.execute(select(PlayerChest).where(PlayerChest.user_id == player.user_id)).scalars())
    assert chests, "mastery must grant a chest"


def test_no_xp_farming_on_resolved_question(player, fast_seeded):
    from app.game.engine import submit_answer
    from app.game.models import GameQuestion

    q = fast_seeded.execute(select(GameQuestion).where(GameQuestion.node_code == "py-intro")).scalars().first()
    first = submit_answer(fast_seeded, player, q.id, option=q.correct_index)
    xp_after_first = player.xp
    second = submit_answer(fast_seeded, player, q.id, option=q.correct_index)
    assert second["already_solved"] is True
    assert player.xp == xp_after_first


def test_locked_node_rejects_answers(player, fast_seeded):
    from app.game.engine import submit_answer
    from app.game.models import GameQuestion

    q = fast_seeded.execute(select(GameQuestion).where(GameQuestion.node_code == "py-boss")).scalars().first()
    with pytest.raises(ValueError):
        submit_answer(fast_seeded, player, q.id, answer="12")


def test_numeric_answers_tolerant_to_format(player, fast_seeded):
    from app.game.engine import _check_answer
    from app.game.models import GameQuestion

    q = GameQuestion(node_code="x", qtype="input", prompt_ru="?", expected="2.75", tolerance=0.05)
    assert _check_answer(q, "2,75", None)
    assert _check_answer(q, " 2.78", None)
    assert not _check_answer(q, "3.2", None)


def test_chest_open_is_server_authoritative(player, fast_seeded):
    from app.game.engine import open_chest, _grant_chest
    from app.game.models import PlayerItem

    fx = {}
    chest = _grant_chest(fast_seeded, player, source="node", world_code="python", fx=fx)
    result = open_chest(fast_seeded, player, chest.id)
    assert result["rarity"] in {"common", "uncommon", "rare", "epic", "legendary", "mythic"}
    assert result["item"]["code"]
    items = list(fast_seeded.execute(select(PlayerItem).where(PlayerItem.user_id == player.user_id)).scalars())
    assert items
    with pytest.raises(ValueError):
        open_chest(fast_seeded, player, chest.id)  # cannot open twice


def test_daily_quests_generated_and_progress(player, fast_seeded):
    from app.game.engine import ensure_daily_quests

    quests = ensure_daily_quests(fast_seeded, player)
    assert len(quests) == 3
    again = ensure_daily_quests(fast_seeded, player)
    assert len(again) == 3  # no duplicates for the same day


def test_level_curve_monotonic():
    from app.game.engine import level_from_xp, xp_for_level

    assert xp_for_level(1) == 0
    prev = 0
    for lvl in range(2, 30):
        cur = xp_for_level(lvl)
        assert cur > prev
        prev = cur
    assert level_from_xp(0) == 1
    assert level_from_xp(xp_for_level(5)) == 5


def test_teacher_offline_reply_is_russian_and_contextual(player, fast_seeded):
    from app.game.teacher import teacher_reply

    res = teacher_reply(fast_seeded, player, message="Объясни эту тему", node_code="py-vars", mode="explain")
    assert res["via_ai"] is False
    assert "Переменные" in res["text"]

    res2 = teacher_reply(fast_seeded, player, message="расскажи про второй закон ньютона", mode="explain")
    assert "закон" in res2["text"].lower()


def test_game_api_full_loop(client):
    r = client.post("/api/auth/register", json={"username": "gamer", "display_name": "Игрок"})
    token = r.json()["token"]
    h = {"Authorization": f"Bearer {token}"}

    r = client.post(
        "/api/game/onboard",
        json={"interests": ["python"], "start_level": "beginner", "goal": "тест", "intensity": "normal"},
        headers=h,
    )
    assert r.status_code == 200
    assert r.json()["state"]["onboarded"] is True
    assert r.json()["state"]["chests_unopened"] >= 1

    state = client.get("/api/game/state", headers=h).json()
    assert state["objective"] is not None
    assert len(state["quests"]) == 3

    node = client.get("/api/game/node/py-intro", headers=h).json()
    assert node["theory"]
    q = node["questions"][0]

    hint = client.post("/api/game/hint", json={"question_id": q["id"]}, headers=h).json()
    assert hint["hint"]

    # the client cannot grant itself XP — only /answer with a correct option does
    res = client.post("/api/game/answer", json={"question_id": q["id"], "option": 1}, headers=h).json()
    assert res["correct"] is True
    assert res["fx"]["xp_gained"] > 0

    chest_id = state["chests"][0]["id"]
    opened = client.post(f"/api/game/chests/{chest_id}/open", headers=h).json()
    assert opened["item"]["code"]

    profile = client.get("/api/game/profile", headers=h).json()
    assert any(a["unlocked"] for a in profile["achievements"])
    assert profile["timeline"]

    lb = client.get("/api/game/leaderboard", headers=h).json()
    assert any(e["me"] for e in lb["entries"])
