"""
Coding Lab: every seeded task must be runnable and gradeable, and the sandbox must be
actually isolated (no network, no process spawning, bounded time/memory, no server env leaks).
"""

from __future__ import annotations

import json

import pytest

from app.config import settings
from app.models import CodingTask, UserSkill, Skill
from app.services import coding as coding_service
from app.services.sandbox import run_python, sandbox_capabilities, screen_imports


def test_every_seeded_task_passes_with_its_own_solution(db, fast_seeded):
    """The reference solution of each seeded task must pass its tests in the sandbox."""
    tasks = list(db.query(CodingTask).all())
    assert len(tasks) >= 10, "seed must include at least 10 coding tasks"
    failures = []
    for task in tasks:
        result = run_python(task.solution, task.test_code)
        if not result.ok:
            failures.append(f"{task.slug}: {result.error[:200] or f'{result.passed_tests}/{result.total_tests} passed'}")
    assert not failures, "broken seed tasks:\n" + "\n".join(failures)


def test_seeded_tests_reject_wrong_solutions(db, fast_seeded):
    """A stub or a subtly wrong implementation must fail - otherwise the lab grades nothing."""
    stub = "raise NotImplementedError\n"
    wrong = "import numpy as np\n\n\ndef _f(*a, **k):\n    return 0.0\n"
    task = db.query(CodingTask).filter(CodingTask.slug == "numpy.moving_average").one()
    assert not run_python(stub, task.test_code).ok
    broken = "import numpy as np\n\n\ndef moving_average(x, k):\n    x = np.asarray(x, float)\n    return np.ones(len(x) - k + 1)\n"
    result = run_python(broken, task.test_code)
    assert not result.ok and result.passed_tests < result.total_tests


def test_from_scratch_tasks_are_marked_and_cover_internals(db, fast_seeded):
    from_scratch = [t for t in db.query(CodingTask).all() if t.from_scratch]
    slugs = {t.slug for t in from_scratch}
    assert len(from_scratch) >= 6
    for expected in ("math.mean_variance", "linalg.dot_matmul", "ml.linear_regression_gd", "ml.kmeans_from_scratch", "nn.backprop_two_layer"):
        assert expected in slugs, slugs


def test_sandbox_blocks_network_and_processes():
    for snippet in (
        "import socket\n\nsocket.socket()",
        "import subprocess\n\nsubprocess.run(['ls'])",
        "from urllib.request import urlopen\n\nurlopen('http://example.com')",
    ):
        violations = screen_imports(snippet)
        assert violations, f"static screen missed: {snippet[:40]}"
        result = run_python(snippet, "import unittest\n\nclass T(unittest.TestCase):\n    def test_x(self):\n        pass\n")
        assert not result.ok


def test_sandbox_kills_runaway_loops_and_reports_timeouts():
    result = run_python("while True:\n    pass", "import unittest\n\nclass T(unittest.TestCase):\n    def test_x(self):\n        pass\n", timeout=4)
    assert result.timed_out is True
    assert "limit" in result.error.lower()


def test_sandbox_environment_has_no_credentials():
    code = "import os\n\nKEYS = sorted(k for k in os.environ if 'KEY' in k or 'TOKEN' in k or 'SECRET' in k)\nPATH_SET = 'PATH' in os.environ\n"
    tests = (
        "import unittest\nfrom solution import KEYS, PATH_SET\n\n"
        "class T(unittest.TestCase):\n"
        "    def test_no_secrets(self):\n        self.assertEqual(KEYS, [])\n"
        "    def test_path_still_works(self):\n        self.assertTrue(PATH_SET)\n"
    )
    result = run_python(code, tests)
    assert result.ok, result.error


def test_sandbox_capabilities_report_is_honest():
    caps = sandbox_capabilities()
    assert caps["enabled"] == settings.code_execution_enabled
    assert caps["numpy"] is True
    assert "isolation" in caps and "rlimits" in caps["isolation"]


def test_run_then_solve_unlocks_solution_and_updates_skills(auth_client, db, fast_seeded):
    client = auth_client
    slug = "math.mean_variance"
    task_row = db.query(CodingTask).filter(CodingTask.slug == slug).one()

    locked = client.get(f"/api/coding/tasks/{slug}/solution").json()
    assert locked["available"] is False and locked["attempts"] == 0

    good = (
        "import numpy as np\n\n\n"
        "def mean(x):\n    x = np.asarray(x, dtype=float)\n    return float(x.sum() / x.size)\n\n\n"
        "def variance(x):\n    x = np.asarray(x, dtype=float)\n    mu = x.sum() / x.size\n    d = x - mu\n    return float((d * d).sum() / x.size)\n"
    )
    result = client.post(f"/api/coding/tasks/{slug}/submit", json={"code": good, "seconds": 120}).json()
    assert result["solved"] is True, result
    assert result["passed_tests"] == result["total_tests"] >= 3
    assert result["solution_unlocked"] is True
    assert result["skill_updates"], "solving a task must move the knowledge model"
    updated = {u["skill_code"]: u for u in result["skill_updates"]}
    assert "coding" in json.dumps(updated) or updated[list(updated)[0]]["after"] > 0

    unlocked = client.get(f"/api/coding/tasks/{slug}/solution").json()
    assert unlocked["available"] is True and "def mean" in unlocked["solution"]

    db.rollback()
    who = client.get("/api/auth/me").json()["user"]["id"]
    row = (
        db.query(UserSkill)
        .join(Skill, Skill.id == UserSkill.skill_id)
        .filter(UserSkill.user_id == who, Skill.code == "stat.describe")
        .one()
    )
    assert row.knowledge_score > 0 and row.attempts >= 1


def test_failing_code_is_recorded_and_hint_counter_shown(auth_client, db, fast_seeded):
    slug = "ml.kmeans_from_scratch"
    result = auth_client.post(f"/api/coding/tasks/{slug}/submit", json={"code": "def kmeans(X, k=3, iters=100, seed=0):\n    return None\n", "hints_used": 1}).json()
    assert result["solved"] is False
    assert result["total_tests"] >= 3
    assert result["test_results"], "per-test results must be returned for the UI"
    hint = auth_client.get(f"/api/coding/tasks/{slug}/hints", params={"index": 0}).json()
    assert hint["available"] >= 1 and hint["hint"]
    attempts = auth_client.get("/api/coding/attempts", params={"limit": 5}).json()
    assert attempts and attempts[0]["passed"] is False


def test_static_check_flags_syntax_before_running(auth_client):
    broken = auth_client.post("/api/coding/check", json={"code": "def f(:\n  pass"}).json()
    assert broken["ok"] is False and "line" in broken["syntax_error"]
    clean = auth_client.post("/api/coding/check", json={"code": "import numpy as np\n\n\ndef f(x):\n    return np.arange(x)\n"}).json()
    assert clean["ok"] is True and clean["imports_blocked"] == []


def test_generated_tasks_are_validated_before_storage(auth_client, db, fast_seeded, monkeypatch):
    """AI-generated coding tasks are only kept when their reference passes their own tests."""
    from app.ai.base import AIResult, DisabledProvider
    from app.ai.manager import AIManager
    from app.services import coding as coding

    class FakeGenerator(DisabledProvider):
        name = "stub"

        def model_label(self):
            return "stub"

        def health(self):
            return {"provider": "stub", "configured": True, "model": "stub"}

        def generate_practice(self, **kwargs):
            good_test = (
                "import unittest\nfrom solution import double\n\n"
                "class T(unittest.TestCase):\n    def test_two(self):\n        self.assertEqual(double(2), 4)\n"
            )
            return AIResult(
                ok=True,
                data={
                    "tasks": [
                        {
                            "title": "Double it",
                            "statement": "Implement double(x) returning 2*x.",
                            "starter_code": "def double(x):\n    raise NotImplementedError\n",
                            "test_code": good_test,
                            "solution": "def double(x):\n    return 2 * x\n",
                            "difficulty": 1,
                            "kind": "code",
                            "skills": ["py.idioms"],
                        },
                        {
                            "title": "Broken task",
                            "statement": "Tests that pass on nothing.",
                            "starter_code": "",
                            "test_code": good_test,
                            "solution": "def double(x):\n    return 3 * x\n",
                            "difficulty": 1,
                        },
                    ]
                },
                provider="stub",
                model="stub",
            )

    import app.ai.manager as manager_module

    monkeypatch.setattr(manager_module.manager, "_provider", FakeGenerator())
    result = coding.generate_tasks(db, _user(db), topic_code="python.core", count=2)
    db.commit()
    assert len(result["created"]) == 1, result
    assert any("reference solution fails" in r["reason"] for r in result["rejected"]), result["rejected"]


def _user(db):
    from app.models import User

    user = db.query(User).order_by(User.id.asc()).first()
    if user is None:
        user = User(username="gen-owner", daily_minutes=30)
        db.add(user)
        db.commit()
        db.refresh(user)
    return user
