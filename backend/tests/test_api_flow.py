"""
End-to-end first run (§49) over HTTP + the "no fake functionality" contract:
every UI route must exist and must return real data.
"""

from __future__ import annotations

import json
from typing import Any

import pytest
from sqlalchemy import select

from app.models import Document, Question, UserSkill, Skill


def _register(client, username: str) -> dict[str, str]:
    response = client.post(
        "/api/auth/register",
        json={"username": username, "display_name": "Anna", "background": "analyst", "daily_minutes": 45},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['token']}"}


def test_first_run_sequence(client, db, seeded):
    """1 profile → 2 diagnostic → 3 skill map → 4 dashboard → 5 path → 6 upload → 7 index →
    8 teacher → 9 uses the material → 10 lesson → 11 questions → 12 practice → 13 stored →
    14 plan changes → 15 offline-safe."""
    headers = _register(client, "first-run")

    # 1 profile
    me = client.get("/api/auth/me", headers=headers).json()
    assert me["user"]["username"] == "first-run"
    assert me["user"]["diagnostic_done"] is False

    # 2 diagnostic
    started = client.post("/api/diagnostic/start", headers=headers).json()
    assert 20 <= started["count"] <= 40, started["count"]
    answers = {}
    for item in started["items"]:
        if item["options"]:
            answers[str(item["id"])] = {"answer": "1", "selected_option": 1, "seconds": 25}
        else:
            answers[str(item["id"])] = {
                "answer": "the gradient is the vector of partial derivatives; we step against it scaled by the learning rate",
                "seconds": 45,
            }
    result = client.post("/api/diagnostic/submit", headers=headers, json={"answers": answers}).json()
    assert result["answered"] >= 20
    assert result["skills"], "diagnostic must produce a skill map"
    assert 0 <= result["level_index"] <= 10
    assert result["skill_map"]["nodes"], "graph payload for the roadmap view"
    assert result["recommendation"]["topic"], "a starting point must be chosen"

    # 3/4 skill map + dashboard
    dashboard = client.get("/api/dashboard", headers=headers).json()
    assert {"ml_engineer_score", "current_level", "today", "weak_skills", "due_for_review", "current_course", "current_project", "streak"} <= set(dashboard)
    assert dashboard["diagnostic_done"] is True

    # 5 recommended learning path
    roadmap = client.get("/api/roadmap", headers=headers).json()
    assert len(roadmap["levels"]) == 11
    assert any(level["topics"] for level in roadmap["levels"])

    # 6/7 upload + indexing of a real file
    pdf_text = (
        "# Field Guide to Regularization\n\n## L1 and L2\n\n"
        "L2 ridge regression shrinks coefficients smoothly and never reaches zero, while lasso can pin "
        "coefficients exactly at zero because the L1 contour has corners on the axes.\n\n"
        "## Practice\n\nStandardise features before penalising, otherwise lambda penalises small-unit columns "
        "more harshly than large-unit ones.\n"
    ) * 4
    upload = client.post(
        "/api/library/upload",
        files={"file": ("guide.md", pdf_text.encode(), "text/markdown")},
        data={"title": "Field Guide", "author": "A. Teacher", "kind": "notes", "wait": "true"},
        headers=headers,
    )
    assert upload.status_code == 201, upload.text
    payload = upload.json()
    assert payload["ingest"]["status"] == "indexed", payload
    document_id = payload["document"]["id"]
    assert payload["document"]["chunk_count"] >= 2

    listed = client.get("/api/library/documents", headers=headers).json()
    doc = next(d for d in listed["items"] if d["id"] == document_id)
    assert doc["status"] == "indexed" and doc["word_count"] >= 60

    # 8/9 AI Teacher uses the uploaded material
    chat = client.post(
        "/api/teacher/chat",
        json={"text": "When does lasso reach exactly zero coefficients?", "mode": "explain"},
        headers=headers,
    ).json()
    assert chat["answer"]
    assert chat["sources"], "an answer built from the library must cite it"
    assert any(s["document"] == "Field Guide" for s in chat["sources"]), chat["sources"]

    # 10 lesson: topic detail + explain endpoint
    topic_code = dashboard["current_course"]["topic"]
    if topic_code:
        detail = client.get(f"/api/topics/{topic_code}", headers=headers).json()
        assert detail["skills"] and "readiness" in detail

    # 11 questions
    question = client.post("/api/questions/next", json={"topic_code": "ml.linear_models"}, headers=headers).json()
    assert question["question"]["id"]
    graded = client.post(
        "/api/questions/answer",
        json={"question_id": question["question"]["id"], "answer": "lambda penalises coefficients; lasso gives sparsity", "seconds": 30},
        headers=headers,
    ).json()
    assert {"verdict", "skill_updates", "learning_engine", "engine"} <= set(graded)

    # 12 practice
    practice = client.get("/api/practice/next", params={"topic_code": "ml.linear_models"}, headers=headers).json()
    if practice["items"]:
        task = practice["items"][0]
        submitted = client.post(
            "/api/practice/submit",
            json={"task_id": task["id"], "answer": "mean 5, variance 4, so the standard deviation is 2", "seconds": 60},
            headers=headers,
        ).json()
        assert submitted["attempt_id"] and submitted["verdict"]["score"] >= 0
    else:
        generated = client.post("/api/practice/generate", json={"topic_code": "ml.linear_models", "level": "beginner"}, headers=headers).json()
        assert generated["items"] or "No practice" in generated["message"]

    # 13 results persisted (fresh session read)
    db.rollback()
    who = db.execute(select(UserSkill.id).limit(1)).scalars().first()
    progress = client.get("/api/progress", headers=headers).json()
    assert progress["skills_assessed"] >= 1, progress
    assert progress["xp"] > 0

    # 14 next plan reflects new state
    plan = client.get("/api/plan/today", headers=headers).json()
    assert plan["items"] and all("why" in i for i in plan["items"])
    refresh = client.post("/api/plan/today/refresh", json={"use_ai": False}, headers=headers).json()
    assert refresh["items"]
    done = client.post("/api/plan/item", json={"item_id": plan["items"][0]["id"], "done": True}, headers=headers).json()
    assert done["completed"] >= 1

    # 15 offline: AI counters show not configured, but everything still works
    assert dashboard["ai"]["configured"] is False
    usage = client.get("/api/ai/usage", headers=headers).json()
    assert usage["budget"]["requests_today"] >= 0


def test_library_search_and_kb_endpoints_return_citations(client, auth_client, db, seeded):
    client = auth_client
    search = client.get("/api/library/search", params={"q": "attention scores variance sqrt d_k"}).json()
    assert search["results"], "the library must be searchable"
    hit = search["results"][0]
    assert "document" in hit

    kb = client.post("/api/kb/search", json={"query": "why subtract the max logit in softmax", "top_k": 5}).json()
    assert kb["hits"]
    for entry in kb["hits"][:3]:
        assert entry["citation"]["document"]
        assert "score" in entry

    context = client.get("/api/kb/context", params={"query": "gradient clipping"}).json()
    assert context["context"]
    assert context["citations"]


def test_every_documented_route_is_reachable(client, auth_client, db, seeded):
    """§43: no button may point at a missing or stubbed endpoint."""
    client = auth_client
    checks: list[tuple[str, str, dict[str, Any] | None]] = [
        ("GET", "/api/health", None),
        ("GET", "/api/auth/me", None),
        ("GET", "/api/roadmap", None),
        ("GET", "/api/topics", None),
        ("GET", "/api/skills/graph", None),
        ("GET", "/api/plan/today", None),
        ("GET", "/api/review/due", None),
        ("GET", "/api/dashboard", None),
        ("GET", "/api/progress", None),
        ("GET", "/api/velocity", None),
        ("GET", "/api/library/documents", None),
        ("GET", "/api/library/stats", None),
        ("GET", "/api/kb/chunks/1", None),
        ("GET", "/api/teacher/modes", None),
        ("GET", "/api/teacher/conversations", None),
        ("GET", "/api/questions", None),
        ("GET", "/api/questions/stats", None),
        ("GET", "/api/practice", None),
        ("GET", "/api/coding/tasks", None),
        ("GET", "/api/coding/stats", None),
        ("GET", "/api/coding/capabilities", None),
        ("GET", "/api/projects", None),
        ("GET", "/api/projects/active", None),
        ("GET", "/api/exams", None),
        ("GET", "/api/exams/history", None),
        ("GET", "/api/interview/history", None),
        ("GET", "/api/notes", None),
        ("GET", "/api/journal", None),
        ("GET", "/api/memory", None),
        ("GET", "/api/ai/config", None),
        ("GET", "/api/ai/usage", None),
        ("GET", "/api/learning/state", None),
        ("GET", "/api/learning/weak", None),
        ("GET", "/api/system/seed", None),
    ]
    missing = []
    for method, path, body in checks:
        response = client.request(method, path, json=body) if body else client.request(method, path)
        if response.status_code in (404, 405, 500, 501):
            missing.append((method, path, response.status_code, response.text[:120]))
    assert not missing, f"unwired endpoints: {missing}"


def test_write_routes_round_trip(client, auth_client, db, seeded):
    """Create/update/delete paths used by the UI actually persist."""
    client = auth_client
    note = client.post("/api/notes", json={"kind": "concept", "title": "log-sum-exp", "body": "subtract max before exp"}).json()
    assert note["id"]
    patched = client.patch(f"/api/notes/{note['id']}", json={"pinned": True, "body": "subtract row max; shift invariance"}).json()
    assert patched["pinned"] is True and "shift invariance" in patched["body"]
    assert any(n["id"] == note["id"] for n in client.get("/api/notes").json())
    assert client.delete(f"/api/notes/{note['id']}").json()["deleted"] == note["id"]
    assert all(n["id"] != note["id"] for n in client.get("/api/notes").json())

    journal = client.post("/api/journal", json={"content": "Today regularization made sense, but I don't understand why batch norm helps at inference.", "mood": 4, "minutes": 40}).json()
    assert journal["id"] and journal["analysis"]["summary"]
    assert client.get("/api/journal").json()

    project = client.get("/api/projects").json()["items"][0]
    enrollment = client.post("/api/projects/enroll", json={"project_id": project["id"]}).json()
    assert enrollment["milestones"]
    mentor = client.post("/api/projects/mentor", json={"message": "How should I structure the data audit?", "enrollment_id": enrollment["id"]}).json()
    assert mentor["text"]
    sub = client.post(
        "/api/projects/submission",
        json={"enrollment_id": enrollment["id"], "content": "audit script + dtype tests", "user_implemented": "column classifier"},
    ).json()
    assert sub["submission_id"]
    evaluation = client.post(f"/api/projects/evaluate?enrollment_id={enrollment['id']}").json()
    assert evaluation["dimensions"]

    review = client.post("/api/plan/item", json={"item_id": (client.get("/api/plan/today").json()["items"][0]["id"]), "done": False}).json()
    assert review["total"] >= 1

    started = client.post("/api/interview/start", json={"level": "junior", "focus": "python"}).json()
    turn = client.post(f"/api/interview/{started['id']}/answer", json={"text": "A list is ordered and mutable; a tuple is immutable, so it can be a dict key.", "seconds": 40}).json()
    assert turn["snapshot"]["turns"]
    finished = client.post(f"/api/interview/{started['id']}/finish").json()
    assert finished["scores"] and finished["report"]


def test_unauthenticated_requests_are_rejected(client):
    for path in ("/api/dashboard", "/api/notes", "/api/plan/today", "/api/progress"):
        assert client.get(path).status_code == 401, path


def test_validation_errors_are_explicit(client, auth_client, db, seeded):
    client = auth_client
    assert client.get("/api/library/search", params={"q": ""}).status_code == 400
    assert client.post("/api/questions/answer", json={"question_id": 999999, "answer": "x"}).status_code == 404
    assert client.get("/api/topics/does.not.exist").status_code == 404
    assert client.post("/api/exams/nonexistent/start").status_code == 404
    assert client.post("/api/coding/tasks/nope/submit", json={"code": "x=1"}).status_code == 404
    bad = client.post("/api/library/url", json={"url": "ht!tp://bad"}).json()
    assert "detail" in bad


def test_ai_limited_endpoints_degrade_with_a_message(client, auth_client, db, seeded):
    """With no AI configured, AI endpoints must return the offline result, not an error page."""
    client = auth_client
    quiz = client.post("/api/teacher/chat", json={"text": "Quiz me on linear regression", "mode": "quiz", "topic_code": "ml.linear_models"}).json()
    assert quiz["answer"]
    assert quiz["engine"] in {"local", "ai"}
    if not quiz["attachments"]:
        assert "could not" in quiz["answer"].lower() or "no" in quiz["answer"].lower()

    gen = client.post("/api/questions/generate", json={"topic_code": "ml.linear_models", "count": 2}).json()
    assert isinstance(gen["created"], list)
    assert gen["engine"] in {"ai", "template-over-knowledge-base"}

    review = client.post(
        "/api/questions/review-answer",
        json={"prompt": "Explain why standardising features matters before ridge regression", "answer": "because the penalty applies to each coefficient, so units change the strength of lambda"},
    ).json()
    assert review["verdict"]["score"] >= 0
    assert review["verdict"]["dimensions"]["precision"] >= 0


def test_coding_lab_runs_through_http(client, auth_client, db, seeded):
    client = auth_client
    tasks = client.get("/api/coding/tasks", params={"only_unsolved": True}).json()["items"]
    assert tasks
    task = tasks[0]
    detail = client.get(f"/api/coding/tasks/{task['slug']}").json()
    assert detail["starter_code"] and detail["test_count"] >= 1
    run = client.post(f"/api/coding/tasks/{task['slug']}/run", json={"code": detail["starter_code"]}).json()
    assert run["total_tests"] >= 1
    assert run["ok"] is False, "an unfilled stub must not pass"
    solved = client.post(f"/api/coding/tasks/{task['slug']}/submit", json={"code": detail["solution"], "seconds": 100}).json() if "solution" in detail else None
    run_with_solution = client.post(f"/api/coding/tasks/{task['slug']}/run", json={"code": _solution_of(db, task["slug"])}).json()
    assert run_with_solution["ok"] is True, run_with_solution
    assert run_with_solution["passed_all"] is True


def _solution_of(db, slug: str) -> str:
    from app.models import CodingTask

    return db.execute(select(CodingTask.solution).where(CodingTask.slug == slug)).scalar_one()


def test_seed_sizes_match_the_spec(client, db):
    from app.seed import seed_summary

    summary = seed_summary(db)
    assert 20 <= summary["questions"] <= 120, summary
    assert summary["coding_tasks"] >= 10
    assert summary["practice_tasks"] >= 5
    assert summary["projects"] >= 3
    assert summary["exams"] == 3
    assert summary["topics"] >= 50 and summary["skills"] >= 100
    assert summary["dependencies"] >= 100
    assert summary["documents"] >= 1 and summary["chunks"] >= 5, "sample knowledge base must be ready on first run"
