"""
Seeding: curriculum graph + initial content + sample knowledge documents.

Idempotent by design: every table is keyed by a stable code/hash so repeated startups
never duplicate questions, tasks or chunks.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.knowledge.ingest import ingest_document
from app.knowledge.textutil import content_hash
from app.models import (
    CodingTask,
    Document,
    Exam,
    PracticeTask,
    Project,
    Question,
    Skill,
    SkillDependency,
    Topic,
    User,
)
from app.seed.content import CODING_TASKS, EXAMS, MATH_TASKS, PROJECTS, QUESTIONS
from app.seed.curriculum import SKILLS, TOPICS

logger = logging.getLogger(__name__)

SAMPLE_DIR = Path(__file__).parent / "sample_docs"
SAMPLE_TAG = "sample-knowledge"


def _seed_curriculum(session: Session) -> dict[str, int]:
    topic_ids: dict[str, int] = {}
    for row in session.execute(select(Topic)).scalars():
        topic_ids[row.code] = row.id
    created_topics = 0
    for code, name, level, domain, summary, keywords, hours in TOPICS:
        if code in topic_ids:
            topic = session.execute(select(Topic).where(Topic.code == code)).scalar_one()
            topic.name, topic.level, topic.domain = name, level, domain
            topic.summary, topic.keywords, topic.est_hours = summary, keywords, hours
            continue
        topic = Topic(code=code, name=name, level=level, domain=domain, summary=summary, keywords=keywords, est_hours=hours)
        session.add(topic)
        session.flush()
        topic_ids[topic.code] = topic.id
        created_topics += 1

    skill_ids: dict[str, int] = {}
    for row in session.execute(select(Skill)).scalars():
        skill_ids[row.code] = row.id
    created_skills = 0
    for code, name, category, difficulty, topic_code, keywords, deps in SKILLS:
        payload = {
            "name": name,
            "category": category,
            "difficulty": difficulty,
            "keywords": keywords,
            "topic_id": topic_ids.get(topic_code),
            "level": next((t[2] for t in TOPICS if t[0] == topic_code), 0),
            "description": f"{name}. Assessed through questions, coding tasks and projects; prerequisites are enforced through the skill graph.",
        }
        if code in skill_ids:
            skill = session.get(Skill, skill_ids[code])
            for key, value in payload.items():
                setattr(skill, key, value)
            continue
        skill = Skill(code=code, **payload)
        session.add(skill)
        session.flush()
        skill_ids[code] = skill.id
        created_skills += 1

    existing = {(d.skill_id, d.depends_on_id) for d in session.execute(select(SkillDependency)).scalars()}
    created_edges = 0
    for code, _name, _cat, _diff, _topic, _kw, deps in SKILLS:
        if code not in skill_ids:
            continue
        for dep in deps:
            if dep in skill_ids and (skill_ids[code], skill_ids[dep]) not in existing:
                session.add(SkillDependency(skill_id=skill_ids[code], depends_on_id=skill_ids[dep], weight=1.0))
                existing.add((skill_ids[code], skill_ids[dep]))
                created_edges += 1
    session.flush()
    return {"topics": created_topics, "skills": created_skills, "dependencies": created_edges}


def _topic_level_map(session: Session) -> dict[str, int]:
    return {t.code: t.level for t in session.execute(select(Topic)).scalars()}


def _seed_questions(session: Session) -> int:
    known = set(session.execute(select(Question.public_code)).scalars())
    levels = _topic_level_map(session)
    created = 0
    for spec in QUESTIONS:
        if spec["code"] in known:
            continue
        qtype = spec["type"]
        if qtype == "coding":
            qtype = "code"
        options: list[dict[str, Any]] = []
        correct = spec.get("correct")
        if spec["options"]:
            options = [{"id": i, "text": text} for i, text in enumerate(spec["options"])]
        digest = content_hash(spec["stem"] + "||" + str(spec.get("answer", ""))[:200])
        session.add(
            Question(
                public_code=spec["code"],
                topic_code=spec["topic"],
                skill_codes=spec["skills"],
                level=int(spec.get("level", levels.get(spec["topic"], 0)) or 0),
                question_type=qtype,
                difficulty=int(spec.get("difficulty", 2)),
                stem=spec["stem"],
                options=options,
                correct_option=int(correct) if isinstance(correct, int) else None,
                expected_answer=str(spec.get("answer") or ""),
                expected_value=str(spec.get("expected_value") or ""),
                tolerance=float(spec.get("tolerance") or 0.0),
                expected_points=list(spec.get("points") or []),
                explanation=str(spec.get("explanation") or ""),
                generated_by="seed",
                content_hash=digest,
                embedding_text=spec["stem"],
                approved=True,
            )
        )
        created += 1
    session.flush()
    return created


def _seed_practice(session: Session) -> int:
    existing = set(session.execute(select(PracticeTask.content_hash)).scalars())
    created = 0
    for spec in MATH_TASKS:
        digest = content_hash(spec["title"] + spec["statement"])
        if digest in existing:
            continue
        session.add(
            PracticeTask(
                topic_code=spec["topic"],
                skill_codes=spec["skills"],
                level=spec["level"],
                kind=spec.get("kind", "math"),
                title=spec["title"],
                statement=spec["statement"],
                given_data=spec.get("given_data", ""),
                expected_answer=spec.get("expected_answer", ""),
                expected_value=str(spec.get("expected_value", "")),
                tolerance=float(spec.get("tolerance", 0.0)),
                steps_required=spec.get("steps_required", []),
                starter_code=spec.get("starter_code", ""),
                hints=spec.get("hints", []),
                solution=spec.get("solution", ""),
                explanation=spec.get("explanation", ""),
                difficulty=int(spec.get("difficulty", 2)),
                est_minutes=int(spec.get("est_minutes", 10)),
                generated_by="seed",
                content_hash=digest,
            )
        )
        created += 1
    session.flush()
    return created


def _seed_coding(session: Session) -> int:
    existing = set(session.execute(select(CodingTask.slug)).scalars())
    created = 0
    for spec in CODING_TASKS:
        if spec["slug"] in existing:
            continue
        session.add(
            CodingTask(
                slug=spec["slug"],
                title=spec["title"],
                description=spec["description"],
                language="python",
                difficulty=int(spec.get("difficulty", 2)),
                level=spec.get("level", "beginner"),
                libraries=spec.get("libraries", ["numpy"]),
                skill_codes=spec.get("skills", []),
                topic_code=spec.get("topic", ""),
                from_scratch=bool(spec.get("from_scratch", False)),
                banned_imports=spec.get("banned_imports", []),
                starter_code=spec.get("starter_code", ""),
                test_code=spec.get("test_code", "") or spec.get("tests", ""),
                hints=spec.get("hints", []),
                solution=spec.get("solution", ""),
                solution_explanation=spec.get("explanation", ""),
                xp=int(spec.get("xp", 20)),
                est_minutes=int(spec.get("est_minutes", 20)),
            )
        )
        created += 1
    session.flush()
    return created


def _seed_projects(session: Session) -> int:
    existing = set(session.execute(select(Project.slug)).scalars())
    created = 0
    for spec in PROJECTS:
        if spec["slug"] in existing:
            continue
        session.add(
            Project(
                slug=spec["slug"],
                title=spec["title"],
                description=spec["description"],
                target_level=spec["target_level"],
                domain=spec.get("domain", "general"),
                skill_codes=spec.get("skills", []),
                stack=spec.get("stack", []),
                deliverables=spec.get("deliverables", []),
                milestones=spec.get("milestones", []),
                rubric=spec.get("rubric", []),
                guidance=spec.get("guidance", ""),
                dataset_hint=spec.get("dataset_hint", ""),
                difficulty=int(spec.get("difficulty", 2)),
                est_hours=float(spec.get("est_hours", 20.0)),
                generated_by="seed",
                content_hash=content_hash(spec["slug"]),
            )
        )
        created += 1
    session.flush()
    return created


def project_milestones(slug: str) -> list[dict[str, Any]]:
    """Milestone templates for a seeded project slug (used when a learner enrols)."""
    for spec in PROJECTS:
        if spec["slug"] == slug:
            return list(spec.get("milestones") or [])
    return []


def _seed_exams(session: Session) -> int:
    existing = set(session.execute(select(Exam.code)).scalars())
    created = 0
    for spec in EXAMS:
        if spec["code"] in existing:
            continue
        session.add(
            Exam(
                code=spec["code"],
                title=spec["title"],
                description=spec["description"],
                target_level=spec["target_level"],
                min_level=int(spec.get("min_level", 0)),
                sections=spec.get("sections", []),
                question_codes=spec.get("question_codes", []),
                passing_score=float(spec.get("passing_score", 70.0)),
                duration_minutes=int(spec.get("duration_minutes", 90)),
            )
        )
        created += 1
    session.flush()
    return created


def _seed_sample_documents(session: Session, *, owner_id: int | None = None) -> int:
    """Index the small self-authored sample library so RAG works on first boot (§45)."""
    if not SAMPLE_DIR.is_dir():
        return 0
    indexed = {
        (doc.title, doc.filename)
        for doc in session.execute(select(Document).where(Document.kind == "sample")).scalars()
    }
    first_owner = owner_id or (session.execute(select(User.id).order_by(User.id.asc()).limit(1)).scalar_one_or_none())
    count = 0
    for path in sorted(SAMPLE_DIR.glob("*.md")):
        if ("Sample · " + path.stem, path.name) in indexed:
            continue
        document = Document(
            owner_id=first_owner,
            title=f"Sample · {path.stem.replace('_', ' ').title()}",
            doc_type="md",
            author="ML Engineer Academy",
            kind="sample",
            tier=1,
            trust=1.0,
            filename=path.name,
            file_path=str(path),
            size_bytes=path.stat().st_size,
            tags=[SAMPLE_TAG],
            status=Document.STATUS_UPLOADING,
            notes="Self-authored sample material used to demonstrate RAG. Free to copy.",
        )
        session.add(document)
        session.commit()
        session.refresh(document)
        ingest_document(session, document.id)
        count += 1
    return count


def ensure_default_user(session: Session, *, username: str = "learner", display_name: str = "Learner") -> User:
    user = session.execute(select(User).where(User.username == username)).scalar_one_or_none()
    if user is not None:
        return user
    user = User(username=username, display_name=display_name, onboarded=False, daily_minutes=60)
    session.add(user)
    session.commit()
    session.refresh(user)
    return user


def seed_all(session: Session, *, index_samples: bool = True, owner_id: int | None = None) -> dict[str, Any]:
    stats: dict[str, Any] = {}
    stats["curriculum"] = _seed_curriculum(session)
    stats["questions"] = _seed_questions(session)
    stats["practice_tasks"] = _seed_practice(session)
    stats["coding_tasks"] = _seed_coding(session)
    stats["projects"] = _seed_projects(session)
    stats["exams"] = _seed_exams(session)
    session.commit()
    if index_samples:
        stats["sample_documents"] = _seed_sample_documents(session, owner_id=owner_id)
    logger.info("seed: %s", stats)
    return stats


def seed_summary(session: Session) -> dict[str, Any]:
    from sqlalchemy import func

    from app.models import DocumentChunk

    def count(model: Any) -> int:
        return int(session.execute(select(func.count()).select_from(model)).scalar_one() or 0)

    return {
        "topics": count(Topic),
        "skills": count(Skill),
        "dependencies": count(SkillDependency),
        "questions": count(Question),
        "practice_tasks": count(PracticeTask),
        "coding_tasks": count(CodingTask),
        "projects": count(Project),
        "exams": count(Exam),
        "documents": count(Document),
        "chunks": count(DocumentChunk),
    }
