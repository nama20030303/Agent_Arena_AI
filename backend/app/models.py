"""
Persistence layer.

Design rule of the whole product: **KNOWLEDGE** (what the library contains) and
**LEARNING STATE** (what this specific user knows) are stored separately and
only joined at query time. Everything the AI produces is persisted so it can be
reused, deduplicated, evaluated and audited.
"""

from __future__ import annotations

from datetime import datetime, date
from typing import Any

from sqlalchemy import (
    JSON,
    Boolean,
    Date,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    LargeBinary,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.db import Base


def utcnow() -> datetime:
    return datetime.utcnow()


# --------------------------------------------------------------------------- #
#  Identity
# --------------------------------------------------------------------------- #
class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(primary_key=True)
    username: Mapped[str] = mapped_column(String(80), unique=True, index=True)
    display_name: Mapped[str] = mapped_column(String(120), default="")
    email: Mapped[str] = mapped_column(String(160), default="")
    target_role: Mapped[str] = mapped_column(String(60), default="Senior ML Engineer")
    background: Mapped[str] = mapped_column(String(120), default="")
    native_language: Mapped[str] = mapped_column(String(16), default="en")
    reply_language: Mapped[str] = mapped_column(String(16), default="auto")  # auto = match question
    daily_minutes: Mapped[int] = mapped_column(Integer, default=60)
    goal: Mapped[str] = mapped_column(Text, default="")
    level_label: Mapped[str] = mapped_column(String(40), default="Absolute Beginner")
    level_index: Mapped[int] = mapped_column(Integer, default=0)
    onboarded: Mapped[bool] = mapped_column(Boolean, default=False)
    diagnostic_done: Mapped[bool] = mapped_column(Boolean, default=False)
    web_search_allowed: Mapped[bool] = mapped_column(Boolean, default=False)
    ai_enabled: Mapped[bool] = mapped_column(Boolean, default=True)
    settings_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_active_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    documents: Mapped[list["Document"]] = relationship(back_populates="owner", cascade="all, delete-orphan")
    skills: Mapped[list["UserSkill"]] = relationship(back_populates="user", cascade="all, delete-orphan")


class UserSession(Base):
    """Minimal bearer-token sessions: local personal app, no password hashing theatre."""

    __tablename__ = "user_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    token: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_used_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------- #
#  KNOWLEDGE: documents, chunks, provenance
# --------------------------------------------------------------------------- #
class Document(Base):
    __tablename__ = "documents"

    STATUS_UPLOADING = "uploading"
    STATUS_PROCESSING = "processing"
    STATUS_INDEXED = "indexed"
    STATUS_ERROR = "error"

    id: Mapped[int] = mapped_column(primary_key=True)
    owner_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), nullable=True)

    title: Mapped[str] = mapped_column(String(400), index=True)
    doc_type: Mapped[str] = mapped_column(String(24), default="md")  # pdf|epub|txt|md|html|url
    author: Mapped[str] = mapped_column(String(300), default="")
    year: Mapped[int | None] = mapped_column(Integer, nullable=True)
    edition: Mapped[str] = mapped_column(String(80), default="")
    publisher: Mapped[str] = mapped_column(String(200), default="")
    language: Mapped[str] = mapped_column(String(16), default="en")

    file_path: Mapped[str] = mapped_column(String(700), default="")
    source_url: Mapped[str] = mapped_column(String(1200), default="")
    filename: Mapped[str] = mapped_column(String(400), default="")
    size_bytes: Mapped[int] = mapped_column(Integer, default=0)
    page_count: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)

    status: Mapped[str] = mapped_column(String(24), default=STATUS_UPLOADING, index=True)
    error_message: Mapped[str] = mapped_column(Text, default="")
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    # 4.1 source tiering: 1 library, 2 official docs, 3 papers, 4 trusted education, 5 general web
    tier: Mapped[int] = mapped_column(Integer, default=1)
    kind: Mapped[str] = mapped_column(String(40), default="book")  # book|docs|paper|course|article|notes
    trust: Mapped[float] = mapped_column(Float, default=1.0)
    topics: Mapped[list[str]] = mapped_column(JSON, default=list)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    chunk_count: Mapped[int] = mapped_column(Integer, default=0)
    ingest_options: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    notes: Mapped[str] = mapped_column(Text, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    owner: Mapped["User | None"] = relationship(back_populates="documents")
    chunks: Mapped[list["DocumentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan", passive_deletes=True
    )

    __table_args__ = (Index("ix_documents_owner_status", "owner_id", "status"),)


class DocumentChunk(Base):
    __tablename__ = "document_chunks"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    chunk_index: Mapped[int] = mapped_column(Integer, default=0)

    text: Mapped[str] = mapped_column(Text)
    headline: Mapped[str] = mapped_column(String(400), default="")
    chapter: Mapped[str] = mapped_column(String(300), default="")
    section: Mapped[str] = mapped_column(String(300), default="")
    page_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    page_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    heading_path: Mapped[list[str]] = mapped_column(JSON, default=list)
    topic: Mapped[str] = mapped_column(String(160), default="", index=True)
    skills: Mapped[list[str]] = mapped_column(JSON, default=list)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    char_count: Mapped[int] = mapped_column(Integer, default=0)
    quality: Mapped[float] = mapped_column(Float, default=1.0)
    meta: Mapped[dict[str, Any]] = mapped_column("metadata", JSON, default=dict)

    embedding: Mapped[bytes | None] = mapped_column(LargeBinary, nullable=True)
    embedding_dim: Mapped[int] = mapped_column(Integer, default=0)
    embedding_model: Mapped[str] = mapped_column(String(80), default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped["Document"] = relationship(back_populates="chunks")

    __table_args__ = (
        UniqueConstraint("document_id", "chunk_index", name="uq_chunk_doc_index"),
        Index("ix_chunks_topic", "topic"),
    )


class Source(Base):
    """Provenance record used for citation + retrieval ranking."""

    __tablename__ = "sources"

    id: Mapped[int] = mapped_column(primary_key=True)
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(40), default="library")
    url: Mapped[str] = mapped_column(String(1200), default="")
    title: Mapped[str] = mapped_column(String(400), default="")
    publisher: Mapped[str] = mapped_column(String(200), default="")
    tier: Mapped[int] = mapped_column(Integer, default=1)
    trust: Mapped[float] = mapped_column(Float, default=1.0)
    official: Mapped[bool] = mapped_column(Boolean, default=False)
    fetched_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    document: Mapped["Document | None"] = relationship()


class EmbeddingStat(Base):
    """Corpus document-frequency table backing the offline embedder's IDF weights."""

    __tablename__ = "embedding_stats"

    id: Mapped[int] = mapped_column(primary_key=True)
    term: Mapped[str] = mapped_column(String(64), index=True)
    doc_freq: Mapped[int] = mapped_column(Integer, default=1)
    total_docs: Mapped[int] = mapped_column(Integer, default=1)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------- #
#  Curriculum: topics, skills, dependency graph
# --------------------------------------------------------------------------- #
class Topic(Base):
    __tablename__ = "topics"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(90), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    level: Mapped[int] = mapped_column(Integer, default=0, index=True)  # 0..10 from the roadmap
    domain: Mapped[str] = mapped_column(String(60), default="general")
    summary: Mapped[str] = mapped_column(Text, default="")
    prerequisites: Mapped[list[str]] = mapped_column(JSON, default=list)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    est_hours: Mapped[float] = mapped_column(Float, default=4.0)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    skills: Mapped[list["Skill"]] = relationship(back_populates="topic", cascade="all, delete-orphan")


class Skill(Base):
    __tablename__ = "skills"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(90), unique=True, index=True)
    name: Mapped[str] = mapped_column(String(200))
    category: Mapped[str] = mapped_column(String(32), default="theory")  # theory|math|coding|engineering|problem_solving
    level: Mapped[int] = mapped_column(Integer, default=0, index=True)
    difficulty: Mapped[int] = mapped_column(Integer, default=2)  # 1..5
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id", ondelete="SET NULL"), nullable=True)
    description: Mapped[str] = mapped_column(Text, default="")
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)
    from_scratch: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    topic: Mapped["Topic | None"] = relationship(back_populates="skills")
    dependents: Mapped[list["SkillDependency"]] = relationship(
        back_populates="skill", foreign_keys="SkillDependency.skill_id", cascade="all, delete-orphan"
    )

    __table_args__ = (Index("ix_skills_level_cat", "level", "category"),)


class SkillDependency(Base):
    __tablename__ = "skill_dependencies"

    id: Mapped[int] = mapped_column(primary_key=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), index=True)
    depends_on_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), index=True)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    required_score: Mapped[float] = mapped_column(Float, default=55.0)
    strict: Mapped[bool] = mapped_column(Boolean, default=True)

    skill: Mapped["Skill"] = relationship(foreign_keys=[skill_id])
    depends_on: Mapped["Skill"] = relationship(foreign_keys=[depends_on_id])


class UserSkill(Base):
    """The user knowledge model - one row per (user, skill)."""

    __tablename__ = "user_skills"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    skill_id: Mapped[int] = mapped_column(ForeignKey("skills.id", ondelete="CASCADE"), index=True)

    knowledge_score: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    theory_score: Mapped[float] = mapped_column(Float, default=0.0)
    math_score: Mapped[float] = mapped_column(Float, default=0.0)
    coding_score: Mapped[float] = mapped_column(Float, default=0.0)
    problem_solving_score: Mapped[float] = mapped_column(Float, default=0.0)
    engineering_score: Mapped[float] = mapped_column(Float, default=0.0)

    last_review: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_review: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    success_rate: Mapped[float] = mapped_column(Float, default=0.0)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    successes: Mapped[int] = mapped_column(Integer, default=0)
    streak: Mapped[int] = mapped_column(Integer, default=0)
    failure_count: Mapped[int] = mapped_column(Integer, default=0)
    difficulty_level: Mapped[int] = mapped_column(Integer, default=1)  # 1 beginner .. 4 production
    mastery_state: Mapped[str] = mapped_column(String(24), default="new")
    xp: Mapped[int] = mapped_column(Integer, default=0)
    minutes_studied: Mapped[float] = mapped_column(Float, default=0.0)
    last_error_type: Mapped[str] = mapped_column(String(60), default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    user: Mapped["User"] = relationship(back_populates="skills")
    skill: Mapped["Skill"] = relationship()

    __table_args__ = (UniqueConstraint("user_id", "skill_id", name="uq_user_skill"),)


# --------------------------------------------------------------------------- #
#  Questions
# --------------------------------------------------------------------------- #
class Question(Base):
    __tablename__ = "questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    public_code: Mapped[str] = mapped_column(String(40), default="", index=True)
    topic_code: Mapped[str] = mapped_column(String(90), default="", index=True)
    skill_codes: Mapped[list[str]] = mapped_column(JSON, default=list)
    level: Mapped[int] = mapped_column(Integer, default=0)

    question_type: Mapped[str] = mapped_column(String(32), default="conceptual")
    # conceptual|mcq|open|math|code|debug|architecture|interview|from_scratch
    difficulty: Mapped[int] = mapped_column(Integer, default=2)  # 1..5
    stem: Mapped[str] = mapped_column(Text)
    options: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    correct_option: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expected_answer: Mapped[str] = mapped_column(Text, default="")
    expected_value: Mapped[str] = mapped_column(String(200), default="")  # numeric / exact match
    tolerance: Mapped[float] = mapped_column(Float, default=0.0)
    expected_points: Mapped[list[str]] = mapped_column(JSON, default=list)
    rubric: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    explanation: Mapped[str] = mapped_column(Text, default="")
    starter_code: Mapped[str] = mapped_column(Text, default="")
    test_code: Mapped[str] = mapped_column(Text, default="")
    hints: Mapped[list[str]] = mapped_column(JSON, default=list)

    source_chunk_id: Mapped[int | None] = mapped_column(ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True)
    source_document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    citation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)

    generated_by: Mapped[str] = mapped_column(String(16), default="seed")  # seed|ai|template
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    embedding_text: Mapped[str] = mapped_column(Text, default="")
    use_count: Mapped[int] = mapped_column(Integer, default=0)
    answer_count: Mapped[int] = mapped_column(Integer, default=0)
    correct_count: Mapped[int] = mapped_column(Integer, default=0)
    approved: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    attempts: Mapped[list["QuestionAttempt"]] = relationship(back_populates="question", cascade="all, delete-orphan")

    __table_args__ = (Index("ix_questions_topic_type", "topic_code", "question_type"),)


class QuestionAttempt(Base):
    __tablename__ = "question_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("questions.id", ondelete="CASCADE"), index=True)

    answer: Mapped[str] = mapped_column(Text, default="")
    selected_option: Mapped[int | None] = mapped_column(Integer, nullable=True)
    code: Mapped[str] = mapped_column(Text, default="")
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)

    correctness: Mapped[float] = mapped_column(Float, default=0.0)
    depth: Mapped[float] = mapped_column(Float, default=0.0)
    reasoning: Mapped[float] = mapped_column(Float, default=0.0)
    precision: Mapped[float] = mapped_column(Float, default=0.0)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)

    error_type: Mapped[str] = mapped_column(String(60), default="")
    missing_points: Mapped[list[str]] = mapped_column(JSON, default=list)
    feedback: Mapped[str] = mapped_column(Text, default="")
    correction: Mapped[str] = mapped_column(Text, default="")
    evaluated_by: Mapped[str] = mapped_column(String(16), default="local")  # local|ai
    time_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    retry_of_id: Mapped[int | None] = mapped_column(ForeignKey("question_attempts.id", ondelete="SET NULL"), nullable=True)
    session_id: Mapped[int | None] = mapped_column(ForeignKey("study_sessions.id", ondelete="SET NULL"), nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    question: Mapped["Question"] = relationship(back_populates="attempts")

    __table_args__ = (Index("ix_attempts_user_created", "user_id", "created_at"),)


# --------------------------------------------------------------------------- #
#  Practice
# --------------------------------------------------------------------------- #
class PracticeTask(Base):
    __tablename__ = "practice_tasks"

    LEVELS = ("beginner", "intermediate", "advanced", "production")

    id: Mapped[int] = mapped_column(primary_key=True)
    topic_code: Mapped[str] = mapped_column(String(90), default="", index=True)
    skill_codes: Mapped[list[str]] = mapped_column(JSON, default=list)
    level: Mapped[str] = mapped_column(String(20), default="beginner")
    kind: Mapped[str] = mapped_column(String(24), default="math")  # math|concept|code|analysis|design
    title: Mapped[str] = mapped_column(String(300))
    statement: Mapped[str] = mapped_column(Text, default="")
    given_data: Mapped[str] = mapped_column(Text, default="")
    expected_answer: Mapped[str] = mapped_column(Text, default="")
    expected_value: Mapped[str] = mapped_column(String(200), default="")
    tolerance: Mapped[float] = mapped_column(Float, default=0.0)
    steps_required: Mapped[list[str]] = mapped_column(JSON, default=list)
    starter_code: Mapped[str] = mapped_column(Text, default="")
    hints: Mapped[list[str]] = mapped_column(JSON, default=list)
    solution: Mapped[str] = mapped_column(Text, default="")
    explanation: Mapped[str] = mapped_column(Text, default="")
    difficulty: Mapped[int] = mapped_column(Integer, default=2)
    est_minutes: Mapped[int] = mapped_column(Integer, default=10)
    source_document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    source_chunk_id: Mapped[int | None] = mapped_column(ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True)
    citation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    generated_by: Mapped[str] = mapped_column(String(16), default="seed")
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class PracticeAttempt(Base):
    __tablename__ = "practice_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("practice_tasks.id", ondelete="CASCADE"), index=True)
    answer: Mapped[str] = mapped_column(Text, default="")
    code: Mapped[str] = mapped_column(Text, default="")
    is_correct: Mapped[bool | None] = mapped_column(Boolean, nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    feedback: Mapped[str] = mapped_column(Text, default="")
    missing_steps: Mapped[list[str]] = mapped_column(JSON, default=list)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    evaluated_by: Mapped[str] = mapped_column(String(16), default="local")
    time_seconds: Mapped[float] = mapped_column(Float, default=0.0)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------- #
#  Coding lab
# --------------------------------------------------------------------------- #
class CodingTask(Base):
    __tablename__ = "coding_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(120), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    language: Mapped[str] = mapped_column(String(20), default="python")
    difficulty: Mapped[int] = mapped_column(Integer, default=2)
    level: Mapped[str] = mapped_column(String(20), default="beginner")
    libraries: Mapped[list[str]] = mapped_column(JSON, default=list)
    skill_codes: Mapped[list[str]] = mapped_column(JSON, default=list)
    topic_code: Mapped[str] = mapped_column(String(90), default="", index=True)
    from_scratch: Mapped[bool] = mapped_column(Boolean, default=False)
    banned_imports: Mapped[list[str]] = mapped_column(JSON, default=list)
    starter_code: Mapped[str] = mapped_column(Text, default="")
    test_code: Mapped[str] = mapped_column(Text, default="")
    hints: Mapped[list[str]] = mapped_column(JSON, default=list)
    solution: Mapped[str] = mapped_column(Text, default="")
    solution_explanation: Mapped[str] = mapped_column(Text, default="")
    xp: Mapped[int] = mapped_column(Integer, default=20)
    est_minutes: Mapped[int] = mapped_column(Integer, default=20)
    run_count: Mapped[int] = mapped_column(Integer, default=0)
    solve_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class CodingAttempt(Base):
    __tablename__ = "coding_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    task_id: Mapped[int] = mapped_column(ForeignKey("coding_tasks.id", ondelete="CASCADE"), index=True)
    code: Mapped[str] = mapped_column(Text, default="")
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    total_tests: Mapped[int] = mapped_column(Integer, default=0)
    passed_tests: Mapped[int] = mapped_column(Integer, default=0)
    test_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    stdout: Mapped[str] = mapped_column(Text, default="")
    error: Mapped[str] = mapped_column(Text, default="")
    runtime_ms: Mapped[int] = mapped_column(Integer, default=0)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    reviewed_by_ai: Mapped[bool] = mapped_column(Boolean, default=False)
    review: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


# --------------------------------------------------------------------------- #
#  Projects
# --------------------------------------------------------------------------- #
class Project(Base):
    __tablename__ = "projects"

    id: Mapped[int] = mapped_column(primary_key=True)
    slug: Mapped[str] = mapped_column(String(140), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    target_level: Mapped[str] = mapped_column(String(40), default="junior")  # beginner|junior|middle|strong_middle|senior
    domain: Mapped[str] = mapped_column(String(60), default="general")
    skill_codes: Mapped[list[str]] = mapped_column(JSON, default=list)
    stack: Mapped[list[str]] = mapped_column(JSON, default=list)
    deliverables: Mapped[list[str]] = mapped_column(JSON, default=list)
    milestones: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    rubric: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    guidance: Mapped[str] = mapped_column(Text, default="")
    dataset_hint: Mapped[str] = mapped_column(Text, default="")
    difficulty: Mapped[int] = mapped_column(Integer, default=2)
    est_hours: Mapped[float] = mapped_column(Float, default=12.0)
    generated_for_user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True)
    generated_by: Mapped[str] = mapped_column(String(16), default="seed")
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ProjectEnrollment(Base):
    """A user's active attempt at a project (keeps the curated project reusable)."""

    __tablename__ = "project_enrollments"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    project_id: Mapped[int] = mapped_column(ForeignKey("projects.id", ondelete="CASCADE"), index=True)
    status: Mapped[str] = mapped_column(String(24), default="active")  # active|paused|submitted|evaluated|archived
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    minutes_spent: Mapped[float] = mapped_column(Float, default=0.0)
    repo_note: Mapped[str] = mapped_column(Text, default="")

    project: Mapped["Project"] = relationship()
    tasks: Mapped[list["ProjectTask"]] = relationship(
        back_populates="enrollment", cascade="all, delete-orphan", order_by="ProjectTask.order_index"
    )
    attempts: Mapped[list["ProjectAttempt"]] = relationship(
        back_populates="enrollment", cascade="all, delete-orphan", order_by="ProjectAttempt.id"
    )


class ProjectTask(Base):
    """Milestone inside a project enrollment."""

    __tablename__ = "project_tasks"

    id: Mapped[int] = mapped_column(primary_key=True)
    enrollment_id: Mapped[int] = mapped_column(ForeignKey("project_enrollments.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(300))
    description: Mapped[str] = mapped_column(Text, default="")
    acceptance_criteria: Mapped[list[str]] = mapped_column(JSON, default=list)
    order_index: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default="todo")  # todo|doing|review|done
    review_notes: Mapped[str] = mapped_column(Text, default="")
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    enrollment: Mapped["ProjectEnrollment"] = relationship(back_populates="tasks")


class ProjectAttempt(Base):
    """Every mentor turn / submission / AI suggestion, attributed to author."""

    __tablename__ = "project_attempts"

    KIND_MESSAGE = "mentor_message"
    KIND_SUBMISSION = "submission"
    KIND_AI_SUGGESTION = "ai_suggestion"
    KIND_EVALUATION = "evaluation"

    id: Mapped[int] = mapped_column(primary_key=True)
    enrollment_id: Mapped[int] = mapped_column(ForeignKey("project_enrollments.id", ondelete="CASCADE"), index=True)
    project_task_id: Mapped[int | None] = mapped_column(ForeignKey("project_tasks.id", ondelete="CASCADE"), nullable=True)
    kind: Mapped[str] = mapped_column(String(24), default=KIND_MESSAGE)
    author: Mapped[str] = mapped_column(String(16), default="user")  # user|ai
    content: Mapped[str] = mapped_column(Text, default="")
    user_implemented: Mapped[str] = mapped_column(Text, default="")
    ai_suggested: Mapped[str] = mapped_column(Text, default="")
    files: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    evaluation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    enrollment: Mapped["ProjectEnrollment"] = relationship(back_populates="attempts")


# --------------------------------------------------------------------------- #
#  Exams & interviews
# --------------------------------------------------------------------------- #
class Exam(Base):
    __tablename__ = "exams"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(40), unique=True, index=True)
    title: Mapped[str] = mapped_column(String(200))
    description: Mapped[str] = mapped_column(Text, default="")
    target_level: Mapped[str] = mapped_column(String(40), default="junior")
    min_level: Mapped[int] = mapped_column(Integer, default=0)
    sections: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    question_codes: Mapped[list[str]] = mapped_column(JSON, default=list)
    passing_score: Mapped[float] = mapped_column(Float, default=70.0)
    duration_minutes: Mapped[int] = mapped_column(Integer, default=90)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class ExamAttempt(Base):
    __tablename__ = "exam_attempts"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    exam_id: Mapped[int] = mapped_column(ForeignKey("exams.id", ondelete="CASCADE"), index=True)
    state: Mapped[str] = mapped_column(String(20), default="in_progress")  # in_progress|submitted|graded
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    answers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    item_results: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    section_scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    passed: Mapped[bool] = mapped_column(Boolean, default=False)
    verdict: Mapped[str] = mapped_column(String(40), default="")
    feedback: Mapped[str] = mapped_column(Text, default="")
    strengths: Mapped[list[str]] = mapped_column(JSON, default=list)
    gaps: Mapped[list[str]] = mapped_column(JSON, default=list)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    duration_seconds: Mapped[int] = mapped_column(Integer, default=0)

    exam: Mapped["Exam"] = relationship()


class InterviewSession(Base):
    __tablename__ = "interview_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role_level: Mapped[str] = mapped_column(String(20), default="middle")  # junior|middle|senior
    focus: Mapped[str] = mapped_column(String(60), default="general")
    state: Mapped[str] = mapped_column(String(20), default="active")  # active|finished
    turns: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    questions_asked: Mapped[list[int]] = mapped_column(JSON, default=list)
    report: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    scores: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# --------------------------------------------------------------------------- #
#  Study loop: sessions, plans, reviews, notes, memory
# --------------------------------------------------------------------------- #
class StudySession(Base):
    __tablename__ = "study_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    activity: Mapped[str] = mapped_column(String(32), default="learn")  # learn|practice|coding|project|exam|interview|review|teacher|diagnostic
    topic_code: Mapped[str] = mapped_column(String(90), default="")
    item_ref: Mapped[str] = mapped_column(String(120), default="")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    ended_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    active_seconds: Mapped[int] = mapped_column(Integer, default=0)
    xp: Mapped[int] = mapped_column(Integer, default=0)
    items_done: Mapped[int] = mapped_column(Integer, default=0)
    ai_used: Mapped[bool] = mapped_column(Boolean, default=False)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)


class DailyPlan(Base):
    __tablename__ = "daily_plans"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    plan_date: Mapped[date] = mapped_column(Date, index=True)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    rationale: Mapped[str] = mapped_column(Text, default="")
    minutes_budget: Mapped[int] = mapped_column(Integer, default=60)
    completed_ids: Mapped[list[str]] = mapped_column(JSON, default=list)
    generated_by: Mapped[str] = mapped_column(String(16), default="engine")
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "plan_date", name="uq_plan_user_date"),)


class ReviewSchedule(Base):
    """FSRS/SM-2-like scheduling for skills, questions and chunks."""

    __tablename__ = "review_schedules"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    item_type: Mapped[str] = mapped_column(String(20), default="skill")  # skill|question|practice|chunk
    item_id: Mapped[int] = mapped_column(Integer, default=0)
    item_code: Mapped[str] = mapped_column(String(120), default="", index=True)

    last_review: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    next_review: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    interval_days: Mapped[float] = mapped_column(Float, default=0.0)
    ease_factor: Mapped[float] = mapped_column(Float, default=2.5)
    repetitions: Mapped[int] = mapped_column(Integer, default=0)
    lapses: Mapped[int] = mapped_column(Integer, default=0)
    streak: Mapped[int] = mapped_column(Integer, default=0)
    difficulty: Mapped[float] = mapped_column(Float, default=5.0)
    quality_last: Mapped[int] = mapped_column(Integer, default=0)
    suspended: Mapped[bool] = mapped_column(Boolean, default=False)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "item_type", "item_id", name="uq_review_item"),)


class UserNote(Base):
    __tablename__ = "user_notes"

    KIND_NOTE = "note"
    KIND_BOOKMARK = "bookmark"
    KIND_MISTAKE = "mistake"
    KIND_CONCEPT = "concept"
    KIND_QUESTION = "question"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    kind: Mapped[str] = mapped_column(String(20), default=KIND_NOTE)
    title: Mapped[str] = mapped_column(String(300), default="")
    body: Mapped[str] = mapped_column(Text, default="")
    topic_code: Mapped[str] = mapped_column(String(90), default="")
    document_id: Mapped[int | None] = mapped_column(ForeignKey("documents.id", ondelete="SET NULL"), nullable=True)
    chunk_id: Mapped[int | None] = mapped_column(ForeignKey("document_chunks.id", ondelete="SET NULL"), nullable=True)
    excerpt: Mapped[str] = mapped_column(Text, default="")
    citation: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tags: Mapped[list[str]] = mapped_column(JSON, default=list)
    pinned: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class LearningJournal(Base):
    __tablename__ = "learning_journal"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    entry_date: Mapped[date] = mapped_column(Date, index=True)
    content: Mapped[str] = mapped_column(Text, default="")
    mood: Mapped[int] = mapped_column(Integer, default=3)  # 1..5
    minutes: Mapped[int] = mapped_column(Integer, default=0)
    ai_analysis: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    gaps_detected: Mapped[list[str]] = mapped_column(JSON, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class LearningMemory(Base):
    """Long-lived, compact memory about the learner (never raw secrets)."""

    __tablename__ = "learning_memory"


    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    category: Mapped[str] = mapped_column(String(32), default="fact")
    # strength|weakness|preference|topic|project|exam|mistake|history|goal
    key: Mapped[str] = mapped_column(String(160), default="", index=True)
    value: Mapped[str] = mapped_column(Text, default="")
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    weight: Mapped[float] = mapped_column(Float, default=1.0)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "category", "key", name="uq_memory_key"),)


class JournalEntry(Base):
    """One reflective entry per day (the daily review ritual)."""

    __tablename__ = "journal_entries"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    entry_date: Mapped[date] = mapped_column(Date, index=True)
    content: Mapped[str] = mapped_column(Text, default="")
    mood: Mapped[int] = mapped_column(Integer, default=3)
    minutes: Mapped[int] = mapped_column(Integer, default=0)
    word_count: Mapped[int] = mapped_column(Integer, default=0)
    # deterministic analysis written back by the offline engine
    topics: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    gaps: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    strengths: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    risk_flags: Mapped[list[str]] = mapped_column(JSON, default=list)
    ai_analysis: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "entry_date", name="uq_journal_user_date"),)


class DiagnosticSession(Base):
    __tablename__ = "diagnostic_sessions"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    items: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    answers: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    result: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    status: Mapped[str] = mapped_column(String(20), default="in_progress")
    started_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    completed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


# --------------------------------------------------------------------------- #
#  AI Teacher conversations
# --------------------------------------------------------------------------- #
class TeacherConversation(Base):
    __tablename__ = "teacher_conversations"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    title: Mapped[str] = mapped_column(String(300), default="New session")
    mode: Mapped[str] = mapped_column(String(32), default="explain")
    topic_code: Mapped[str] = mapped_column(String(90), default="")
    level: Mapped[int] = mapped_column(Integer, default=0)
    context: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    message_count: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    messages: Mapped[list["TeacherMessage"]] = relationship(
        back_populates="conversation", cascade="all, delete-orphan", order_by="TeacherMessage.id"
    )


class TeacherMessage(Base):
    __tablename__ = "teacher_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    conversation_id: Mapped[int] = mapped_column(ForeignKey("teacher_conversations.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(16), default="user")  # user|assistant|system
    content: Mapped[str] = mapped_column(Text, default="")
    mode: Mapped[str] = mapped_column(String(32), default="explain")
    sources: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    attachments: Mapped[list[dict[str, Any]]] = mapped_column(JSON, default=list)
    pending_item: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    engine: Mapped[str] = mapped_column(String(16), default="local")  # ai|local|hybrid
    ai_request_id: Mapped[int | None] = mapped_column(ForeignKey("ai_requests.id", ondelete="SET NULL"), nullable=True)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    conversation: Mapped["TeacherConversation"] = relationship(back_populates="messages")


# --------------------------------------------------------------------------- #
#  AI plumbing: usage, cost control, cache
# --------------------------------------------------------------------------- #
class AIRequest(Base):
    __tablename__ = "ai_requests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int | None] = mapped_column(ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    provider: Mapped[str] = mapped_column(String(32), default="none")
    method: Mapped[str] = mapped_column(String(32), default="generate", index=True)
    model: Mapped[str] = mapped_column(String(120), default="")
    prompt_hash: Mapped[str] = mapped_column(String(64), default="", index=True)
    prompt_chars: Mapped[int] = mapped_column(Integer, default=0)
    completion_chars: Mapped[int] = mapped_column(Integer, default=0)
    input_tokens: Mapped[int] = mapped_column(Integer, default=0)
    output_tokens: Mapped[int] = mapped_column(Integer, default=0)
    total_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    latency_ms: Mapped[int] = mapped_column(Integer, default=0)
    status: Mapped[str] = mapped_column(String(24), default="ok")  # ok|error|cached|limited|disabled
    error: Mapped[str] = mapped_column(Text, default="")
    cached: Mapped[bool] = mapped_column(Boolean, default=False)
    conversation_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    meta: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)


class AIUsage(Base):
    __tablename__ = "ai_usage"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    requests: Mapped[int] = mapped_column(Integer, default=0)
    cached_hits: Mapped[int] = mapped_column(Integer, default=0)
    errors: Mapped[int] = mapped_column(Integer, default=0)
    limited: Mapped[int] = mapped_column(Integer, default=0)
    prompt_tokens: Mapped[int] = mapped_column(Integer, default=0)
    completion_tokens: Mapped[int] = mapped_column(Integer, default=0)
    estimated_cost: Mapped[float] = mapped_column(Float, default=0.0)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "day", name="uq_usage_user_day"),)


class AICacheEntry(Base):
    __tablename__ = "ai_cache"

    id: Mapped[int] = mapped_column(primary_key=True)
    cache_key: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    method: Mapped[str] = mapped_column(String(32), default="generate")
    model: Mapped[str] = mapped_column(String(120), default="")
    request_json: Mapped[str] = mapped_column(Text, default="")
    response_text: Mapped[str] = mapped_column(Text, default="")
    response_json: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    tokens: Mapped[int] = mapped_column(Integer, default=0)
    hits: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    last_hit_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


def register_models() -> None:  # pragma: no cover - convenience for Alembic autogen
    _ = (User, Document, DocumentChunk, Source, Topic, Skill, SkillDependency, UserSkill,
         Question, QuestionAttempt, PracticeTask, PracticeAttempt, CodingTask, CodingAttempt,
         Project, ProjectEnrollment, ProjectTask, ProjectAttempt, Exam, ExamAttempt,
         InterviewSession, StudySession, DailyPlan, ReviewSchedule, UserNote, LearningJournal,
         LearningMemory, DiagnosticSession, TeacherConversation, TeacherMessage,
         AIRequest, AIUsage, AICacheEntry, EmbeddingStat, Source)
