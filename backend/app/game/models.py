"""Game-layer persistence: knowledge universe, progression, economy, events."""

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
    String,
    Text,
    UniqueConstraint,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db import Base


def utcnow() -> datetime:
    return datetime.utcnow()


# --------------------------------------------------------------------------- #
#  Knowledge universe (content)
# --------------------------------------------------------------------------- #
class GameWorld(Base):
    """A subject world: Программирование, Физика, Математика, ..."""

    __tablename__ = "game_worlds"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(60), unique=True, index=True)
    name_ru: Mapped[str] = mapped_column(String(120))
    tagline_ru: Mapped[str] = mapped_column(String(300), default="")
    color: Mapped[str] = mapped_column(String(16), default="#38bdf8")
    glyph: Mapped[str] = mapped_column(String(8), default="◆")
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class GameNode(Base):
    """A concept node inside a world. Bosses are nodes with is_boss=True."""

    __tablename__ = "game_nodes"

    id: Mapped[int] = mapped_column(primary_key=True)
    world_code: Mapped[str] = mapped_column(String(60), index=True)
    code: Mapped[str] = mapped_column(String(90), unique=True, index=True)
    name_ru: Mapped[str] = mapped_column(String(160))
    short_ru: Mapped[str] = mapped_column(String(300), default="")
    theory_ru: Mapped[str] = mapped_column(Text, default="")  # markdown-ish
    difficulty: Mapped[int] = mapped_column(Integer, default=1)  # 1..5
    xp_reward: Mapped[int] = mapped_column(Integer, default=40)  # on mastery
    is_boss: Mapped[bool] = mapped_column(Boolean, default=False)
    is_root: Mapped[bool] = mapped_column(Boolean, default=False)
    # map layout, 0..100 local world coordinates
    pos_x: Mapped[float] = mapped_column(Float, default=50.0)
    pos_y: Mapped[float] = mapped_column(Float, default=50.0)
    order_index: Mapped[int] = mapped_column(Integer, default=0)


class GameEdge(Base):
    """Prerequisite edge: `from_code` must be mastered before `to_code` unlocks."""

    __tablename__ = "game_edges"

    id: Mapped[int] = mapped_column(primary_key=True)
    from_code: Mapped[str] = mapped_column(String(90), index=True)
    to_code: Mapped[str] = mapped_column(String(90), index=True)

    __table_args__ = (UniqueConstraint("from_code", "to_code", name="uq_game_edge"),)


class GameQuestion(Base):
    __tablename__ = "game_questions"

    id: Mapped[int] = mapped_column(primary_key=True)
    node_code: Mapped[str] = mapped_column(String(90), index=True)
    qtype: Mapped[str] = mapped_column(String(16), default="mcq")  # mcq|input|open
    prompt_ru: Mapped[str] = mapped_column(Text)
    options: Mapped[list[str]] = mapped_column(JSON, default=list)
    correct_index: Mapped[int | None] = mapped_column(Integer, nullable=True)
    expected: Mapped[str] = mapped_column(String(300), default="")  # input answers
    tolerance: Mapped[float] = mapped_column(Float, default=0.0)
    keywords: Mapped[list[str]] = mapped_column(JSON, default=list)  # open answers
    explanation_ru: Mapped[str] = mapped_column(Text, default="")
    hints: Mapped[list[str]] = mapped_column(JSON, default=list)
    difficulty: Mapped[int] = mapped_column(Integer, default=1)
    content_hash: Mapped[str] = mapped_column(String(64), default="", index=True)


# --------------------------------------------------------------------------- #
#  Player progression (server-authoritative)
# --------------------------------------------------------------------------- #
class PlayerState(Base):
    __tablename__ = "player_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), unique=True, index=True)

    xp: Mapped[int] = mapped_column(Integer, default=0)
    level: Mapped[int] = mapped_column(Integer, default=1)
    knowledge_power: Mapped[int] = mapped_column(Integer, default=0)
    streak_days: Mapped[int] = mapped_column(Integer, default=0)
    best_streak: Mapped[int] = mapped_column(Integer, default=0)
    last_active_day: Mapped[date | None] = mapped_column(Date, nullable=True)
    streak_shields: Mapped[int] = mapped_column(Integer, default=1)  # streak protection (§45)

    selected_title: Mapped[str] = mapped_column(String(80), default="")
    selected_frame: Mapped[str] = mapped_column(String(80), default="")
    interests: Mapped[list[str]] = mapped_column(JSON, default=list)
    goal_ru: Mapped[str] = mapped_column(String(300), default="")
    start_level: Mapped[str] = mapped_column(String(30), default="beginner")
    intensity: Mapped[str] = mapped_column(String(20), default="normal")  # light|normal|deep
    onboarded: Mapped[bool] = mapped_column(Boolean, default=False)

    answers_total: Mapped[int] = mapped_column(Integer, default=0)
    answers_correct: Mapped[int] = mapped_column(Integer, default=0)
    bosses_defeated: Mapped[int] = mapped_column(Integer, default=0)
    chests_opened: Mapped[int] = mapped_column(Integer, default=0)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)


class PlayerNode(Base):
    """Per-user state of a knowledge node."""

    __tablename__ = "player_nodes"

    ST_LOCKED = "locked"
    ST_AVAILABLE = "available"
    ST_ACTIVE = "active"
    ST_MASTERED = "mastered"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    node_code: Mapped[str] = mapped_column(String(90), index=True)
    status: Mapped[str] = mapped_column(String(16), default=ST_LOCKED)
    progress: Mapped[float] = mapped_column(Float, default=0.0)  # 0..100
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    correct: Mapped[int] = mapped_column(Integer, default=0)
    theory_read: Mapped[bool] = mapped_column(Boolean, default=False)
    mastered_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "node_code", name="uq_player_node"),)


class PlayerQuestionState(Base):
    """Tracks which questions a user already answered correctly (no XP farming)."""

    __tablename__ = "player_question_states"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    question_id: Mapped[int] = mapped_column(ForeignKey("game_questions.id", ondelete="CASCADE"), index=True)
    solved: Mapped[bool] = mapped_column(Boolean, default=False)
    attempts: Mapped[int] = mapped_column(Integer, default=0)
    hints_used: Mapped[int] = mapped_column(Integer, default=0)
    last_answer: Mapped[str] = mapped_column(Text, default="")
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "question_id", name="uq_player_question"),)


class PlayerQuest(Base):
    """Daily quests, generated server-side per user per day."""

    __tablename__ = "player_quests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    day: Mapped[date] = mapped_column(Date, index=True)
    code: Mapped[str] = mapped_column(String(60))  # solve_n | master_node | talk_teacher | review
    title_ru: Mapped[str] = mapped_column(String(200))
    target: Mapped[int] = mapped_column(Integer, default=1)
    progress: Mapped[int] = mapped_column(Integer, default=0)
    xp_reward: Mapped[int] = mapped_column(Integer, default=25)
    chest_on_complete: Mapped[bool] = mapped_column(Boolean, default=False)
    completed: Mapped[bool] = mapped_column(Boolean, default=False)
    claimed: Mapped[bool] = mapped_column(Boolean, default=False)

    __table_args__ = (UniqueConstraint("user_id", "day", "code", name="uq_player_quest"),)


# --------------------------------------------------------------------------- #
#  Economy: items, chests, inventory, achievements
# --------------------------------------------------------------------------- #
class GameItem(Base):
    __tablename__ = "game_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(90), unique=True, index=True)
    name_ru: Mapped[str] = mapped_column(String(160))
    description_ru: Mapped[str] = mapped_column(String(400), default="")
    rarity: Mapped[str] = mapped_column(String(16), default="common")
    # collectible | title | frame | theme | badge
    kind: Mapped[str] = mapped_column(String(24), default="collectible")
    world_code: Mapped[str] = mapped_column(String(60), default="", index=True)
    glyph: Mapped[str] = mapped_column(String(8), default="◇")


class PlayerItem(Base):
    __tablename__ = "player_items"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    item_code: Mapped[str] = mapped_column(String(90), index=True)
    count: Mapped[int] = mapped_column(Integer, default=1)
    acquired_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "item_code", name="uq_player_item"),)


class PlayerChest(Base):
    __tablename__ = "player_chests"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    source: Mapped[str] = mapped_column(String(60), default="")  # node|boss|quest|level|onboarding
    world_code: Mapped[str] = mapped_column(String(60), default="")
    opened: Mapped[bool] = mapped_column(Boolean, default=False)
    rarity: Mapped[str] = mapped_column(String(16), default="")  # rolled server-side on open
    item_code: Mapped[str] = mapped_column(String(90), default="")
    xp_bonus: Mapped[int] = mapped_column(Integer, default=0)
    granted_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    opened_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class GameAchievement(Base):
    __tablename__ = "game_achievements"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(90), unique=True, index=True)
    name_ru: Mapped[str] = mapped_column(String(160))
    description_ru: Mapped[str] = mapped_column(String(400), default="")
    glyph: Mapped[str] = mapped_column(String(8), default="✦")
    xp_reward: Mapped[int] = mapped_column(Integer, default=30)
    secret: Mapped[bool] = mapped_column(Boolean, default=False)


class PlayerAchievement(Base):
    __tablename__ = "player_achievements"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    achievement_code: Mapped[str] = mapped_column(String(90), index=True)
    unlocked_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)

    __table_args__ = (UniqueConstraint("user_id", "achievement_code", name="uq_player_achievement"),)


# --------------------------------------------------------------------------- #
#  Events (audit) & teacher chat
# --------------------------------------------------------------------------- #
class GameEvent(Base):
    """Audit trail of everything that grants progression (spec §73)."""

    __tablename__ = "game_events"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    etype: Mapped[str] = mapped_column(String(60), index=True)  # xp.granted, level.up, ...
    payload: Mapped[dict[str, Any]] = mapped_column(JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)

    __table_args__ = (Index("ix_game_events_user_created", "user_id", "created_at"),)


class GameChatMessage(Base):
    __tablename__ = "game_chat_messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    user_id: Mapped[int] = mapped_column(ForeignKey("users.id", ondelete="CASCADE"), index=True)
    role: Mapped[str] = mapped_column(String(12))  # user|teacher
    mode: Mapped[str] = mapped_column(String(24), default="explain")
    node_code: Mapped[str] = mapped_column(String(90), default="")
    text: Mapped[str] = mapped_column(Text)
    via_ai: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow, index=True)
