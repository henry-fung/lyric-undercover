from datetime import datetime, timezone

from sqlalchemy import Boolean, DateTime, ForeignKey, Integer, String, Text, UniqueConstraint
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(DeclarativeBase):
    pass


def utcnow() -> datetime:
    """Store UTC as a naive value because SQLite has no native timezone type."""
    return datetime.now(timezone.utc).replace(tzinfo=None)


class Room(Base):
    __tablename__ = "rooms"

    id: Mapped[int] = mapped_column(primary_key=True)
    code: Mapped[str] = mapped_column(String(8), unique=True, index=True)
    state: Mapped[str] = mapped_column(String(16), default="lobby")
    host_player_id: Mapped[int | None] = mapped_column(ForeignKey("players.id"), nullable=True)
    winner: Mapped[str | None] = mapped_column(String(16), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class Player(Base):
    __tablename__ = "players"
    __table_args__ = (UniqueConstraint("room_id", "name", name="uq_player_name_per_room"),)

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"), index=True)
    name: Mapped[str] = mapped_column(String(40))
    session_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    recovery_hash: Mapped[str] = mapped_column(String(64), unique=True)
    role: Mapped[str | None] = mapped_column(String(16), nullable=True)
    is_alive: Mapped[bool] = mapped_column(Boolean, default=True)
    joined_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)


class GameRound(Base):
    __tablename__ = "rounds"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"), index=True)
    number: Mapped[int] = mapped_column(Integer)
    theme: Mapped[str] = mapped_column(String(80))
    civilian_prompt: Mapped[str] = mapped_column(Text)
    undercover_prompt: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
    closed_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class Elimination(Base):
    __tablename__ = "eliminations"

    id: Mapped[int] = mapped_column(primary_key=True)
    room_id: Mapped[int] = mapped_column(ForeignKey("rooms.id"), index=True)
    round_id: Mapped[int] = mapped_column(ForeignKey("rounds.id"))
    player_id: Mapped[int] = mapped_column(ForeignKey("players.id"), unique=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=utcnow)
