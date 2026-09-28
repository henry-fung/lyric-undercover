from __future__ import annotations

import re
import secrets
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import Depends, FastAPI, Form, HTTPException, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, desc, make_url, select, text
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session, sessionmaker

from app.config import Settings
from app.models import Base, Elimination, Game, GamePlayer, GameRound, Player, Room, utcnow
from app.scenes import OpenAICompatibleSceneGenerator, SceneGenerationError, SceneGenerator
from app.security import hash_token, new_recovery_code, new_token

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


@dataclass(frozen=True)
class PlayerView:
    """Template/API-safe view of a player in the current or historical game."""

    id: int
    name: str
    role: str | None = None
    is_alive: bool = True


def upgrade_sqlite_schema(engine) -> None:
    """Upgrade pre-rematch SQLite databases without discarding their one game.

    The application intentionally has no external migration dependency.  This
    small, idempotent upgrade only covers the schema that shipped before games
    were separated from rooms.
    """
    if engine.dialect.name != "sqlite":
        return

    with engine.begin() as connection:
        round_columns = {row[1] for row in connection.execute(text("PRAGMA table_info(rounds)"))}
        elimination_columns = {row[1] for row in connection.execute(text("PRAGMA table_info(eliminations)"))}

        if "game_id" not in round_columns:
            connection.execute(text("ALTER TABLE rounds ADD COLUMN game_id INTEGER REFERENCES games(id)"))

        # The old player_id UNIQUE constraint prevented eliminating the same
        # person in a later game, so SQLite needs a table rebuild to replace it.
        if "game_id" not in elimination_columns:
            connection.execute(
                text(
                    """
                    CREATE TABLE eliminations_rematch (
                        id INTEGER NOT NULL PRIMARY KEY,
                        room_id INTEGER NOT NULL REFERENCES rooms(id),
                        game_id INTEGER REFERENCES games(id),
                        round_id INTEGER NOT NULL REFERENCES rounds(id),
                        player_id INTEGER NOT NULL REFERENCES players(id),
                        created_at DATETIME NOT NULL,
                        CONSTRAINT uq_elimination_player_per_game UNIQUE (game_id, player_id)
                    )
                    """
                )
            )
            connection.execute(
                text(
                    """
                    INSERT INTO eliminations_rematch (id, room_id, round_id, player_id, created_at)
                    SELECT id, room_id, round_id, player_id, created_at FROM eliminations
                    """
                )
            )
            connection.execute(text("DROP TABLE eliminations"))
            connection.execute(text("ALTER TABLE eliminations_rematch RENAME TO eliminations"))
            connection.execute(text("CREATE INDEX ix_eliminations_room_id ON eliminations (room_id)"))
            connection.execute(text("CREATE INDEX ix_eliminations_game_id ON eliminations (game_id)"))

        legacy_room_ids = connection.execute(
            text("SELECT DISTINCT room_id FROM rounds WHERE game_id IS NULL")
        ).scalars().all()
        for room_id in legacy_room_ids:
            room = connection.execute(
                text("SELECT state, winner, created_at FROM rooms WHERE id = :room_id"), {"room_id": room_id}
            ).mappings().one()
            game_id = connection.execute(
                text(
                    """
                    INSERT INTO games (room_id, number, winner, created_at, finished_at)
                    VALUES (:room_id, 1, :winner, :created_at, :finished_at)
                    """
                ),
                {
                    "room_id": room_id,
                    "winner": room["winner"],
                    "created_at": room["created_at"],
                    "finished_at": utcnow() if room["state"] == "finished" else None,
                },
            ).lastrowid
            connection.execute(text("UPDATE rounds SET game_id = :game_id WHERE room_id = :room_id"), {"game_id": game_id, "room_id": room_id})
            connection.execute(
                text("UPDATE eliminations SET game_id = :game_id WHERE room_id = :room_id"),
                {"game_id": game_id, "room_id": room_id},
            )
            legacy_players = connection.execute(
                text("SELECT id, role, is_alive FROM players WHERE room_id = :room_id"), {"room_id": room_id}
            ).mappings()
            for player in legacy_players:
                connection.execute(
                    text(
                        """
                        INSERT INTO game_players (game_id, player_id, role, is_alive)
                        VALUES (:game_id, :player_id, :role, :is_alive)
                        """
                    ),
                    {
                        "game_id": game_id,
                        "player_id": player["id"],
                        "role": player["role"] or "civilian",
                        "is_alive": player["is_alive"],
                    },
                )


class ConnectionManager:
    def __init__(self) -> None:
        self.connections: dict[str, set[WebSocket]] = {}

    async def connect(self, code: str, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.setdefault(code, set()).add(websocket)

    def disconnect(self, code: str, websocket: WebSocket) -> None:
        self.connections.get(code, set()).discard(websocket)

    async def broadcast(self, code: str, event: str) -> None:
        stale: list[WebSocket] = []
        for websocket in self.connections.get(code, set()):
            try:
                await websocket.send_json({"event": event})
            except Exception:
                stale.append(websocket)
        for websocket in stale:
            self.disconnect(code, websocket)


def create_app(settings: Settings | None = None, generator: SceneGenerator | None = None) -> FastAPI:
    settings = settings or Settings()
    if settings.database_url.startswith("sqlite"):
        database_path = make_url(settings.database_url).database
        if database_path and database_path != ":memory:":
            Path(database_path).parent.mkdir(parents=True, exist_ok=True)
    engine = create_engine(settings.database_url, connect_args={"check_same_thread": False} if settings.database_url.startswith("sqlite") else {})
    sessions = sessionmaker(bind=engine, expire_on_commit=False)
    manager = ConnectionManager()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        Base.metadata.create_all(engine)
        upgrade_sqlite_schema(engine)
        yield
        engine.dispose()

    app = FastAPI(title="歌词谁是卧底", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    app.state.settings = settings
    app.state.sessions = sessions
    app.state.generator = generator or OpenAICompatibleSceneGenerator(settings)
    app.state.manager = manager

    def get_db():
        db = sessions()
        try:
            yield db
        finally:
            db.close()

    def current_player(request: Request, db: Session) -> Player | None:
        token = request.cookies.get(settings.session_cookie_name)
        if not token:
            return None
        return db.scalar(select(Player).where(Player.session_hash == hash_token(token)))

    def require_player(request: Request, db: Session, room: Room) -> Player:
        player = current_player(request, db)
        if not player or player.room_id != room.id:
            raise HTTPException(403, "请先加入此房间。")
        return player

    def require_host(request: Request, db: Session, room: Room) -> Player:
        player = require_player(request, db, room)
        if room.host_player_id != player.id:
            raise HTTPException(403, "只有房主可以执行此操作。")
        return player

    def find_room(db: Session, code: str) -> Room:
        room = db.scalar(select(Room).where(Room.code == code.upper()))
        if not room:
            raise HTTPException(404, "房间不存在。")
        return room

    def set_session(response: RedirectResponse, token: str) -> None:
        response.set_cookie(
            settings.session_cookie_name,
            token,
            httponly=True,
            samesite="lax",
            secure=settings.session_secure,
            max_age=60 * 60 * 24 * 14,
        )

    def room_code(db: Session) -> str:
        alphabet = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"
        for _ in range(12):
            code = "".join(secrets.choice(alphabet) for _ in range(6))
            if not db.scalar(select(Room.id).where(Room.code == code)):
                return code
        raise RuntimeError("无法分配房间号")

    def latest_game(db: Session, room: Room) -> Game | None:
        return db.scalar(select(Game).where(Game.room_id == room.id).order_by(desc(Game.number)))

    def game_players(db: Session, game: Game) -> list[PlayerView]:
        rows = db.execute(
            select(GamePlayer, Player)
            .join(Player, Player.id == GamePlayer.player_id)
            .where(GamePlayer.game_id == game.id)
            .order_by(Player.joined_at)
        ).all()
        return [
            PlayerView(id=player.id, name=player.name, role=participant.role, is_alive=participant.is_alive)
            for participant, player in rows
        ]

    def room_players(db: Session, room: Room) -> list[PlayerView]:
        players = db.scalars(select(Player).where(Player.room_id == room.id).order_by(Player.joined_at)).all()
        return [PlayerView(id=player.id, name=player.name) for player in players]

    def current_participant(db: Session, game: Game | None, player: Player) -> PlayerView | None:
        if not game:
            return None
        participant = db.scalar(
            select(GamePlayer).where(GamePlayer.game_id == game.id, GamePlayer.player_id == player.id)
        )
        if not participant:
            return None
        return PlayerView(id=player.id, name=player.name, role=participant.role, is_alive=participant.is_alive)

    def finished_games(db: Session, room: Room) -> list[Game]:
        return db.scalars(
            select(Game).where(Game.room_id == room.id, Game.finished_at.is_not(None)).order_by(desc(Game.number))
        ).all()

    def public_state(db: Session, room: Room) -> dict:
        game = latest_game(db, room) if room.state in {"playing", "finished"} else None
        players = game_players(db, game) if game else room_players(db, room)
        active_round = (
            db.scalar(select(GameRound).where(GameRound.game_id == game.id).order_by(desc(GameRound.number)))
            if game
            else None
        )
        return {
            "code": room.code,
            "state": room.state,
            "winner": room.winner,
            "round": active_round.number if active_round else None,
            "round_closed": bool(active_round and active_round.closed_at),
            "players": [{"name": p.name, "is_alive": p.is_alive} for p in players],
        }

    def render(request: Request, template: str, **context):
        return templates.TemplateResponse(request, template, {"settings": settings, **context})

    def validate_name(name: str) -> str:
        name = name.strip()
        if not 1 <= len(name) <= 40:
            raise HTTPException(422, "玩家名须为 1 至 40 个字符。")
        return name

    def recent_summaries(db: Session, game: Game) -> list[str]:
        rounds = db.scalars(
            select(GameRound).where(GameRound.game_id == game.id).order_by(desc(GameRound.number)).limit(6)
        ).all()
        return [round_.theme for round_ in rounds]

    def add_round(db: Session, room: Room, game: Game, topic_mode: str, custom_topic: str) -> GameRound:
        if topic_mode not in {"random", "theme", "custom"}:
            raise HTTPException(422, "未知的主题选项。")
        topic = None
        if topic_mode == "theme":
            topic = custom_topic.strip()
        elif topic_mode == "custom":
            topic = custom_topic.strip()
        if topic_mode != "random" and not topic:
            raise HTTPException(422, "请选择或填写主题。")
        if topic and len(topic) > 80:
            raise HTTPException(422, "主题不能超过 80 个字符。")
        try:
            pair = app.state.generator.generate(topic, recent_summaries(db, game))
        except SceneGenerationError as exc:
            raise HTTPException(503, str(exc)) from exc
        number = (db.scalar(select(GameRound.number).where(GameRound.game_id == game.id).order_by(desc(GameRound.number))) or 0) + 1
        round_ = GameRound(
            room_id=room.id,
            game_id=game.id,
            number=number,
            theme=pair.summary,
            civilian_prompt=pair.civilian_prompt,
            undercover_prompt=pair.undercover_prompt,
        )
        db.add(round_)
        return round_

    @app.get("/", name="home")
    def home(request: Request):
        return render(request, "home.html")

    @app.post("/rooms")
    async def create_room(
        request: Request,
        name: Annotated[str, Form()],
        invite_code: Annotated[str, Form()],
        db: Session = Depends(get_db),
    ):
        if not secrets.compare_digest(invite_code, settings.invite_code):
            return render(request, "home.html", error="邀请码不正确。")
        name = validate_name(name)
        room = Room(code=room_code(db))
        token, recovery = new_token(), new_recovery_code()
        player = Player(name=name, session_hash=hash_token(token), recovery_hash=hash_token(recovery), room_id=0)
        db.add(room)
        db.flush()
        player.room_id = room.id
        db.add(player)
        db.flush()
        room.host_player_id = player.id
        db.commit()
        response = RedirectResponse(f"/rooms/{room.code}/play?recovery={recovery}", status_code=303)
        set_session(response, token)
        return response

    @app.get("/rooms/{code}")
    def room_landing(request: Request, code: str, db: Session = Depends(get_db)):
        room = find_room(db, code)
        player = current_player(request, db)
        if player and player.room_id == room.id:
            return RedirectResponse(f"/rooms/{room.code}/play", status_code=303)
        return render(request, "join.html", room=room)

    @app.post("/rooms/{code}/join")
    async def join_room(
        request: Request,
        code: str,
        name: Annotated[str, Form()],
        invite_code: Annotated[str, Form()],
        db: Session = Depends(get_db),
    ):
        room = find_room(db, code)
        if room.state != "lobby":
            return render(request, "join.html", room=room, error="游戏已经开始，不能再加入。")
        if not secrets.compare_digest(invite_code, settings.invite_code):
            return render(request, "join.html", room=room, error="邀请码不正确。")
        name = validate_name(name)
        token, recovery = new_token(), new_recovery_code()
        player = Player(room_id=room.id, name=name, session_hash=hash_token(token), recovery_hash=hash_token(recovery))
        db.add(player)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            return render(request, "join.html", room=room, error="这个名字已被使用。")
        await manager.broadcast(room.code, "player_joined")
        response = RedirectResponse(f"/rooms/{room.code}/play?recovery={recovery}", status_code=303)
        set_session(response, token)
        return response

    @app.post("/rooms/{code}/recover")
    def recover_player(
        request: Request,
        code: str,
        name: Annotated[str, Form()],
        recovery_code: Annotated[str, Form()],
        db: Session = Depends(get_db),
    ):
        room = find_room(db, code)
        player = db.scalar(select(Player).where(Player.room_id == room.id, Player.name == name.strip()))
        if not player or not secrets.compare_digest(player.recovery_hash, hash_token(recovery_code.strip().upper())):
            return render(request, "join.html", room=room, error="姓名或恢复码不正确。")
        token = new_token()
        player.session_hash = hash_token(token)
        db.commit()
        response = RedirectResponse(f"/rooms/{room.code}/play", status_code=303)
        set_session(response, token)
        return response

    @app.get("/rooms/{code}/play")
    def play(request: Request, code: str, recovery: str | None = None, db: Session = Depends(get_db)):
        room = find_room(db, code)
        player = require_player(request, db, room)
        game = latest_game(db, room) if room.state in {"playing", "finished"} else None
        round_ = (
            db.scalar(select(GameRound).where(GameRound.game_id == game.id).order_by(desc(GameRound.number)))
            if game
            else None
        )
        players = game_players(db, game) if game else room_players(db, room)
        return render(
            request,
            "play.html",
            room=room,
            player=player,
            players=players,
            round=round_,
            game=game,
            participant=current_participant(db, game, player),
            history=finished_games(db, room),
            recovery_code=recovery,
            is_host=room.host_player_id == player.id,
        )

    @app.get("/rooms/{code}/host")
    def host_panel(request: Request, code: str, db: Session = Depends(get_db)):
        room = find_room(db, code)
        require_host(request, db, room)
        game = latest_game(db, room) if room.state in {"playing", "finished"} else None
        players = game_players(db, game) if game else room_players(db, room)
        round_ = (
            db.scalar(select(GameRound).where(GameRound.game_id == game.id).order_by(desc(GameRound.number)))
            if game
            else None
        )
        return render(
            request,
            "host.html",
            room=room,
            players=players,
            round=round_,
            game=game,
            history=finished_games(db, room),
        )

    @app.post("/rooms/{code}/start")
    async def start_game(
        request: Request,
        code: str,
        undercover_count: Annotated[int, Form()],
        topic_mode: Annotated[str, Form()],
        topic: Annotated[str, Form()] = "",
        db: Session = Depends(get_db),
    ):
        room = find_room(db, code)
        require_host(request, db, room)
        players = db.scalars(select(Player).where(Player.room_id == room.id)).all()
        if room.state != "lobby":
            raise HTTPException(409, "游戏已经开始。")
        if len(players) < 3 or not 1 <= undercover_count <= len(players) - 2:
            raise HTTPException(422, "至少需要 3 名玩家，卧底人数须在 1 到玩家数减 2 之间。")
        game_number = (db.scalar(select(Game.number).where(Game.room_id == room.id).order_by(desc(Game.number))) or 0) + 1
        game = Game(room_id=room.id, number=game_number)
        db.add(game)
        db.flush()
        try:
            add_round(db, room, game, topic_mode, topic)
        except HTTPException as exc:
            db.rollback()
            return render(request, "host.html", room=room, players=room_players(db, room), round=None, game=None, error=exc.detail)
        undercover_ids = set(secrets.SystemRandom().sample([p.id for p in players], undercover_count))
        for player in players:
            player.role = "undercover" if player.id in undercover_ids else "civilian"
            player.is_alive = True
            db.add(GamePlayer(game_id=game.id, player_id=player.id, role=player.role, is_alive=True))
        room.state = "playing"
        room.winner = None
        db.commit()
        await manager.broadcast(room.code, "game_started")
        return RedirectResponse(f"/rooms/{room.code}/host", status_code=303)

    @app.post("/rooms/{code}/next-round")
    async def next_round(
        request: Request,
        code: str,
        topic_mode: Annotated[str, Form()],
        topic: Annotated[str, Form()] = "",
        db: Session = Depends(get_db),
    ):
        room = find_room(db, code)
        require_host(request, db, room)
        game = latest_game(db, room)
        current = (
            db.scalar(select(GameRound).where(GameRound.game_id == game.id).order_by(desc(GameRound.number))) if game else None
        )
        if room.state != "playing" or not game or not current or not current.closed_at:
            raise HTTPException(409, "请先完成当前轮次的淘汰记录。")
        try:
            add_round(db, room, game, topic_mode, topic)
        except HTTPException as exc:
            players = game_players(db, game)
            return render(request, "host.html", room=room, players=players, round=current, game=game, error=exc.detail)
        db.commit()
        await manager.broadcast(room.code, "round_started")
        return RedirectResponse(f"/rooms/{room.code}/host", status_code=303)

    @app.post("/rooms/{code}/eliminate")
    async def eliminate(
        request: Request,
        code: str,
        player_id: Annotated[int, Form()],
        db: Session = Depends(get_db),
    ):
        room = find_room(db, code)
        require_host(request, db, room)
        game = latest_game(db, room)
        round_ = (
            db.scalar(select(GameRound).where(GameRound.game_id == game.id).order_by(desc(GameRound.number))) if game else None
        )
        player = db.get(Player, player_id)
        participant = (
            db.scalar(select(GamePlayer).where(GamePlayer.game_id == game.id, GamePlayer.player_id == player_id)) if game else None
        )
        if room.state != "playing" or not game or not round_ or round_.closed_at:
            raise HTTPException(409, "当前没有可记录淘汰的进行中轮次。")
        if not player or player.room_id != room.id or not participant or not participant.is_alive:
            raise HTTPException(422, "请选择仍在场的本房间玩家。")
        player.is_alive = False
        participant.is_alive = False
        round_.closed_at = datetime.now(timezone.utc).replace(tzinfo=None)
        db.add(Elimination(room_id=room.id, game_id=game.id, round_id=round_.id, player_id=player.id))
        survivors = db.scalars(select(GamePlayer).where(GamePlayer.game_id == game.id, GamePlayer.is_alive.is_(True))).all()
        undercover_alive = sum(p.role == "undercover" for p in survivors)
        civilian_alive = sum(p.role == "civilian" for p in survivors)
        if undercover_alive == 0:
            room.state, room.winner, game.winner, game.finished_at = "finished", "civilian", "civilian", utcnow()
        elif undercover_alive >= civilian_alive:
            room.state, room.winner, game.winner, game.finished_at = "finished", "undercover", "undercover", utcnow()
        db.commit()
        await manager.broadcast(room.code, "player_eliminated")
        return RedirectResponse(f"/rooms/{room.code}/host", status_code=303)

    @app.post("/rooms/{code}/restart")
    async def restart_game(request: Request, code: str, db: Session = Depends(get_db)):
        room = find_room(db, code)
        require_host(request, db, room)
        game = latest_game(db, room)
        if room.state != "finished" or not game or not game.finished_at:
            raise HTTPException(409, "只能在本局结束后再开一局。")
        for player in db.scalars(select(Player).where(Player.room_id == room.id)):
            player.role = None
            player.is_alive = True
        room.state = "lobby"
        room.winner = None
        db.commit()
        await manager.broadcast(room.code, "game_restarted")
        return RedirectResponse(f"/rooms/{room.code}/host", status_code=303)

    def render_result(request: Request, room: Room, game: Game, db: Session):
        players = game_players(db, game)
        rounds = db.scalars(select(GameRound).where(GameRound.game_id == game.id).order_by(GameRound.number)).all()
        eliminated = {
            item.player_id for item in db.scalars(select(Elimination).where(Elimination.game_id == game.id)).all()
        }
        history = finished_games(db, room)
        return render(
            request,
            "result.html",
            room=room,
            game=game,
            players=players,
            rounds=rounds,
            eliminated=eliminated,
            history=history,
        )

    @app.get("/rooms/{code}/result")
    def result(request: Request, code: str, db: Session = Depends(get_db)):
        room = find_room(db, code)
        require_player(request, db, room)
        game = db.scalar(
            select(Game)
            .where(Game.room_id == room.id, Game.finished_at.is_not(None))
            .order_by(desc(Game.number))
        )
        if not game:
            return RedirectResponse(f"/rooms/{room.code}/play", status_code=303)
        return render_result(request, room, game, db)

    @app.get("/rooms/{code}/result/{game_number}")
    def historical_result(request: Request, code: str, game_number: int, db: Session = Depends(get_db)):
        room = find_room(db, code)
        require_player(request, db, room)
        game = db.scalar(
            select(Game).where(Game.room_id == room.id, Game.number == game_number, Game.finished_at.is_not(None))
        )
        if not game:
            raise HTTPException(404, "该局结算不存在。")
        return render_result(request, room, game, db)

    @app.get("/api/rooms/{code}")
    def room_status(request: Request, code: str, db: Session = Depends(get_db)):
        room = find_room(db, code)
        require_player(request, db, room)
        return public_state(db, room)

    @app.websocket("/ws/rooms/{code}")
    async def room_socket(websocket: WebSocket, code: str):
        token = websocket.cookies.get(settings.session_cookie_name)
        with sessions() as db:
            room = db.scalar(select(Room).where(Room.code == code.upper()))
            player = db.scalar(select(Player).where(Player.session_hash == hash_token(token))) if token else None
            if not room or not player or player.room_id != room.id:
                await websocket.close(code=4403)
                return
        await manager.connect(code.upper(), websocket)
        try:
            while True:
                await websocket.receive_text()
        except WebSocketDisconnect:
            manager.disconnect(code.upper(), websocket)

    return app


app = create_app()
