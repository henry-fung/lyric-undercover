from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import Settings
from app.models import Elimination, Game, GamePlayer, GameRound, Player, Room
from app.scenes import OpenAICompatibleSceneGenerator, SceneGenerationError, ScenePair
from main import create_app


class FakeGenerator:
    def __init__(self):
        self.calls = []

    def generate(self, topic, recent_summaries):
        self.calls.append((topic, recent_summaries))
        return ScenePair(
            civilian_prompt="在毕业典礼后偶遇了多年未见的同桌，最想唱什么歌？",
            undercover_prompt="偶遇多年未见的人，最想唱什么歌？",
            summary="久别重逢",
        )


def make_app(tmp_path: Path):
    settings = Settings(
        invite_code="party-code",
        database_url=f"sqlite:///{tmp_path / 'game.db'}",
        openai_api_key="test",
    )
    fake = FakeGenerator()
    return create_app(settings, fake), fake


def test_invite_name_recovery_and_complete_round(tmp_path):
    app, fake = make_app(tmp_path)
    with TestClient(app) as host:
        bad = host.post("/rooms", data={"name": "房主", "invite_code": "wrong"})
        assert "邀请码不正确" in bad.text

        created = host.post("/rooms", data={"name": "房主", "invite_code": "party-code"})
        assert created.status_code == 200
        assert "保存你的恢复码" in created.text
        room_code = created.url.path.split("/")[2]

        guest_a = TestClient(app)
        joined = guest_a.post(f"/rooms/{room_code}/join", data={"name": "阿兰", "invite_code": "party-code"})
        assert "保存你的恢复码" in joined.text
        guest_b = TestClient(app)
        guest_b.post(f"/rooms/{room_code}/join", data={"name": "小北", "invite_code": "party-code"})
        duplicate = guest_b.post(f"/rooms/{room_code}/join", data={"name": "阿兰", "invite_code": "party-code"})
        assert "这个名字已被使用" in duplicate.text

        started = host.post(
            f"/rooms/{room_code}/start",
            data={"undercover_count": "1", "topic_mode": "random", "topic": ""},
        )
        assert started.status_code == 200
        assert fake.calls[0][0] is None
        public = host.get(f"/api/rooms/{room_code}").json()
        assert public["state"] == "playing"
        assert len(public["players"]) == 3
        assert "role" not in public["players"][0]

        with app.state.sessions() as db:
            room = db.scalar(select(Room).where(Room.code == room_code))
            undercover = db.scalar(select(Player).where(Player.room_id == room.id, Player.role == "undercover"))
            assert undercover is not None
            undercover_id = undercover.id
        eliminated = host.post(f"/rooms/{room_code}/eliminate", data={"player_id": str(undercover_id)})
        assert eliminated.status_code == 200
        state = host.get(f"/api/rooms/{room_code}").json()
        assert state["state"] == "finished"
        assert state["winner"] == "civilian"
        assert "平民获胜" in host.get(f"/rooms/{room_code}/result").text


def test_recovery_code_changes_browser_session(tmp_path):
    app, _ = make_app(tmp_path)
    with TestClient(app) as client:
        page = client.post("/rooms", data={"name": "可可", "invite_code": "party-code"})
        room_code = page.url.path.split("/")[2]
        import re

        recovery = re.search(r"恢复码：([A-Z0-9]+)", page.text).group(1)
        new_browser = TestClient(app)
        restored = new_browser.post(
            f"/rooms/{room_code}/recover", data={"name": "可可", "recovery_code": recovery}
        )
        assert restored.status_code == 200
        assert "你好，可可" in restored.text


def test_websocket_broadcasts_public_room_change(tmp_path):
    app, _ = make_app(tmp_path)
    with TestClient(app) as host:
        page = host.post("/rooms", data={"name": "房主", "invite_code": "party-code"})
        room_code = page.url.path.split("/")[2]
        with host.websocket_connect(f"/ws/rooms/{room_code}") as socket:
            guest = TestClient(app)
            guest.post(f"/rooms/{room_code}/join", data={"name": "听众", "invite_code": "party-code"})
            assert socket.receive_json() == {"event": "player_joined"}


def test_host_can_restart_room_and_keep_game_history(tmp_path):
    app, fake = make_app(tmp_path)
    with TestClient(app) as host:
        created = host.post("/rooms", data={"name": "host", "invite_code": "party-code"})
        room_code = created.url.path.split("/")[2]
        guest_a = TestClient(app)
        guest_b = TestClient(app)
        guest_a.post(f"/rooms/{room_code}/join", data={"name": "alpha", "invite_code": "party-code"})
        guest_b.post(f"/rooms/{room_code}/join", data={"name": "bravo", "invite_code": "party-code"})

        host.post(
            f"/rooms/{room_code}/start",
            data={"undercover_count": "1", "topic_mode": "random", "topic": ""},
        )
        with app.state.sessions() as db:
            room = db.scalar(select(Room).where(Room.code == room_code))
            first_game = db.scalar(select(Game).where(Game.room_id == room.id, Game.number == 1))
            first_undercover = db.scalar(
                select(GamePlayer).where(GamePlayer.game_id == first_game.id, GamePlayer.role == "undercover")
            )
        host.post(f"/rooms/{room_code}/eliminate", data={"player_id": str(first_undercover.player_id)})

        forbidden = guest_a.post(f"/rooms/{room_code}/restart", follow_redirects=False)
        assert forbidden.status_code == 403
        with host.websocket_connect(f"/ws/rooms/{room_code}") as socket:
            restarted = host.post(f"/rooms/{room_code}/restart")
            assert restarted.status_code == 200
            assert socket.receive_json() == {"event": "game_restarted"}

        lobby = host.get(f"/api/rooms/{room_code}").json()
        assert lobby["state"] == "lobby"
        assert lobby["winner"] is None
        assert {player["name"] for player in lobby["players"]} == {"host", "alpha", "bravo"}
        guest_c = TestClient(app)
        guest_c.post(f"/rooms/{room_code}/join", data={"name": "charlie", "invite_code": "party-code"})

        host.post(
            f"/rooms/{room_code}/start",
            data={"undercover_count": "1", "topic_mode": "random", "topic": ""},
        )
        assert fake.calls[1][1] == []
        with app.state.sessions() as db:
            room = db.scalar(select(Room).where(Room.code == room_code))
            second_game = db.scalar(select(Game).where(Game.room_id == room.id, Game.number == 2))
            assert second_game is not None
            assert len(db.scalars(select(GamePlayer).where(GamePlayer.game_id == second_game.id)).all()) == 4
            second_round = db.scalar(select(GameRound).where(GameRound.game_id == second_game.id))
            assert second_round.number == 1
            second_undercover = db.scalar(
                select(GamePlayer).where(GamePlayer.game_id == second_game.id, GamePlayer.role == "undercover")
            )
        repeated_elimination = host.post(f"/rooms/{room_code}/eliminate", data={"player_id": str(first_undercover.player_id)})
        assert repeated_elimination.status_code == 200
        if host.get(f"/api/rooms/{room_code}").json()["state"] == "playing":
            host.post(
                f"/rooms/{room_code}/next-round",
                data={"topic_mode": "random", "topic": ""},
            )
            host.post(f"/rooms/{room_code}/eliminate", data={"player_id": str(second_undercover.player_id)})

        first_result = host.get(f"/rooms/{room_code}/result/1")
        assert first_result.status_code == 200
        assert "charlie" not in first_result.text
        latest_result = host.get(f"/rooms/{room_code}/result")
        assert f"/rooms/{room_code}/result/1" in latest_result.text


def test_sqlite_upgrade_migrates_a_legacy_finished_game(tmp_path):
    import sqlite3

    database_path = tmp_path / "legacy.db"
    connection = sqlite3.connect(database_path)
    connection.executescript(
        """
        CREATE TABLE rooms (id INTEGER PRIMARY KEY, code VARCHAR(8) UNIQUE, state VARCHAR(16),
            host_player_id INTEGER, winner VARCHAR(16), created_at DATETIME);
        CREATE TABLE players (id INTEGER PRIMARY KEY, room_id INTEGER, name VARCHAR(40),
            session_hash VARCHAR(64) UNIQUE, recovery_hash VARCHAR(64) UNIQUE, role VARCHAR(16),
            is_alive BOOLEAN, joined_at DATETIME);
        CREATE TABLE rounds (id INTEGER PRIMARY KEY, room_id INTEGER, number INTEGER, theme VARCHAR(80),
            civilian_prompt TEXT, undercover_prompt TEXT, created_at DATETIME, closed_at DATETIME);
        CREATE TABLE eliminations (id INTEGER PRIMARY KEY, room_id INTEGER, round_id INTEGER,
            player_id INTEGER UNIQUE, created_at DATETIME);
        INSERT INTO rooms VALUES (1, 'LEGACY', 'finished', 1, 'civilian', '2026-01-01 00:00:00');
        INSERT INTO players VALUES (1, 1, 'host', 'session-1', 'recovery-1', 'civilian', 1, '2026-01-01 00:00:00');
        INSERT INTO players VALUES (2, 1, 'undercover', 'session-2', 'recovery-2', 'undercover', 0, '2026-01-01 00:00:00');
        INSERT INTO rounds VALUES (1, 1, 1, 'legacy theme', 'civilian prompt', 'undercover prompt',
            '2026-01-01 00:00:00', '2026-01-01 00:01:00');
        INSERT INTO eliminations VALUES (1, 1, 1, 2, '2026-01-01 00:01:00');
        """
    )
    connection.commit()
    connection.close()

    settings = Settings(invite_code="party-code", database_url=f"sqlite:///{database_path}", openai_api_key="test")
    app = create_app(settings, FakeGenerator())
    with TestClient(app):
        with app.state.sessions() as db:
            game = db.scalar(select(Game).where(Game.room_id == 1, Game.number == 1))
            assert game is not None and game.winner == "civilian" and game.finished_at is not None
            assert db.scalar(select(GameRound).where(GameRound.id == 1)).game_id == game.id
            assert len(db.scalars(select(GamePlayer).where(GamePlayer.game_id == game.id)).all()) == 2
            elimination = db.scalar(select(Elimination).where(Elimination.id == 1))
            assert elimination.game_id == game.id


def test_scene_generator_logs_original_failures(monkeypatch, caplog):
    class FailingCompletions:
        def create(self, **kwargs):
            raise TimeoutError("upstream timed out")

    class FailingClient:
        class Chat:
            completions = FailingCompletions()

        chat = Chat()

    monkeypatch.setattr("app.scenes.OpenAI", lambda **kwargs: FailingClient())
    generator = OpenAICompatibleSceneGenerator(Settings(openai_api_key="test", openai_model="test-model"))

    with caplog.at_level("ERROR", logger="app.scenes"):
        try:
            generator.generate(None, [])
        except SceneGenerationError:
            pass
        else:
            raise AssertionError("Expected scene generation to fail")

    assert caplog.text.count("Scene generation failed") == 2
    assert "upstream timed out" in caplog.text
