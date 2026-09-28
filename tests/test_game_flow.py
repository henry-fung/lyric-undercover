from pathlib import Path

from fastapi.testclient import TestClient
from sqlalchemy import select

from app.config import Settings
from app.models import Player, Room
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
