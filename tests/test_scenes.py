from app.config import Settings
from app.scenes import OpenAICompatibleSceneGenerator


def test_openai_compatible_scene_generation_retries_and_uses_randomized_prompt(monkeypatch):
    calls = []
    client_options = []

    class Responses:
        def create(self, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise RuntimeError("temporary API error")
            return type(
                "Response",
                (),
                {
                    "choices": [
                        type(
                            "Choice",
                            (),
                            {
                                "message": type(
                                    "Message",
                                    (),
                                    {
                                        "content": '{"civilian_prompt":"在雨后的车站等到多年未见的恋人，最想唱什么歌？","undercover_prompt":"等到多年未见的人，最想唱什么歌？","summary":"雨后重逢"}'
                                    },
                                )()
                            },
                        )()
                    ]
                },
            )()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            client_options.append(kwargs)
            self.chat = type("Chat", (), {"completions": Responses()})()

    monkeypatch.setattr("app.scenes.OpenAI", FakeOpenAI)
    generator = OpenAICompatibleSceneGenerator(
        Settings(
            openai_api_key="key",
            openai_base_url="https://llm.example/v1",
            openai_model="test-model",
        )
    )
    pair = generator.generate("爱情", ["校园离别"])

    assert pair.summary == "雨后重逢"
    assert len(calls) == 2
    assert client_options[0]["base_url"] == "https://llm.example/v1"
    assert calls[0]["model"] == "test-model"
    assert calls[0]["response_format"] == {"type": "json_object"}
    prompt = calls[0]["messages"][0]["content"]
    assert "校园离别" in prompt
    assert "随机种子" in prompt
