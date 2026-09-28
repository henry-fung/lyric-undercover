import json
import secrets
from dataclasses import dataclass
from typing import Protocol

from openai import OpenAI
from pydantic import BaseModel, Field

from .config import Settings


class ScenePair(BaseModel):
    civilian_prompt: str = Field(min_length=8, max_length=140)
    undercover_prompt: str = Field(min_length=8, max_length=140)
    summary: str = Field(min_length=3, max_length=80)


class SceneGenerationError(RuntimeError):
    pass


class SceneGenerator(Protocol):
    def generate(self, topic: str | None, recent_summaries: list[str]) -> ScenePair: ...


@dataclass
class OpenAICompatibleSceneGenerator:
    settings: Settings

    def generate(self, topic: str | None, recent_summaries: list[str]) -> ScenePair:
        if not self.settings.openai_api_key:
            raise SceneGenerationError("尚未配置 OPENAI_API_KEY。")

        client = OpenAI(
            api_key=self.settings.openai_api_key,
            base_url=self.settings.openai_base_url,
            timeout=self.settings.openai_timeout_seconds,
        )
        dimensions = ["人物关系", "地点", "时间", "情绪转折", "记忆细节", "天气氛围"]
        seed = secrets.token_hex(12)
        history = "；".join(recent_summaries[-6:]) or "无"
        requested_topic = topic or "完全随机的日常、情感或成长主题"
        prompt = f"""你是线下中文聚会游戏《歌词谁是卧底》的场景编辑。
生成一对可让玩家各自联想到歌曲、但绝不要求或输出歌词的中文场景题。
主题：{requested_topic}
本次创意随机种子：{seed}；请重点变化的维度：{secrets.choice(dimensions)}。
最近已用摘要（务必避开相同核心情节）：{history}。
平民题必须比卧底题多出一个自然、关键且不直接点破答案的具体情境信息；两题必须指向同一大意。
不要写任何歌词、完整歌名、歌手名、现实私人信息、侮辱、露骨性内容或违法内容。
只输出符合 JSON Schema 的对象。"""
        last_error: Exception | None = None
        for _ in range(2):
            try:
                response = client.chat.completions.create(
                    model=self.settings.openai_model,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.settings.openai_temperature,
                    response_format={"type": "json_object"},
                )
                content = response.choices[0].message.content or ""
                return ScenePair.model_validate(json.loads(content))
            except Exception as exc:  # API and schema failures are both safe to retry once.
                last_error = exc
        raise SceneGenerationError("场景生成失败，请稍后重试。") from last_error
