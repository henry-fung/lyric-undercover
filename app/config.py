from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    invite_code: str = "change-me"
    openai_api_key: str = ""
    openai_base_url: str | None = None
    openai_model: str = "gpt-4.1-mini"
    database_url: str = "sqlite:///./data/lyric_undercover.db"
    openai_timeout_seconds: float = 25
    openai_temperature: float = 1.1
    session_cookie_name: str = "lyric_undercover_session"
    session_secure: bool = False

    model_config = SettingsConfigDict(
        env_file=Path(".env"), env_file_encoding="utf-8", extra="ignore"
    )
