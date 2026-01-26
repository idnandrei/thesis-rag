from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

load_dotenv()


def _req(name: str) -> str:
    v = (os.getenv(name) or "").strip()
    if not v:
        raise RuntimeError(f"Missing required env var: {name}")
    return v


def _req_int(name: str) -> int:
    v = _req(name)
    try:
        return int(v)
    except ValueError as e:
        raise RuntimeError(f"Env var {name} must be an integer, got: {v!r}") from e


@dataclass(frozen=True)
class Settings:
    # --- Postgres (required) ---
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str

    # --- OpenAI (required) ---
    openai_api_key: str
    openai_embedding_model: str
    openai_embed_batch_size: int

    @property
    def sqlalchemy_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.pg_user}:{self.pg_password}"
            f"@{self.pg_host}:{self.pg_port}/{self.pg_db}"
        )


_settings: Settings | None = None


def get_settings() -> Settings:
    global _settings
    if _settings is None:
        _settings = Settings(
            pg_host=_req("PGHOST"),
            pg_port=_req_int("PGPORT"),
            pg_db=_req("PGDATABASE"),
            pg_user=_req("PGUSER"),
            pg_password=_req("PGPASSWORD"),
            openai_api_key=_req("OPENAI_API_KEY"),
            openai_embedding_model=_req("OPENAI_EMBEDDING_MODEL"),
            openai_embed_batch_size=_req_int("OPENAI_EMBED_BATCH_SIZE"),
        )
    return _settings
