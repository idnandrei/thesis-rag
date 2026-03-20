from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class PathSettings:
    data_dir: Path = Path("data")
    raw_dir: Path = Path("data/raw")
    derived_dir: Path = Path("data/derived")
    registry_path: Path = Path("data/registry.json")

    def to_log_dict(self) -> dict[str, str]:
        return {
            "data_dir": str(self.data_dir),
            "raw_dir": str(self.raw_dir),
            "derived_dir": str(self.derived_dir),
            "registry_path": str(self.registry_path),
        }


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "baseline"

    def to_log_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FrameSamplingSettings:
    enabled: bool = True
    interval_seconds: float = 2.0
    image_ext: str = "jpg"
    jpeg_quality: int = 90

    def to_log_dict(self) -> dict[str, bool | float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ASRSettings:
    language: str = "en"
    model_name: str = "mlx-community/whisper-tiny"

    def to_log_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ChunkingSettings:
    chunk_tokens: int = 512
    overlap_tokens: int = 100
    max_tokens: int = 640
    tokenizer_name: str = "cl100k_base"

    def to_log_dict(self) -> dict[str, int | str]:
        return asdict(self)


@dataclass(frozen=True)
class RetrievalSettings:
    top_k: int = 5

    def to_log_dict(self) -> dict[str, int]:
        return asdict(self)


@dataclass(frozen=True)
class Settings:
    pg_host: str
    pg_port: int
    pg_db: str
    pg_user: str
    pg_password: str

    openai_api_key: str
    openai_embedding_model: str
    openai_embed_batch_size: int

    paths: PathSettings = field(default_factory=PathSettings)
    experiment: ExperimentSettings = field(default_factory=ExperimentSettings)
    frame_sampling: FrameSamplingSettings = field(default_factory=FrameSamplingSettings)
    asr: ASRSettings = field(default_factory=ASRSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    retrieval: RetrievalSettings = field(default_factory=RetrievalSettings)

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
        pg_host = os.getenv("PGHOST")
        pg_port = os.getenv("PGPORT")
        pg_db = os.getenv("PGDATABASE")
        pg_user = os.getenv("PGUSER")
        pg_password = os.getenv("PGPASSWORD")
        openai_api_key = os.getenv("OPENAI_API_KEY")

        if not pg_host:
            raise RuntimeError("Missing required env var: PGHOST")
        if not pg_port:
            raise RuntimeError("Missing required env var: PGPORT")
        if not pg_db:
            raise RuntimeError("Missing required env var: PGDATABASE")
        if not pg_user:
            raise RuntimeError("Missing required env var: PGUSER")
        if not pg_password:
            raise RuntimeError("Missing required env var: PGPASSWORD")
        if not openai_api_key:
            raise RuntimeError("Missing required env var: OPENAI_API_KEY")

        _settings = Settings(
            pg_host=pg_host,
            pg_port=int(pg_port),
            pg_db=pg_db,
            pg_user=pg_user,
            pg_password=pg_password,
            openai_api_key=openai_api_key,
            openai_embedding_model=os.getenv(
                "OPENAI_EMBEDDING_MODEL",
                "text-embedding-3-small",
            ),
            openai_embed_batch_size=int(os.getenv("OPENAI_EMBED_BATCH_SIZE", "100")),
        )

    return _settings
