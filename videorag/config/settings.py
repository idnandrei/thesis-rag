from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

load_dotenv()


def _load_config_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}

    with open(path, "r", encoding="utf-8") as file:
        data = yaml.safe_load(file) or {}

    if not isinstance(data, dict):
        raise RuntimeError(f"Config file must contain a top-level mapping: {path}")

    return data


def _section(config: dict[str, Any], name: str) -> dict[str, Any]:
    value = config.get(name, {})
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise RuntimeError(f"Config section '{name}' must be a mapping.")
    return value


@dataclass(frozen=True)
class PathSettings:
    data_dir: Path = Path("data")
    derived_dir: Path = Path("data/derived")
    registry_path: Path = Path("data/registry.json")
    dataset_root: Path = Path("dataset/longervideos")

    def to_log_dict(self) -> dict[str, str]:
        return {
            "data_dir": str(self.data_dir),
            "derived_dir": str(self.derived_dir),
            "registry_path": str(self.registry_path),
            "dataset_root": str(self.dataset_root),
        }


@dataclass(frozen=True)
class PromptingLLMSettings:
    model_name: str = "gpt-5.4"
    reasoning_effort: str = "medium"
    verbosity: str = "low"

    def to_log_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentSettings:
    name: str = "baseline"

    def to_log_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class FrameSamplingSettings:
    enabled: bool = True
    interval_seconds: float = 1.0
    image_ext: str = "jpg"
    jpeg_quality: int = 90

    def to_log_dict(self) -> dict[str, bool | float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class KeyframeFilteringSettings:
    enabled: bool = False
    method: str = "ssim"

    ssim_threshold: float = 0.80
    ssim_dynamic_threshold: bool = True
    ssim_dynamic_c: float = 0.01
    ssim_dynamic_k: float = 1000.0

    histogram_threshold: float = 0.005
    histogram_dynamic_threshold: bool = False
    histogram_grid_rows: int = 4
    histogram_grid_cols: int = 4
    histogram_bins: int = 64

    def to_log_dict(self) -> dict[str, bool | float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class VisualExtractionSettings:
    model_path: str = "models/visual_object_detector/best.pt"
    conf: float = 0.5
    iou: float = 0.45
    pad_frac: float = 0.03
    min_area_frac: float = 0.01

    def to_log_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


@dataclass(frozen=True)
class ASRSettings:
    language: str = "en"
    model_name: str = "mlx-community/whisper-tiny"

    def to_log_dict(self) -> dict[str, str]:
        return asdict(self)


@dataclass(frozen=True)
class OCRSettings:
    min_confidence: float = 0.0
    use_gpu: bool = True

    def to_log_dict(self) -> dict[str, bool | float]:
        return asdict(self)


@dataclass(frozen=True)
class ImageCaptionSettings:
    model_name: str = "Qwen/Qwen3-VL-4B-Instruct"
    max_new_tokens: int = 128
    do_sample: bool = False

    def to_log_dict(self) -> dict[str, str | int | bool]:
        return asdict(self)


@dataclass(frozen=True)
class PersonFilterSettings:
    model_name: str = "yolo26n.pt"
    person_conf_threshold: float = 0.30
    person_box_ratio_threshold: float = 0.40

    def to_log_dict(self) -> dict[str, str | float]:
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
    keyframe_filtering: KeyframeFilteringSettings = field(
        default_factory=KeyframeFilteringSettings
    )
    visual_extraction: VisualExtractionSettings = field(
        default_factory=VisualExtractionSettings
    )
    asr: ASRSettings = field(default_factory=ASRSettings)
    ocr: OCRSettings = field(default_factory=OCRSettings)
    image_caption: ImageCaptionSettings = field(default_factory=ImageCaptionSettings)
    prompting_llm: PromptingLLMSettings = field(default_factory=PromptingLLMSettings)
    chunking: ChunkingSettings = field(default_factory=ChunkingSettings)
    person_filter: PersonFilterSettings = field(default_factory=PersonFilterSettings)
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

        config_path = Path(os.getenv("APP_CONFIG_PATH", "videorag/config/config.yaml"))
        config = _load_config_file(config_path)

        person_filter_cfg = _section(config, "person_filter")
        paths_cfg = _section(config, "paths")
        experiment_cfg = _section(config, "experiment")
        frame_sampling_cfg = _section(config, "frame_sampling")
        keyframe_filtering_cfg = _section(config, "keyframe_filtering")
        visual_extraction_cfg = _section(config, "visual_extraction")
        asr_cfg = _section(config, "asr")
        ocr_cfg = _section(config, "ocr")
        image_caption_cfg = _section(config, "image_caption")
        prompting_llm_cfg = _section(config, "prompting_llm")
        chunking_cfg = _section(config, "chunking")
        retrieval_cfg = _section(config, "retrieval")

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
            paths=PathSettings(
                data_dir=Path(paths_cfg.get("data_dir", "data")),
                derived_dir=Path(paths_cfg.get("derived_dir", "data/derived")),
                registry_path=Path(
                    paths_cfg.get("registry_path", "data/registry.json")
                ),
                dataset_root=Path(
                    paths_cfg.get("dataset_root", "dataset/longervideos")
                ),
            ),
            experiment=ExperimentSettings(
                name=experiment_cfg.get("name", "baseline"),
            ),
            person_filter=PersonFilterSettings(
                model_name=person_filter_cfg.get("model_name", "yolo26n.pt"),
                person_conf_threshold=person_filter_cfg.get(
                    "person_conf_threshold", 0.30
                ),
                person_box_ratio_threshold=person_filter_cfg.get(
                    "person_box_ratio_threshold", 0.35
                ),
            ),
            frame_sampling=FrameSamplingSettings(
                enabled=frame_sampling_cfg.get("enabled", True),
                interval_seconds=frame_sampling_cfg.get("interval_seconds", 1.0),
                image_ext=frame_sampling_cfg.get("image_ext", "jpg"),
                jpeg_quality=frame_sampling_cfg.get("jpeg_quality", 90),
            ),
            keyframe_filtering=KeyframeFilteringSettings(
                enabled=keyframe_filtering_cfg.get("enabled", False),
                method=keyframe_filtering_cfg.get("method", "histogram"),
                ssim_threshold=keyframe_filtering_cfg.get("ssim_threshold", 0.60),
                ssim_dynamic_threshold=keyframe_filtering_cfg.get(
                    "ssim_dynamic_threshold", False
                ),
                ssim_dynamic_c=keyframe_filtering_cfg.get("ssim_dynamic_c", 0.01),
                ssim_dynamic_k=keyframe_filtering_cfg.get("ssim_dynamic_k", 1000.0),
                histogram_threshold=keyframe_filtering_cfg.get(
                    "histogram_threshold", 0.005
                ),
                histogram_dynamic_threshold=keyframe_filtering_cfg.get(
                    "histogram_dynamic_threshold", True
                ),
                histogram_grid_rows=keyframe_filtering_cfg.get(
                    "histogram_grid_rows", 4
                ),
                histogram_grid_cols=keyframe_filtering_cfg.get(
                    "histogram_grid_cols", 4
                ),
                histogram_bins=keyframe_filtering_cfg.get("histogram_bins", 64),
            ),
            visual_extraction=VisualExtractionSettings(
                model_path=visual_extraction_cfg.get(
                    "model_path", "models/visual_object_detector/best.pt"
                ),
                conf=visual_extraction_cfg.get("conf", 0.5),
                iou=visual_extraction_cfg.get("iou", 0.45),
                pad_frac=visual_extraction_cfg.get("pad_frac", 0.03),
                min_area_frac=visual_extraction_cfg.get("min_area_frac", 0.01),
            ),
            asr=ASRSettings(
                language=asr_cfg.get("language", "en"),
                model_name=asr_cfg.get("model_name", "tiny"),
            ),
            ocr=OCRSettings(
                min_confidence=ocr_cfg.get("min_confidence", 0.0),
                use_gpu=ocr_cfg.get("use_gpu", True),
            ),
            image_caption=ImageCaptionSettings(
                model_name=image_caption_cfg.get(
                    "model_name", "Qwen/Qwen3-VL-4B-Instruct"
                ),
                max_new_tokens=image_caption_cfg.get("max_new_tokens", 128),
                do_sample=image_caption_cfg.get("do_sample", False),
            ),
            prompting_llm=PromptingLLMSettings(
                model_name=prompting_llm_cfg.get("model_name", "gpt-5.4"),
                reasoning_effort=prompting_llm_cfg.get("reasoning_effort", "medium"),
                verbosity=prompting_llm_cfg.get("verbosity", "low"),
            ),
            chunking=ChunkingSettings(
                chunk_tokens=chunking_cfg.get("chunk_tokens", 512),
                overlap_tokens=chunking_cfg.get("overlap_tokens", 100),
                max_tokens=chunking_cfg.get("max_tokens", 640),
                tokenizer_name=chunking_cfg.get("tokenizer_name", "cl100k_base"),
            ),
            retrieval=RetrievalSettings(
                top_k=retrieval_cfg.get("top_k", 5),
            ),
        )

    return _settings
