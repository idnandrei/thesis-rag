from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class VideoPaths:
    video_id: str
    raw_root: Path
    derived_root: Path

    @property
    def raw_dir(self) -> Path:
        return self.raw_root / self.video_id

    @property
    def derived_dir(self) -> Path:
        return self.derived_root / self.video_id

    # ---- Raw inputs ----
    @property
    def raw_video_path(self) -> Path:
        # your convention: data/raw/<video_id>/video.mp4
        return self.raw_dir / "video.mp4"

    # ---- Derived artifacts ----
    @property
    def transcript_segments_path(self) -> Path:
        return self.derived_dir / "transcript_segments.json"

    @property
    def transcript_raw_path(self) -> Path:
        return self.derived_dir / "transcript_raw.txt"

    @property
    def events_path(self) -> Path:
        return self.derived_dir / "events.json"

    @property
    def chunks_path(self) -> Path:
        return self.derived_dir / "chunks.json"


def video_paths(
    video_id: str,
    *,
    raw_root: Path = Path("data/raw"),
    derived_root: Path = Path("data/derived"),
) -> VideoPaths:
    return VideoPaths(video_id=video_id, raw_root=raw_root, derived_root=derived_root)
