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
        """
        Finds data/raw/<video_id>/video.<ext> (mp4/mkv/mov/...)
        """
        matches = sorted(self.raw_dir.glob("video.*"))
        if not matches:
            raise FileNotFoundError(
                f"No raw video found in {self.raw_dir}. Expected something like video.mp4 / video.mkv"
            )

        # Prefer common formats if multiple exist
        preferred = [".mp4", ".mkv", ".mov", ".webm", ".avi"]
        for ext in preferred:
            for p in matches:
                if p.suffix.lower() == ext:
                    return p

        return matches[0]

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
