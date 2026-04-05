from __future__ import annotations

from pathlib import Path

from videorag.config.settings import get_settings
from videorag.keyframes.histogram_filter import filter_keyframes_histogram
from videorag.keyframes.ssim_filter import filter_keyframes_ssim


def filter_keyframes(
    *,
    video_id: str,
    frames_manifest_path: Path,
    output_manifest_path: Path,
    method: str | None = None,
) -> Path:
    settings = get_settings()
    method = (method or settings.keyframe_filtering.method).lower().strip()

    if method == "ssim":
        return filter_keyframes_ssim(
            video_id=video_id,
            frames_manifest_path=frames_manifest_path,
            output_manifest_path=output_manifest_path,
        )

    if method == "histogram":
        return filter_keyframes_histogram(
            video_id=video_id,
            frames_manifest_path=frames_manifest_path,
            output_manifest_path=output_manifest_path,
        )

    raise ValueError(f"Unsupported keyframe filtering method: {method}")
