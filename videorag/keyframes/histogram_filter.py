from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from videorag.config.settings import get_settings
from videorag.keyframes.common import (
    copy_frame_to_kept_dir,
    load_manifest,
    write_manifest,
)


def _load_bgr(image_path: Path) -> np.ndarray:
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"Failed to load image: {image_path}")
    return image


def _region_histogram_difference(
    prev_img: np.ndarray,
    curr_img: np.ndarray,
    *,
    grid_rows: int,
    grid_cols: int,
    bins: int,
) -> float:
    height, width = prev_img.shape[:2]
    region_h = height // grid_rows
    region_w = width // grid_cols

    if region_h == 0 or region_w == 0:
        raise ValueError("Image too small for requested histogram grid size.")

    region_scores: list[float] = []

    for r in range(grid_rows):
        for c in range(grid_cols):
            y1 = r * region_h
            y2 = height if r == grid_rows - 1 else (r + 1) * region_h
            x1 = c * region_w
            x2 = width if c == grid_cols - 1 else (c + 1) * region_w

            prev_region = prev_img[y1:y2, x1:x2]
            curr_region = curr_img[y1:y2, x1:x2]

            channel_scores: list[float] = []
            for ch in range(prev_region.shape[2]):
                prev_hist = cv2.calcHist([prev_region], [ch], None, [bins], [0, 256])
                curr_hist = cv2.calcHist([curr_region], [ch], None, [bins], [0, 256])

                diff = float(np.sum(np.abs(prev_hist - curr_hist)))

                # Normalize using:
                # bins * region_height * region_width
                norm = bins * (y2 - y1) * (x2 - x1)
                if norm <= 0:
                    raise ValueError("Invalid histogram normalization factor.")
                diff /= norm

                channel_scores.append(diff)

            region_scores.append(float(np.mean(channel_scores)))

    return float(np.mean(region_scores))


def _dynamic_threshold_from_study(metric_values: list[float]) -> float:
    if not metric_values:
        raise ValueError("Cannot compute dynamic histogram threshold without values.")
    return float(np.median(metric_values))


def _frame_sample(frame: dict[str, Any]) -> int | None:
    value = frame.get("sample_index")
    return int(value) if value is not None else None


def _frame_time(frame: dict[str, Any]) -> float | None:
    value = frame.get("actual_time_sec")
    return float(value) if value is not None else None


def _init_span(kept_frame: dict[str, Any], source_frame: dict[str, Any]) -> None:
    kept_frame["sample_start"] = _frame_sample(source_frame)
    kept_frame["sample_end"] = _frame_sample(source_frame)
    kept_frame["time_start"] = _frame_time(source_frame)
    kept_frame["time_end"] = _frame_time(source_frame)


def _extend_span(kept_frame: dict[str, Any], source_frame: dict[str, Any]) -> None:
    kept_frame["sample_end"] = _frame_sample(source_frame)
    kept_frame["time_end"] = _frame_time(source_frame)


def filter_keyframes_histogram(
    *,
    video_id: str,
    frames_manifest_path: Path,
    output_manifest_path: Path,
    threshold: float | None = None,
    use_dynamic_threshold: bool | None = None,
    grid_rows: int | None = None,
    grid_cols: int | None = None,
    bins: int | None = None,
) -> Path:
    settings = get_settings()
    cfg = settings.keyframe_filtering

    threshold = threshold if threshold is not None else cfg.histogram_threshold
    use_dynamic_threshold = (
        use_dynamic_threshold
        if use_dynamic_threshold is not None
        else cfg.histogram_dynamic_threshold
    )
    grid_rows = grid_rows if grid_rows is not None else cfg.histogram_grid_rows
    grid_cols = grid_cols if grid_cols is not None else cfg.histogram_grid_cols
    bins = bins if bins is not None else cfg.histogram_bins

    manifest = load_manifest(frames_manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames manifest has no frames")

    prepared: list[tuple[dict[str, Any], np.ndarray]] = []
    for frame in frames:
        image_path = Path(str(frame["image_path"]))
        image = _load_bgr(image_path)
        prepared.append((frame, image))

    consecutive_scores: list[float] = []
    for i in range(1, len(prepared)):
        prev_img = prepared[i - 1][1]
        curr_img = prepared[i][1]
        score = _region_histogram_difference(
            prev_img,
            curr_img,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            bins=bins,
        )
        consecutive_scores.append(score)

    effective_threshold = threshold
    threshold_mode = "static"

    if use_dynamic_threshold:
        effective_threshold = _dynamic_threshold_from_study(consecutive_scores)
        threshold_mode = "dynamic"

    output_dir = output_manifest_path.parent
    kept_frames_dir = output_dir / "frames"

    kept_frames: list[dict[str, Any]] = []

    # Always keep the first sampled frame
    first_source_frame = prepared[0][0]
    first_kept_frame = copy_frame_to_kept_dir(first_source_frame, kept_frames_dir)
    first_kept_frame["keyframe_score_to_previous"] = None
    _init_span(first_kept_frame, first_source_frame)
    kept_frames.append(first_kept_frame)

    # Keep frame when histogram difference >= threshold.
    # Otherwise, extend the current kept frame's represented span.
    for i in range(1, len(prepared)):
        frame, curr_img = prepared[i]
        prev_img = prepared[i - 1][1]

        score = _region_histogram_difference(
            prev_img,
            curr_img,
            grid_rows=grid_rows,
            grid_cols=grid_cols,
            bins=bins,
        )

        if score >= effective_threshold:
            kept_frame = copy_frame_to_kept_dir(frame, kept_frames_dir)
            kept_frame["keyframe_score_to_previous"] = round(score, 6)
            _init_span(kept_frame, frame)
            kept_frames.append(kept_frame)
        else:
            _extend_span(kept_frames[-1], frame)

    payload = {
        "video_id": video_id,
        "source_frames_manifest_path": str(frames_manifest_path),
        "filtering": {
            "method": "histogram",
            "comparison": "current_sampled_frame_vs_previous_sampled_frame",
            "threshold_mode": threshold_mode,
            "static_threshold": threshold,
            "effective_threshold": round(float(effective_threshold), 6),
            "study_dynamic_formula": "median(histogram_difference_values)",
            "histogram_config": {
                "grid_rows": grid_rows,
                "grid_cols": grid_cols,
                "bins": bins,
            },
        },
        "counts": {
            "input_frames": len(frames),
            "kept_frames": len(kept_frames),
        },
        "frames": kept_frames,
    }

    return write_manifest(output_manifest_path, payload)
