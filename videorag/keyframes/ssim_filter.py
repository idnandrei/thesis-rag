from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from skimage.metrics import structural_similarity

from videorag.config.settings import get_settings
from videorag.keyframes.common import (
    copy_frame_to_kept_dir,
    load_manifest,
    prepare_gray,
    write_manifest,
)


def _ssim_score(prev_gray, curr_gray) -> float:
    return float(structural_similarity(prev_gray, curr_gray, data_range=255))


def _dynamic_threshold_from_study(
    ssim_scores: list[float], c: float, k: float
) -> float:
    if not ssim_scores:
        raise ValueError("Cannot compute dynamic SSIM threshold without SSIM scores.")

    mean_ssim = float(np.mean(ssim_scores))
    if mean_ssim <= 0:
        raise ValueError("Mean SSIM must be > 0 for the study formula.")
    if k == 0:
        raise ValueError("K must not be 0 for the study formula.")

    # Study formula:
    # mean_ssim + C + (1 / mean_ssim) * (1 / K)
    return mean_ssim + c + (1.0 / mean_ssim) * (1.0 / k)


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


def filter_keyframes_ssim(
    *,
    video_id: str,
    frames_manifest_path: Path,
    output_manifest_path: Path,
    threshold: float | None = None,
    use_dynamic_threshold: bool | None = None,
) -> Path:
    settings = get_settings()
    cfg = settings.keyframe_filtering

    threshold = threshold if threshold is not None else cfg.ssim_threshold
    use_dynamic_threshold = (
        use_dynamic_threshold
        if use_dynamic_threshold is not None
        else cfg.ssim_dynamic_threshold
    )

    c = cfg.ssim_dynamic_c
    k = cfg.ssim_dynamic_k

    manifest = load_manifest(frames_manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("frames manifest has no frames")

    prepared: list[tuple[dict[str, Any], Any]] = []
    for frame in frames:
        image_path = Path(str(frame["image_path"]))
        gray = prepare_gray(image_path)
        prepared.append((frame, gray))

    consecutive_scores: list[float] = []
    for i in range(1, len(prepared)):
        prev_gray = prepared[i - 1][1]
        curr_gray = prepared[i][1]
        consecutive_scores.append(_ssim_score(prev_gray, curr_gray))

    effective_threshold = threshold
    threshold_mode = "static"

    if use_dynamic_threshold:
        effective_threshold = _dynamic_threshold_from_study(
            consecutive_scores,
            c=c,
            k=k,
        )
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

    # Study-consistent logic: keep frame when SSIM <= threshold.
    # Otherwise, extend the current kept frame's represented span.
    for i in range(1, len(prepared)):
        frame, curr_gray = prepared[i]
        prev_gray = prepared[i - 1][1]

        score = _ssim_score(prev_gray, curr_gray)

        if score <= effective_threshold:
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
            "method": "ssim",
            "comparison": "current_sampled_frame_vs_previous_sampled_frame",
            "threshold_mode": threshold_mode,
            "static_threshold": threshold,
            "effective_threshold": round(float(effective_threshold), 6),
            "study_dynamic_formula": "mean_ssim + C + (1/mean_ssim)*(1/K)",
            "study_dynamic_constants": {
                "C": c,
                "K": k,
            },
        },
        "counts": {
            "input_frames": len(frames),
            "kept_frames": len(kept_frames),
        },
        "frames": kept_frames,
    }

    return write_manifest(output_manifest_path, payload)
