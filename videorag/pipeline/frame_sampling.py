from pathlib import Path
import json

import cv2


def sample_frames_every_n_seconds(
    *,
    video_path: Path,
    output_dir: Path,
    interval_seconds: float = 1.0,
    image_ext: str = "jpg",
    jpeg_quality: int = 90,
) -> Path:
    """
    Sample frames from a video at fixed time intervals and save:
      - frames under: <output_dir>/frames/
      - manifest under: <output_dir>/frames_manifest.json

    Returns:
        Path to the manifest file.
    """
    if interval_seconds <= 0:
        raise ValueError("interval_seconds must be > 0")

    out_dir = Path(output_dir)
    frames_dir = out_dir / "frames"
    frames_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid FPS detected (<= 0).")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration_sec = (total_frames / fps) if total_frames > 0 else None

    manifest_frames = []

    frame_idx = 0
    sample_idx = 0
    next_target_time = 0.0

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        # Approximate frame timestamp
        actual_time_sec = frame_idx / fps

        # Keep first frame at/after each target timestamp
        while actual_time_sec >= next_target_time:
            ext = image_ext.lower()
            filename = f"frame_{sample_idx:06d}.{ext}"
            out_path = frames_dir / filename

            if ext in ("jpg", "jpeg"):
                saved = cv2.imwrite(
                    str(out_path),
                    frame,
                    [int(cv2.IMWRITE_JPEG_QUALITY), int(jpeg_quality)],
                )
            elif ext == "png":
                saved = cv2.imwrite(str(out_path), frame)
            else:
                cap.release()
                raise ValueError("Unsupported image_ext. Use 'jpg', 'jpeg', or 'png'.")

            if not saved:
                cap.release()
                raise RuntimeError(f"Failed to save frame: {out_path}")

            manifest_frames.append(
                {
                    "sample_index": sample_idx,
                    "frame_index": frame_idx,
                    "target_time_sec": round(next_target_time, 6),
                    "actual_time_sec": round(actual_time_sec, 6),
                    "image_path": str(out_path),
                }
            )

            sample_idx += 1
            next_target_time += interval_seconds

        frame_idx += 1

    cap.release()

    manifest = {
        "video_path": str(video_path),
        "sampling": {
            "method": "interval_time_based",
            "interval_seconds": interval_seconds,
            "timestamp_source": "frame_index_div_fps (OpenCV estimate)",
        },
        "video_meta": {
            "fps": fps,
            "total_frames": total_frames,
            "duration_sec": round(duration_sec, 6) if duration_sec is not None else None,
        },
        "counts": {
            "saved_frames": len(manifest_frames),
        },
        "frames": manifest_frames,
    }

    manifest_path = out_dir / "frames_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    return manifest_path
