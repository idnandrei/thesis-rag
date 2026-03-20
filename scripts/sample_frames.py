from pathlib import Path

from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id
from videorag.pipeline.frame_sampling import sample_frames_every_n_seconds


def main() -> None:
    video_id = pick_video_id("Select video to sample frames")
    paths = video_paths(video_id)

    if not paths.raw_video_path.exists():
        raise FileNotFoundError(f"Video file not found: {paths.raw_video_path}")

    raw = input("Interval seconds (default 2.0): ").strip()
    interval_seconds = float(raw) if raw else 2.0

    out_dir = paths.derived_dir / "frame_samples"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Sampling frames for '{video_id}'...")
    manifest_path = sample_frames_every_n_seconds(
        video_path=paths.raw_video_path,
        output_dir=out_dir,
        interval_seconds=interval_seconds,
        image_ext="jpg",
        jpeg_quality=90,
    )

    print(f"Frames saved to: {out_dir / 'frames'}")
    print(f"Manifest written to: {manifest_path}")


if __name__ == "__main__":
    main()
