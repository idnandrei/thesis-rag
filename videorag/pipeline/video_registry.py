from pathlib import Path
import shutil


def register_video(
    input_video: Path,
    raw_root: Path = Path("data/raw"),
    derived_root: Path = Path("data/derived"),
) -> tuple[str, Path, Path]:
    """
    Register a video into the pipeline.

    - Derives video_id from filename (stem)
    - Creates:
        data/raw/<video_id>/video.<ext>
        data/derived/<video_id>/
    - Copies the video into the raw folder

    Returns:
        (video_id, raw_video_path, derived_video_dir)
    """
    if not input_video.exists():
        raise FileNotFoundError(f"Video not found: {input_video}")

    video_id = input_video.stem

    # Create raw video folder
    raw_video_dir = raw_root / video_id
    raw_video_dir.mkdir(parents=True, exist_ok=True)

    raw_video_path = raw_video_dir / f"video{input_video.suffix}"

    # Copy video if not already there
    if not raw_video_path.exists():
        shutil.copy2(input_video, raw_video_path)

    # Create derived folder
    derived_video_dir = derived_root / video_id
    derived_video_dir.mkdir(parents=True, exist_ok=True)

    return video_id, raw_video_path, derived_video_dir