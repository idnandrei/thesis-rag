import subprocess
from pathlib import Path
from datetime import datetime, UTC
import json

from videorag.pipeline.video_registry import register_video

REGISTRY_PATH = Path("data/registry.json")


def load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text())
    return {}


def save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(exist_ok=True)
    REGISTRY_PATH.write_text(json.dumps(registry, indent=2))


def pick_file_with_finder() -> Path | None:
    """
    Opens macOS Finder file picker and returns selected file path.
    """
    script = (
        'POSIX path of (choose file with prompt "Select a video file")'
    )

    try:
        result = subprocess.check_output(
            ["osascript", "-e", script],
            text=True,
        ).strip()
    except subprocess.CalledProcessError:
        return None

    return Path(result)


def main() -> None:
    input_video = pick_file_with_finder()

    if input_video is None:
        print("No file selected.")
        return

    video_id, raw_video_path, _ = register_video(input_video)

    registry = load_registry()
    registry[video_id] = {
        "filename": input_video.name,
        "raw_path": str(raw_video_path),
        "added_at": datetime.now(UTC).isoformat(),
    }

    save_registry(registry)

    print(f"Registered video '{video_id}'")
    print(f"Stored at: {raw_video_path}")


if __name__ == "__main__":
    main()