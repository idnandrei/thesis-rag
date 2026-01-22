from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

REGISTRY_PATH = Path("data/registry.json")


def _load_registry() -> dict:
    if REGISTRY_PATH.exists():
        return json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    return {"videos": {}}


def _save_registry(registry: dict) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    REGISTRY_PATH.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    raw = input("Paste/drag video file path:\n> ").strip()
    if not raw:
        print("No file selected.")
        return

    input_path = Path(raw.strip('"').strip("'")).expanduser()
    if not input_path.is_file():
        print(f"Invalid file: {input_path}")
        return

    video_id = input_path.stem  # keep your current simple rule
    raw_dir = Path("data/raw") / video_id
    derived_dir = Path("data/derived") / video_id

    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    raw_video_path = raw_dir / f"video{input_path.suffix}"
    if not raw_video_path.exists():
        shutil.copy2(input_path, raw_video_path)

    registry = _load_registry()
    registry.setdefault("videos", {})
    registry["videos"][video_id] = {
        "title": video_id,
        "added_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_video_path),
    }
    _save_registry(registry)

    print(f"Registered video '{video_id}'")
    print(f"Stored at: {raw_video_path}")
    print(f"Derived dir: {derived_dir}")


if __name__ == "__main__":
    main()

