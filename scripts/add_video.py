from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from videorag.config.settings import get_settings


def _load_registry(registry_path: Path) -> dict:
    if registry_path.exists():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    return {"videos": {}}


def _save_registry(registry_path: Path, registry: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    settings = get_settings()

    raw = input("Paste/drag video file path:\n> ").strip()
    if not raw:
        print("No file selected.")
        return

    input_path = Path(raw.strip('"').strip("'")).expanduser()
    if not input_path.is_file():
        print(f"Invalid file: {input_path}")
        return

    video_id = input_path.stem

    raw_dir = settings.paths.raw_dir / video_id
    derived_dir = settings.paths.derived_dir / video_id
    registry_path = settings.paths.registry_path

    raw_dir.mkdir(parents=True, exist_ok=True)
    derived_dir.mkdir(parents=True, exist_ok=True)

    raw_video_path = raw_dir / f"video{input_path.suffix}"
    if not raw_video_path.exists():
        shutil.copy2(input_path, raw_video_path)

    registry = _load_registry(registry_path)
    registry.setdefault("videos", {})
    registry["videos"][video_id] = {
        "title": video_id,
        "added_at": datetime.now(timezone.utc).isoformat(),
        "raw_path": str(raw_video_path),
    }
    _save_registry(registry_path, registry)

    print(f"Registered video '{video_id}'")
    print(f"Stored at: {raw_video_path}")
    print(f"Derived dir: {derived_dir}")


if __name__ == "__main__":
    main()
