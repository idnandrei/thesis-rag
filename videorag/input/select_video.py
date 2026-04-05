from __future__ import annotations

import json
import sys
from pathlib import Path

from videorag.config.settings import get_settings
from videorag.input.registry import list_video_ids


def _load_registry() -> dict:
    settings = get_settings()
    registry_path = settings.paths.registry_path

    if not registry_path.exists():
        return {"videos": {}}

    return json.loads(registry_path.read_text(encoding="utf-8"))


def _group_video_ids() -> dict[str, list[str]]:
    registry = _load_registry()
    videos = registry.get("videos", {})

    grouped: dict[str, list[str]] = {}

    for video_id, meta in videos.items():
        if not isinstance(meta, dict):
            continue

        group_name = str(meta.get("group_name", "ungrouped")).strip() or "ungrouped"
        grouped.setdefault(group_name, []).append(video_id)

    for group_name in grouped:
        grouped[group_name] = sorted(grouped[group_name])

    return dict(sorted(grouped.items()))


def get_video_source_path(video_id: str) -> Path:
    registry = _load_registry()
    videos = registry.get("videos", {})

    if video_id not in videos:
        raise KeyError(f"Video '{video_id}' not found in registry.")

    raw_path = videos[video_id].get("raw_path")
    if not raw_path:
        raise ValueError(f"Video '{video_id}' has no raw_path in registry.")

    source_path = Path(str(raw_path))
    if not source_path.exists():
        raise FileNotFoundError(f"Source video file not found: {source_path}")

    return source_path


def _pick_one_video(prompt: str) -> list[str]:
    ids = list_video_ids()
    if not ids:
        print("No videos found in registry.")
        print("Add a video first")
        sys.exit(1)

    print(f"{prompt}:")
    for i, vid in enumerate(ids, start=1):
        print(f"  [{i}] {vid}")

    print("You can enter either the number or the exact video name.")

    while True:
        choice = input("Enter number or video name: ").strip()
        if not choice:
            print("Invalid choice. Enter a number or an exact video name.")
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(ids):
                return [ids[idx - 1]]

        if choice in ids:
            return [choice]

        print(
            f"Invalid choice. Enter a number between 1 and {len(ids)}, "
            "or an exact video name from the list."
        )


def _pick_group(prompt: str) -> list[str]:
    grouped = _group_video_ids()

    if not grouped:
        print("No grouped videos found in registry.")
        print("Add videos first")
        sys.exit(1)

    group_names = list(grouped.keys())

    print(f"{prompt}:")
    for i, group_name in enumerate(group_names, start=1):
        print(f"  [{i}] {group_name} ({len(grouped[group_name])} videos)")

    print("You can enter either the number or the exact group name.")

    while True:
        choice = input("Enter number or group name: ").strip()
        if not choice:
            print("Invalid choice. Enter a number or an exact group name.")
            continue

        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(group_names):
                return grouped[group_names[idx - 1]]

        if choice in grouped:
            return grouped[choice]

        print(
            f"Invalid choice. Enter a number between 1 and {len(group_names)}, "
            "or an exact group name from the list."
        )


def pick_video_ids(prompt: str = "Select target videos") -> list[str]:
    ids = list_video_ids()
    if not ids:
        print("No videos found in registry.")
        print("Add a video first")
        sys.exit(1)

    print(f"{prompt}:")
    print("  [1] Individual video")
    print("  [2] Group")
    print("  [3] All videos")

    while True:
        mode = input("Choose mode (1/2/3): ").strip()

        if mode == "1":
            return _pick_one_video("Select one video")

        if mode == "2":
            return _pick_group("Select one group")

        if mode == "3":
            return sorted(ids)

        print("Invalid choice. Enter 1, 2, or 3.")


def pick_video_id(prompt: str = "Select video") -> str:
    """
    Backward-compatible single-video selector.

    Uses the new selection flow but requires the final result
    to contain exactly one video.
    """
    selected_ids = pick_video_ids(prompt)

    if len(selected_ids) != 1:
        print("This operation requires exactly one video.")
        print("Please choose 'Individual video'.")
        sys.exit(1)

    return selected_ids[0]
