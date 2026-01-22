from __future__ import annotations

import sys

from videorag.input.registry import list_video_ids


def pick_video_id(prompt: str = "Select video") -> str:
    """
    Numeric selection only from registry.json.

    - If registry is empty: print message and exit.
    - Otherwise: show [1..N] list and require a valid number.
    """
    ids = list_video_ids()
    if not ids:
        print("No videos found in registry.")
        print("Add a video first")
        sys.exit(1)

    print(f"{prompt}:")
    for i, vid in enumerate(ids, start=1):
        print(f"  [{i}] {vid}")

    while True:
        choice = input("Enter number: ").strip()
        if choice.isdigit():
            idx = int(choice)
            if 1 <= idx <= len(ids):
                return ids[idx - 1]
        print(f"Invalid choice. Enter a number between 1 and {len(ids)}.")
