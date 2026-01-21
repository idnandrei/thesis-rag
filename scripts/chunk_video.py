import json
from pathlib import Path

from videorag.pipeline.chunk_events import (
    chunk_events_to_file,
    default_chunking_config,
)

REGISTRY_PATH = Path("data/registry.json")


def main() -> None:
    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))

    print("Available videos:")
    for vid in registry:
        print(f" - {vid}")

    video_id = input("\nEnter video ID to chunk: ").strip()
    if video_id not in registry:
        print("❌ Unknown video ID.")
        return

    events_path = Path(f"data/derived/{video_id}/events.json")
    if not events_path.exists():
        print("❌ events.json not found. Run ASR → events first.")
        return

    print(f"\n▶ Chunking '{video_id}'...")
    out_path = chunk_events_to_file(
        video_id=video_id,
        events_path=events_path,
        derived_root=Path("data/derived"),
        cfg=default_chunking_config(),
    )

    print("✅ Chunks written to:", out_path)


if __name__ == "__main__":
    main()