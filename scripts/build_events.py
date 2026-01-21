import json
from pathlib import Path

from videorag.pipeline.events_builder import build_events_from_asr

REGISTRY_PATH = Path("data/registry.json")


def main() -> None:
    if not REGISTRY_PATH.exists():
        print("❌ No registry found. Run: uv run python -m scripts.add_video")
        return

    registry = json.loads(REGISTRY_PATH.read_text(encoding="utf-8"))
    if not registry:
        print("❌ Registry is empty. Add a video first.")
        return

    print("Available videos:")
    for vid in registry:
        print(f" - {vid}")

    video_id = input("\nEnter video ID to build events for: ").strip()
    if video_id not in registry:
        print("❌ Unknown video ID.")
        return

    transcript_path = Path(f"data/derived/{video_id}/transcript_segments.json")
    if not transcript_path.exists():
        print(f"❌ Missing transcript: {transcript_path}")
        print("Run: uv run python -m scripts.transcribe_video")
        return

    out_path = build_events_from_asr(
        video_id=video_id,
        transcript_segments_path=transcript_path,
        derived_root=Path("data/derived"),
    )
    print(f"✅ Wrote: {out_path}")


if __name__ == "__main__":
    main()