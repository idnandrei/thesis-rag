import json
from pathlib import Path

from videorag.pipeline.asr_transcription import transcribe_video_to_derived

REGISTRY_PATH = Path("data/registry.json")


def main() -> None:
    if not REGISTRY_PATH.exists():
        print("Video registry not found. Add an initial video first.")
        return

    registry = json.loads(REGISTRY_PATH.read_text())

    print("Available videos:")
    for vid in registry:
        print(f" - {vid}")

    video_id = input("\nEnter video ID to transcribe: ").strip()

    if video_id not in registry:
        print("Unknown video ID.")
        return

    raw_video_path = Path(registry[video_id]["raw_path"])

    print(f"\nTranscribing '{video_id}'...")
    transcribe_video_to_derived(
        video_id=video_id,
        input_path=raw_video_path,
    )

    print(f"Transcription completed for '{video_id}'")
    print(f"   Output in: data/derived/{video_id}/")


if __name__ == "__main__":
    main()