from pathlib import Path

from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id
from videorag.pipeline.events_builder import build_events_from_asr


def main() -> None:
    video_id = pick_video_id("Select video to build events for")
    paths = video_paths(video_id)

    transcript_path = paths.transcript_segments_path
    if not transcript_path.exists():
        print(f"Missing transcript: {transcript_path}")
        print("Run your transcribe script first.")
        return

    out_path = build_events_from_asr(
        video_id=video_id,
        transcript_segments_path=transcript_path,
        derived_root=Path("data/derived"),
    )
    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()

