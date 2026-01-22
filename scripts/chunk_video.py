from pathlib import Path

from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id
from videorag.pipeline.chunk_events import chunk_events_to_file, default_chunking_config


def main() -> None:
    video_id = pick_video_id("Select video to chunk")
    data_paths = video_paths(video_id)

    events_path = data_paths.events_path
    if not events_path.exists():
        print("events.json not found. Run ASR -> events first.")
        return

    print(f"Chunking '{video_id}'...")
    out_path = chunk_events_to_file(
        video_id=video_id,
        events_path=events_path,
        derived_root=Path("data/derived"),
        cfg=default_chunking_config(),
    )

    print("Chunks written to:", out_path)


if __name__ == "__main__":
    main()

