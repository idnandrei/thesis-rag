from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_ids
from videorag.pipeline.chunk_events import chunk_events_to_file


def main() -> None:
    settings = get_settings()

    video_ids = pick_video_ids("Select videos to chunk")

    for idx, video_id in enumerate(video_ids, start=1):
        print(f"\n=== [{idx}/{len(video_ids)}] Chunking {video_id} ===")

        data_paths = video_paths(video_id)
        events_path = data_paths.events_path

        logger = RunLogger("chunk_events")
        logger.set_context(
            video_id=video_id,
            events_path=str(events_path),
        )
        logger.config_group("chunking", settings.chunking)

        if not events_path.exists():
            message = "events.json not found. Run ASR -> events first."
            print(message)
            logger.stage_failed("chunking", error=message)
            logger.finish(status="failed")
            continue

        print(f"Chunking '{video_id}'...")

        try:
            with timed_logged_block("chunking", logger):
                out_path = chunk_events_to_file(
                    video_id=video_id,
                    events_path=events_path,
                )

            logger.stage_finished("chunking")

            logger.set_summary(
                chunks_path=str(out_path),
                output_dir=str(data_paths.derived_dir),
            )
            logger.finish(status="success")

            print("Chunks written to:", out_path)

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"Chunking failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
