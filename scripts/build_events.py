from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id
from videorag.pipeline.events_builder import build_events_from_asr


def main() -> None:
    video_id = pick_video_id("Select video to build events for")
    paths = video_paths(video_id)
    transcript_path = paths.transcript_segments_path

    logger = RunLogger("build_events")
    logger.set_context(
        video_id=video_id,
        transcript_path=str(transcript_path),
    )

    if not transcript_path.exists():
        message = f"Missing transcript: {transcript_path}"
        print(message)
        print("Run your transcribe script first.")

        logger.stage_failed("event_building", error=message)
        logger.finish(status="failed")
        return

    try:
        with timed_logged_block("event_building", logger):
            out_path = build_events_from_asr(
                video_id=video_id,
                transcript_segments_path=transcript_path,
            )

        logger.stage_finished("event_building")
        logger.set_summary(
            events_path=str(out_path),
        )
        logger.finish(status="success")

    except Exception as e:
        logger.finish(status="failed", error=str(e))
        raise

    print(f"Wrote: {out_path}")


if __name__ == "__main__":
    main()
