from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_ids
from videorag.pipeline.events_builder import build_events_from_asr


def main() -> None:
    video_ids = pick_video_ids("Select videos to build events for")

    for idx, video_id in enumerate(video_ids, start=1):
        print(f"\n=== [{idx}/{len(video_ids)}] Building events for {video_id} ===")

        paths = video_paths(video_id)
        transcript_path = paths.transcript_segments_path

        logger = RunLogger("events_builder")
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
            continue

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

            print(f"Wrote: {out_path}")

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"Event building failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
