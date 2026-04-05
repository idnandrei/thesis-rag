from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import get_video_source_path, pick_video_ids
from videorag.pipeline.asr_transcription import transcribe_video_to_derived


def main() -> None:
    settings = get_settings()

    video_ids = pick_video_ids("Select videos to transcribe")

    for idx, video_id in enumerate(video_ids, start=1):
        print(f"\n=== [{idx}/{len(video_ids)}] Transcribing {video_id} ===")

        paths = video_paths(video_id)
        raw_video_path = get_video_source_path(video_id)

        logger = RunLogger("asr_transcription")
        logger.set_context(
            video_id=video_id,
            raw_video_path=str(raw_video_path),
        )
        logger.config_group("asr", settings.asr)

        if not raw_video_path.exists():
            message = f"Raw video not found: {raw_video_path}"
            print(message)
            print("Check the registry raw_path or dataset source path.")

            logger.stage_failed("transcription", error=message)
            logger.finish(status="failed")
            continue

        print(f"Transcribing '{video_id}'...")

        try:
            with timed_logged_block("transcription", logger):
                transcript_segments_path, transcript_raw_path = (
                    transcribe_video_to_derived(
                        video_id=video_id,
                        input_path=raw_video_path,
                    )
                )

            logger.stage_finished("transcription")

            logger.set_summary(
                transcript_segments_path=str(transcript_segments_path),
                transcript_raw_path=str(transcript_raw_path),
                output_dir=str(paths.derived_dir),
            )
            logger.finish(status="success")

            print(f"Transcription completed for '{video_id}'")
            print(f"Output in: {paths.derived_dir}")

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"Transcription failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
