from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id
from videorag.pipeline.asr_transcription import transcribe_video_to_derived


def main() -> None:
    settings = get_settings()

    video_id = pick_video_id("Select video to transcribe")
    paths = video_paths(video_id)
    raw_video_path = paths.raw_video_path

    logger = RunLogger("transcribe_video")
    logger.set_context(
        video_id=video_id,
        raw_video_path=str(raw_video_path),
    )
    logger.config_group("asr", settings.asr)

    if not raw_video_path.exists():
        message = f"Raw video not found: {raw_video_path}"
        print(message)
        print("Add the video again or check your raw folder structure.")

        logger.stage_failed("transcription", error=message)
        logger.finish(status="failed")
        return

    print(f"Transcribing '{video_id}'...")

    try:
        with timed_logged_block("transcription", logger):
            transcript_segments_path, transcript_raw_path = transcribe_video_to_derived(
                video_id=video_id,
                input_path=raw_video_path,
            )

        logger.stage_finished("transcription")

        logger.set_summary(
            transcript_segments_path=str(transcript_segments_path),
            transcript_raw_path=str(transcript_raw_path),
            output_dir=str(paths.derived_dir),
        )
        logger.finish(status="success")

    except Exception as e:
        logger.finish(status="failed", error=str(e))
        raise

    print(f"Transcription completed for '{video_id}'")
    print(f"Output in: {paths.derived_dir}")


if __name__ == "__main__":
    main()
