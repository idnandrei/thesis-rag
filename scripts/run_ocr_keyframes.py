from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_ids
from videorag.pipeline.ocr_keyframes import run_ocr_on_keyframes


def main() -> None:
    settings = get_settings()

    video_ids = pick_video_ids("Select videos to OCR keyframes for")

    for idx, video_id in enumerate(video_ids, start=1):
        print(f"\n=== [{idx}/{len(video_ids)}] OCR keyframes for {video_id} ===")

        paths = video_paths(video_id)
        keyframes_manifest_path = (
            paths.derived_dir / "key_frames" / "keyframes_manifest.json"
        )
        output_dir = paths.derived_dir / "ocr_keyframes"

        logger = RunLogger("ocr_keyframes")
        logger.set_context(
            video_id=video_id,
            keyframes_manifest_path=str(keyframes_manifest_path),
        )
        logger.config_group("ocr", settings.ocr)

        if not keyframes_manifest_path.exists():
            message = f"Missing keyframes manifest: {keyframes_manifest_path}"
            logger.stage_failed("ocr_keyframes", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        print(f"Running OCR on keyframes for '{video_id}'...")

        try:
            with timed_logged_block("ocr_keyframes", logger):
                results_path = run_ocr_on_keyframes(
                    keyframes_manifest_path=keyframes_manifest_path,
                    output_dir=output_dir,
                    languages=["en"],
                    min_confidence=settings.ocr.min_confidence,
                    use_gpu=settings.ocr.use_gpu,
                )

            logger.stage_finished("ocr_keyframes")
            logger.set_summary(
                ocr_output_dir=str(output_dir),
                ocr_results_path=str(results_path),
                ocr_text_path=str(output_dir / "ocr_text.txt"),
            )
            logger.finish(status="success")

            print(f"OCR results written to: {results_path}")
            print(f"OCR text written to: {output_dir / 'ocr_text.txt'}")

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"OCR failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
