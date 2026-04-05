from __future__ import annotations

import shutil

from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_ids
from videorag.pipeline.image_caption import run_image_caption


def main() -> None:
    settings = get_settings()

    video_ids = pick_video_ids("Select videos to caption visual objects for")

    for idx, video_id in enumerate(video_ids, start=1):
        print(
            f"\n=== [{idx}/{len(video_ids)}] Captioning visual objects for {video_id} ==="
        )

        paths = video_paths(video_id)

        visual_objects_dir = paths.derived_dir / "visual_objects"
        visual_objects_manifest_path = visual_objects_dir / "visual_objects.json"
        caption_source_dir = (
            paths.derived_dir / "visual_person_filter_yolo" / "kept_crops"
        )
        output_dir = paths.derived_dir / "image_caption"

        logger = RunLogger("image_caption")
        logger.set_context(
            video_id=video_id,
            visual_objects_manifest_path=str(visual_objects_manifest_path),
            caption_source_dir=str(caption_source_dir),
        )
        logger.config_group("image_caption", settings.image_caption)

        if not visual_objects_manifest_path.exists():
            message = f"Missing visual objects manifest: {visual_objects_manifest_path}"
            logger.stage_failed("image_caption", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        if not caption_source_dir.exists():
            message = f"Missing kept crops directory: {caption_source_dir}"
            logger.stage_failed("image_caption", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        print(f"Running image captioning for '{video_id}'...")

        if output_dir.exists():
            shutil.rmtree(output_dir)

        try:
            with timed_logged_block("image_caption", logger):
                results_path = run_image_caption(
                    video_id=video_id,
                    visual_objects_manifest_path=visual_objects_manifest_path,
                    caption_source_dir=caption_source_dir,
                    output_dir=output_dir,
                    model_name=settings.image_caption.model_name,
                    max_new_tokens=settings.image_caption.max_new_tokens,
                    do_sample=settings.image_caption.do_sample,
                )

            logger.stage_finished("image_caption")
            logger.set_summary(
                image_caption_output_dir=str(output_dir),
                image_caption_results_path=str(results_path),
                image_caption_text_path=str(output_dir / "captions.txt"),
            )
            logger.finish(status="success")

            print(f"Image caption results written to: {results_path}")
            print(f"Image caption text written to: {output_dir / 'captions.txt'}")

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"Image captioning failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
