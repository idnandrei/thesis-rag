from __future__ import annotations

import shutil

from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_ids
from videorag.pipeline.person_filter import run_person_filter


def main() -> None:
    settings = get_settings()

    video_ids = pick_video_ids("Select videos to filter person crops for")

    for idx, video_id in enumerate(video_ids, start=1):
        print(
            f"\n=== [{idx}/{len(video_ids)}] Filtering person crops for {video_id} ==="
        )

        paths = video_paths(video_id)

        visual_objects_dir = paths.derived_dir / "visual_objects"
        visual_objects_manifest_path = visual_objects_dir / "visual_objects.json"
        crops_dir = visual_objects_dir / "crops"
        output_dir = paths.derived_dir / "visual_person_filter_yolo"

        logger = RunLogger("person_filter")
        logger.set_context(
            video_id=video_id,
            visual_objects_manifest_path=str(visual_objects_manifest_path),
            crops_dir=str(crops_dir),
        )
        logger.config_group("person_filter", settings.person_filter)

        if not visual_objects_manifest_path.exists():
            message = f"Missing visual objects manifest: {visual_objects_manifest_path}"
            logger.stage_failed("person_filter", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        if not crops_dir.exists():
            message = f"Missing crops directory: {crops_dir}"
            logger.stage_failed("person_filter", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        if output_dir.exists():
            shutil.rmtree(output_dir)

        try:
            with timed_logged_block("person_filter", logger):
                results_path = run_person_filter(
                    video_id=video_id,
                    visual_objects_manifest_path=visual_objects_manifest_path,
                    crops_dir=crops_dir,
                    output_dir=output_dir,
                    model_name=settings.person_filter.model_name,
                    person_conf_threshold=settings.person_filter.person_conf_threshold,
                    person_box_ratio_threshold=settings.person_filter.person_box_ratio_threshold,
                )

            logger.stage_finished("person_filter")
            logger.set_summary(
                results_path=str(results_path),
                kept_crops_dir=str(output_dir / "kept_crops"),
                removed_crops_dir=str(output_dir / "removed_crops"),
            )
            logger.finish(status="success")

            print(f"Results written to: {results_path}")
            print(f"Kept crops dir: {output_dir / 'kept_crops'}")
            print(f"Removed crops dir: {output_dir / 'removed_crops'}")

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"Person filtering failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
