from __future__ import annotations

import shutil
from pathlib import Path

from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_ids
from videorag.pipeline.obj_detection import extract_visual_objects


def main() -> None:
    settings = get_settings()

    video_ids = pick_video_ids("Select videos to extract visual objects for")
    model_path = Path(settings.visual_extraction.model_path)

    for idx, video_id in enumerate(video_ids, start=1):
        print(
            f"\n=== [{idx}/{len(video_ids)}] Extracting visual objects for {video_id} ==="
        )

        paths = video_paths(video_id)

        frame_manifest_path = (
            paths.derived_dir / "key_frames" / "keyframes_manifest.json"
        )
        output_dir = paths.derived_dir / "visual_objects"

        logger = RunLogger("visual_object_extraction")
        logger.set_context(
            video_id=video_id,
            frame_manifest_path=str(frame_manifest_path),
        )
        logger.config_group("visual_extraction", settings.visual_extraction)

        if not frame_manifest_path.exists():
            message = f"Missing keyframe manifest: {frame_manifest_path}"
            logger.stage_failed("visual_object_extraction", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        if not model_path.exists():
            message = f"Missing model file: {model_path}"
            logger.stage_failed("visual_object_extraction", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        print(f"Extracting visual objects for '{video_id}' from key frames...")

        if output_dir.exists():
            shutil.rmtree(output_dir)

        try:
            with timed_logged_block("visual_object_extraction", logger):
                manifest_path = extract_visual_objects(
                    video_id=video_id,
                    frame_manifest_path=frame_manifest_path,
                    model_path=model_path,
                    output_dir=output_dir,
                )

            logger.stage_finished("visual_object_extraction")
            logger.set_summary(
                visual_objects_dir=str(output_dir),
                visual_objects_manifest_path=str(manifest_path),
                annotated_frames_dir=str(output_dir / "annotated_frames"),
                crops_dir=str(output_dir / "crops"),
            )
            logger.finish(status="success")

            print(f"Annotated frames saved to: {output_dir / 'annotated_frames'}")
            print(f"Crops saved to: {output_dir / 'crops'}")
            print(f"Manifest written to: {manifest_path}")

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"Visual object extraction failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
