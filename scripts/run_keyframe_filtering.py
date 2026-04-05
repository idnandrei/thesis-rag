from __future__ import annotations

import shutil

from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_ids
from videorag.keyframes.manager import filter_keyframes


def main() -> None:
    settings = get_settings()
    method = settings.keyframe_filtering.method.lower().strip()

    video_ids = pick_video_ids("Select videos to filter key frames for")

    for idx, video_id in enumerate(video_ids, start=1):
        print(f"\n=== [{idx}/{len(video_ids)}] Filtering key frames for {video_id} ===")

        paths = video_paths(video_id)

        frames_manifest_path = (
            paths.derived_dir / "frame_samples" / "frames_manifest.json"
        )
        output_dir = paths.derived_dir / "key_frames"
        output_manifest_path = output_dir / "keyframes_manifest.json"

        logger = RunLogger("keyframe_filtering")
        logger.set_context(
            video_id=video_id,
            frames_manifest_path=str(frames_manifest_path),
            method=method,
        )
        logger.config_group("keyframe_filtering", settings.keyframe_filtering)

        if not frames_manifest_path.exists():
            message = f"Missing frames manifest: {frames_manifest_path}"
            logger.stage_failed("keyframe_filtering", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        print(f"Filtering key frames for '{video_id}' using '{method}'...")

        if output_dir.exists():
            shutil.rmtree(output_dir)

        try:
            with timed_logged_block("keyframe_filtering", logger):
                manifest_path = filter_keyframes(
                    video_id=video_id,
                    frames_manifest_path=frames_manifest_path,
                    output_manifest_path=output_manifest_path,
                )

            logger.stage_finished("keyframe_filtering")
            logger.set_summary(
                keyframes_dir=str(output_dir / "frames"),
                keyframes_manifest_path=str(manifest_path),
                method=method,
            )
            logger.finish(status="success")

            print(f"Key frames saved to: {output_dir / 'frames'}")
            print(f"Manifest written to: {manifest_path}")

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"Keyframe filtering failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
