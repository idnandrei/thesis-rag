from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import get_video_source_path, pick_video_ids
from videorag.pipeline.frame_sampling import sample_frames_every_n_seconds


def main() -> None:
    settings = get_settings()

    video_ids = pick_video_ids("Select videos to sample frames")

    raw = input(
        f"Interval seconds (default {settings.frame_sampling.interval_seconds}): "
    ).strip()
    interval_seconds = float(raw) if raw else settings.frame_sampling.interval_seconds

    for idx, video_id in enumerate(video_ids, start=1):
        print(f"\n=== [{idx}/{len(video_ids)}] Sampling frames for {video_id} ===")

        paths = video_paths(video_id)
        raw_video_path = get_video_source_path(video_id)

        logger = RunLogger("frame_sampling")
        logger.set_context(
            video_id=video_id,
            raw_video_path=str(raw_video_path),
        )
        logger.config_group("frame_sampling", settings.frame_sampling)

        if not raw_video_path.exists():
            message = f"Video file not found: {raw_video_path}"
            logger.stage_failed("frame_sampling", error=message)
            logger.finish(status="failed")
            print(message)
            continue

        out_dir = paths.derived_dir / "frame_samples"
        out_dir.mkdir(parents=True, exist_ok=True)

        print(f"Sampling frames for '{video_id}'...")

        try:
            with timed_logged_block("frame_sampling", logger):
                manifest_path = sample_frames_every_n_seconds(
                    video_path=raw_video_path,
                    output_dir=out_dir,
                    interval_seconds=interval_seconds,
                    image_ext=settings.frame_sampling.image_ext,
                    jpeg_quality=settings.frame_sampling.jpeg_quality,
                )

            logger.stage_finished("frame_sampling")
            logger.set_summary(
                frames_dir=str(out_dir / "frames"),
                manifest_path=str(manifest_path),
                interval_seconds=interval_seconds,
            )
            logger.finish(status="success")

            print(f"Frames saved to: {out_dir / 'frames'}")
            print(f"Manifest written to: {manifest_path}")

        except Exception as e:
            logger.finish(status="failed", error=str(e))
            print(f"Frame sampling failed for '{video_id}': {e}")


if __name__ == "__main__":
    main()
