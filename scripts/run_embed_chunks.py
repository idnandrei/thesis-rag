from __future__ import annotations

from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.select_video import pick_video_id
from videorag.pipeline.embed_chunks import embed_missing_chunks_to_db


def main() -> None:
    settings = get_settings()
    video_id = pick_video_id("Select video to embed chunks for")

    logger = RunLogger("embed_chunks")
    logger.set_context(video_id=video_id)
    logger.config_group(
        "embedding",
        {
            "model": settings.openai_embedding_model,
            "batch_size": settings.openai_embed_batch_size,
        },
    )

    try:
        with timed_logged_block("embedding", logger):
            stats = embed_missing_chunks_to_db(video_id)

        logger.stage_finished("embedding")

        logger.set_summary(
            embedded_chunk_count=stats.embedded,
            model=stats.model,
            dim=stats.dim,
        )
        logger.finish(status="success")

    except Exception as e:
        logger.finish(status="failed", error=str(e))
        raise

    if stats.embedded == 0:
        print("No chunks needed embeddings.")
        return

    print(f"Embedded {stats.embedded} chunks for '{stats.video_id}'.")
    print(f"Model: {stats.model}")
    print(f"Dim: {stats.dim}")


if __name__ == "__main__":
    main()
