from __future__ import annotations

import json

from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id
from videorag.repos.chunk_repo import ChunkRepository


def main() -> None:
    video_id = pick_video_id("Select video to index chunks for")
    paths = video_paths(video_id)
    chunks_path = paths.chunks_path

    logger = RunLogger("index_chunks")
    logger.set_context(
        video_id=video_id,
        chunks_path=str(chunks_path),
    )

    if not chunks_path.exists():
        message = f"Missing chunks file: {chunks_path}"
        logger.stage_failed("indexing", error=message)
        logger.finish(status="failed")
        raise FileNotFoundError(message)

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        message = "chunks.json has no 'chunks' list (or it's empty)."
        logger.stage_failed("indexing", error=message)
        logger.finish(status="failed")
        raise ValueError(message)

    repo = ChunkRepository()

    try:
        with timed_logged_block("indexing", logger):
            indexed_chunk_count = repo.upsert_chunks(video_id, chunks)

        logger.stage_finished("indexing")
        logger.set_summary(
            indexed_chunk_count=indexed_chunk_count,
            chunks_path=str(chunks_path),
        )
        logger.finish(status="success")

    except Exception as e:
        logger.finish(status="failed", error=str(e))
        raise

    print(f"Indexed {indexed_chunk_count} chunks for video '{video_id}' into Postgres.")


if __name__ == "__main__":
    main()
