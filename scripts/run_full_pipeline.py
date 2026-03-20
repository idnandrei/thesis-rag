from __future__ import annotations

import json

from sqlalchemy import text

from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.db.session import db_session
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id
from videorag.pipeline.asr_transcription import transcribe_video_to_derived
from videorag.pipeline.chunk_events import chunk_events_to_file
from videorag.pipeline.embed_chunks import embed_missing_chunks_to_db
from videorag.pipeline.events_builder import build_events_from_asr

UPSERT_SQL = text(
    """
    INSERT INTO chunks (video_id, chunk_id, ts_start, ts_end, text)
    VALUES (:video_id, :chunk_id, :ts_start, :ts_end, :text)
    ON CONFLICT (video_id, chunk_id)
    DO UPDATE SET
      ts_start = EXCLUDED.ts_start,
      ts_end   = EXCLUDED.ts_end,
      text     = EXCLUDED.text
    """
)


def main() -> None:
    settings = get_settings()

    video_id = pick_video_id("Select video to run full pipeline for")
    paths = video_paths(video_id)
    raw_video_path = paths.raw_video_path

    logger = RunLogger("full_pipeline")
    logger.set_context(
        video_id=video_id,
        raw_video_path=str(raw_video_path),
    )
    logger.config_group("asr", settings.asr)
    logger.config_group("chunking", settings.chunking)
    logger.config_group(
        "embedding",
        {
            "model": settings.openai_embedding_model,
            "batch_size": settings.openai_embed_batch_size,
        },
    )

    if not raw_video_path.exists():
        message = f"Raw video not found: {raw_video_path}"
        print(message)
        print("Add the video again or check your raw folder structure.")
        logger.stage_failed("full_pipeline", error=message)
        logger.finish(status="failed")
        return

    transcript_segments_path = None
    transcript_raw_path = None
    events_path = None
    chunks_path = None
    indexed_chunk_count = 0
    embed_stats = None

    try:
        with timed_logged_block("full_pipeline", logger):
            print(f"Running full pipeline for '{video_id}'...")

            # 1) Transcription
            with timed_logged_block("transcription", logger):
                transcript_segments_path, transcript_raw_path = (
                    transcribe_video_to_derived(
                        video_id=video_id,
                        input_path=raw_video_path,
                    )
                )
            logger.stage_finished("transcription")

            # 2) Build events
            with timed_logged_block("event_building", logger):
                events_path = build_events_from_asr(
                    video_id=video_id,
                    transcript_segments_path=transcript_segments_path,
                )
            logger.stage_finished("event_building")

            # 3) Chunk events
            with timed_logged_block("chunking", logger):
                chunks_path = chunk_events_to_file(
                    video_id=video_id,
                    events_path=events_path,
                )
            logger.stage_finished("chunking")

            # 4) Index chunks into Postgres
            with timed_logged_block("indexing", logger):
                chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
                chunks = chunks_payload.get("chunks", [])
                if not isinstance(chunks, list) or not chunks:
                    raise ValueError(
                        "chunks.json has no 'chunks' list (or it's empty)."
                    )

                rows = [
                    {
                        "video_id": video_id,
                        "chunk_id": int(ch["chunk_id"]),
                        "ts_start": float(ch["ts_start"]),
                        "ts_end": float(ch["ts_end"]),
                        "text": str(ch["text"]),
                    }
                    for ch in chunks
                ]

                with db_session() as session:
                    session.execute(UPSERT_SQL, rows)

                indexed_chunk_count = len(rows)
            logger.stage_finished("indexing")

            # 5) Embed missing chunks
            with timed_logged_block("embedding", logger):
                embed_stats = embed_missing_chunks_to_db(video_id)
            logger.stage_finished("embedding")

        logger.stage_finished("full_pipeline")

        # Build summary
        transcript_segment_count = 0
        event_count = 0
        chunk_count = 0

        if transcript_segments_path is not None:
            transcript_payload = json.loads(
                transcript_segments_path.read_text(encoding="utf-8")
            )
            transcript_segment_count = len(transcript_payload.get("segments", []))

        if events_path is not None:
            events_payload = json.loads(events_path.read_text(encoding="utf-8"))
            event_count = len(events_payload.get("events", []))

        if chunks_path is not None:
            chunks_payload = json.loads(chunks_path.read_text(encoding="utf-8"))
            chunk_count = len(chunks_payload.get("chunks", []))

        logger.set_summary(
            transcript_segments_path=(
                str(transcript_segments_path) if transcript_segments_path else None
            ),
            transcript_raw_path=(
                str(transcript_raw_path) if transcript_raw_path else None
            ),
            events_path=str(events_path) if events_path else None,
            chunks_path=str(chunks_path) if chunks_path else None,
            transcript_segment_count=transcript_segment_count,
            event_count=event_count,
            chunk_count=chunk_count,
            indexed_chunk_count=indexed_chunk_count,
            embedded_chunk_count=embed_stats.embedded if embed_stats else 0,
            embedding_model=(
                embed_stats.model if embed_stats else settings.openai_embedding_model
            ),
            embedding_dim=embed_stats.dim if embed_stats else 0,
            output_dir=str(paths.derived_dir),
        )
        logger.finish(status="success")

    except Exception as e:
        logger.finish(status="failed", error=str(e))
        raise

    print(f"Full pipeline completed for '{video_id}'")
    print(f"Transcript: {transcript_segments_path}")
    print(f"Events: {events_path}")
    print(f"Chunks: {chunks_path}")
    print(f"Indexed chunks: {indexed_chunk_count}")
    if embed_stats is not None:
        print(f"Embedded chunks: {embed_stats.embedded}")
        print(f"Embedding model: {embed_stats.model}")
        print(f"Embedding dim: {embed_stats.dim}")


if __name__ == "__main__":
    main()
