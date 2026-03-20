from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from sqlalchemy import text

from videorag.db.session import db_session


@dataclass(frozen=True)
class ChunkRow:
    chunk_id: int
    ts_start: float
    ts_end: float
    text: str


@dataclass(frozen=True)
class MissingEmbeddingRow:
    chunk_id: int
    text: str


@dataclass(frozen=True)
class SearchResultRow:
    chunk_id: int
    ts_start: float
    ts_end: float
    text: str
    cosine_distance: float


UPSERT_CHUNKS_SQL = text(
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

SELECT_MISSING_EMBEDDINGS_SQL = text(
    """
    SELECT chunk_id, text
    FROM chunks
    WHERE video_id = :video_id
      AND embedding IS NULL
    ORDER BY chunk_id
    """
)

UPDATE_EMBEDDINGS_SQL = text(
    """
    UPDATE chunks
    SET embedding = :embedding
    WHERE video_id = :video_id
      AND chunk_id = :chunk_id
    """
)

SEARCH_SIMILAR_SQL = text(
    """
    SELECT
      chunk_id,
      ts_start,
      ts_end,
      text,
      (embedding <=> CAST(:qvec AS vector)) AS cosine_distance
    FROM chunks
    WHERE video_id = :video_id
      AND embedding IS NOT NULL
    ORDER BY embedding <=> CAST(:qvec AS vector)
    LIMIT :k
    """
)


class ChunkRepository:
    def upsert_chunks(self, video_id: str, chunks: Iterable[dict[str, Any]]) -> int:
        rows = [
            {
                "video_id": video_id,
                "chunk_id": int(chunk["chunk_id"]),
                "ts_start": float(chunk["ts_start"]),
                "ts_end": float(chunk["ts_end"]),
                "text": str(chunk["text"]),
            }
            for chunk in chunks
        ]

        if not rows:
            return 0

        with db_session() as session:
            session.execute(UPSERT_CHUNKS_SQL, rows)

        return len(rows)

    def get_missing_embeddings(self, video_id: str) -> list[MissingEmbeddingRow]:
        with db_session() as session:
            result = session.execute(
                SELECT_MISSING_EMBEDDINGS_SQL,
                {"video_id": video_id},
            ).all()

        return [
            MissingEmbeddingRow(
                chunk_id=int(row[0]),
                text=str(row[1]),
            )
            for row in result
        ]

    def update_embeddings(
        self,
        video_id: str,
        embeddings: Iterable[tuple[int, list[float]]],
    ) -> int:
        rows = [
            {
                "video_id": video_id,
                "chunk_id": int(chunk_id),
                "embedding": vector,
            }
            for chunk_id, vector in embeddings
        ]

        if not rows:
            return 0

        with db_session() as session:
            session.execute(UPDATE_EMBEDDINGS_SQL, rows)

        return len(rows)

    def search_similar(
        self,
        *,
        video_id: str,
        query_vector_literal: str,
        top_k: int,
    ) -> list[SearchResultRow]:
        with db_session() as session:
            rows = (
                session.execute(
                    SEARCH_SIMILAR_SQL,
                    {
                        "video_id": video_id,
                        "qvec": query_vector_literal,
                        "k": top_k,
                    },
                )
                .mappings()
                .all()
            )

        return [
            SearchResultRow(
                chunk_id=int(row["chunk_id"]),
                ts_start=float(row["ts_start"]),
                ts_end=float(row["ts_end"]),
                text=str(row["text"]),
                cosine_distance=float(row["cosine_distance"]),
            )
            for row in rows
        ]
