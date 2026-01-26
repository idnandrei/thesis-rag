from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

from openai import OpenAI
from sqlalchemy import text

from videorag.config.settings import get_settings
from videorag.db.session import db_session


@dataclass(frozen=True)
class EmbedStats:
    video_id: str
    embedded: int
    dim: int
    model: str


SELECT_MISSING = text(
    """
    SELECT chunk_id, text
    FROM chunks
    WHERE video_id = :video_id
      AND embedding IS NULL
    ORDER BY chunk_id
    """
)

UPDATE_EMBEDDING = text(
    """
    UPDATE chunks
    SET embedding = :embedding
    WHERE video_id = :video_id AND chunk_id = :chunk_id
    """
)


def embed_missing_chunks_to_db(video_id: str) -> EmbedStats:
    s = get_settings()
    client = OpenAI(api_key=s.openai_api_key)

    # 1) Fetch chunks that still need embeddings
    with db_session() as session:
        rows: List[Tuple[int, str]] = list(
            session.execute(SELECT_MISSING, {"video_id": video_id}).all()
        )

    if not rows:
        return EmbedStats(
            video_id=video_id, embedded=0, dim=0, model=s.openai_embedding_model
        )

    chunk_ids = [int(r[0]) for r in rows]
    texts = [str(r[1]) for r in rows]

    # 2) Embed in batches
    vectors: list[list[float]] = []
    for i in range(0, len(texts), s.openai_embed_batch_size):
        batch = texts[i : i + s.openai_embed_batch_size]
        resp = client.embeddings.create(model=s.openai_embedding_model, input=batch)
        vectors.extend([item.embedding for item in resp.data])

    if len(vectors) != len(texts):
        raise RuntimeError("Embedding count mismatch")

    dim = len(vectors[0]) if vectors else 0

    # 3) Update DB
    updates = [
        {"video_id": video_id, "chunk_id": cid, "embedding": vec}
        for cid, vec in zip(chunk_ids, vectors)
    ]

    with db_session() as session:
        session.execute(UPDATE_EMBEDDING, updates)

    return EmbedStats(
        video_id=video_id,
        embedded=len(updates),
        dim=dim,
        model=s.openai_embedding_model,
    )
