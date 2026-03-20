from __future__ import annotations

from dataclasses import dataclass

from openai import OpenAI

from videorag.config.settings import get_settings
from videorag.repos.chunk_repo import ChunkRepository


@dataclass(frozen=True)
class EmbedStats:
    video_id: str
    embedded: int
    dim: int
    model: str


def embed_missing_chunks_to_db(video_id: str) -> EmbedStats:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    repo = ChunkRepository()

    missing = repo.get_missing_embeddings(video_id)

    if not missing:
        return EmbedStats(
            video_id=video_id,
            embedded=0,
            dim=0,
            model=settings.openai_embedding_model,
        )

    chunk_ids = [row.chunk_id for row in missing]
    texts = [row.text for row in missing]

    vectors: list[list[float]] = []
    for i in range(0, len(texts), settings.openai_embed_batch_size):
        batch = texts[i : i + settings.openai_embed_batch_size]
        resp = client.embeddings.create(
            model=settings.openai_embedding_model,
            input=batch,
        )
        vectors.extend([item.embedding for item in resp.data])

    if len(vectors) != len(texts):
        raise RuntimeError("Embedding count mismatch")

    dim = len(vectors[0]) if vectors else 0
    embedded = repo.update_embeddings(video_id, zip(chunk_ids, vectors))

    return EmbedStats(
        video_id=video_id,
        embedded=embedded,
        dim=dim,
        model=settings.openai_embedding_model,
    )
