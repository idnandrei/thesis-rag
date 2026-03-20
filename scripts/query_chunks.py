from __future__ import annotations

from typing import List

from openai import OpenAI

from videorag.config.settings import get_settings
from videorag.core.run_logger import RunLogger
from videorag.core.timer import timed_logged_block
from videorag.input.select_video import pick_video_id
from videorag.repos.chunk_repo import ChunkRepository


def _vec_to_pgvector_literal(vec: List[float]) -> str:
    return "[" + ",".join(f"{x:.8f}" for x in vec) + "]"


def main() -> None:
    settings = get_settings()
    client = OpenAI(api_key=settings.openai_api_key)
    repo = ChunkRepository()

    video_id = pick_video_id("Select video to search")

    query = input("Question: ").strip()
    if not query:
        raise ValueError("Question cannot be empty.")

    k_raw = input(f"Top-K (default {settings.retrieval.top_k}): ").strip()
    top_k = int(k_raw) if k_raw else settings.retrieval.top_k

    logger = RunLogger("query_chunks")
    logger.set_context(
        video_id=video_id,
        query=query,
        top_k=top_k,
    )
    logger.config_group("retrieval", settings.retrieval)
    logger.config_group(
        "embedding",
        {
            "model": settings.openai_embedding_model,
        },
    )

    try:
        with timed_logged_block("retrieval", logger):
            resp = client.embeddings.create(
                model=settings.openai_embedding_model,
                input=[query],
            )
            query_vector = resp.data[0].embedding
            query_vector_literal = _vec_to_pgvector_literal(query_vector)

            hits = repo.search_similar(
                video_id=video_id,
                query_vector_literal=query_vector_literal,
                top_k=top_k,
            )

        logger.stage_finished("retrieval")
        logger.set_summary(
            result_count=len(hits),
            embedding_dim=len(query_vector),
        )
        logger.finish(status="success")

    except Exception as e:
        logger.finish(status="failed", error=str(e))
        raise

    print(f"Embedding dims: {len(query_vector)}")

    if not hits:
        print("No results found.")
        return

    print("\nTop matches:\n")
    for hit in hits:
        print(
            f"- chunk_id={hit.chunk_id}  "
            f"[{hit.ts_start:.2f}s–{hit.ts_end:.2f}s]  "
            f"dist={hit.cosine_distance:.4f}"
        )
        print(hit.text)
        print("-" * 80)


if __name__ == "__main__":
    main()
