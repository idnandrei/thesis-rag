from __future__ import annotations

from videorag.input.select_video import pick_video_id
from videorag.pipeline.embed_chunks import embed_missing_chunks_to_db


def main() -> None:
    video_id = pick_video_id("Select video to embed chunks for")
    stats = embed_missing_chunks_to_db(video_id)

    if stats.embedded == 0:
        print("No chunks needed embeddings.")
        return

    print(f"Embedded {stats.embedded} chunks for '{stats.video_id}'.")
    print(f"Model: {stats.model}")
    print(f"Dim: {stats.dim}")


if __name__ == "__main__":
    main()
