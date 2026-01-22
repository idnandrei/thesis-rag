from __future__ import annotations

import json

from sqlalchemy import text

from videorag.db.session import db_session
from videorag.input.paths import video_paths
from videorag.input.select_video import pick_video_id

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
    video_id = pick_video_id("Select video to index chunks for")
    paths = video_paths(video_id)

    chunks_path = paths.chunks_path
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json has no 'chunks' list (or it's empty).")

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

    print(f"Indexed {len(rows)} chunks for video '{video_id}' into Postgres.")


if __name__ == "__main__":
    main()
