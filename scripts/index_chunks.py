from __future__ import annotations

import json
from pathlib import Path

from sqlalchemy import text

from videorag.db.session import db_session

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
    video_id = input("Video ID (e.g., StanfordCS229): ").strip()

    chunks_path = Path(f"data/derived/{video_id}/chunks.json")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json has no 'chunks' list (or it's empty).")

    rows = []
    for ch in chunks:
        rows.append(
            {
                "video_id": video_id,
                "chunk_id": int(ch["chunk_id"]),
                "ts_start": float(ch["ts_start"]),
                "ts_end": float(ch["ts_end"]),
                "text": str(ch["text"]),
            }
        )

    with db_session() as session:
        session.execute(UPSERT_SQL, rows)

    print(f"Indexed {len(rows)} chunks for video '{video_id}' into Postgres.")


if __name__ == "__main__":
    main()

