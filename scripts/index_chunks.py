from __future__ import annotations

import json
from pathlib import Path

import psycopg

from videorag.config.settings import get_settings


def main() -> None:
    settings = get_settings()

    video_id = input("Video ID: ").strip()

    chunks_path = Path(f"data/derived/{video_id}/chunks.json")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json has no 'chunks' list (or it's empty).")

    with psycopg.connect(
        host=settings.pg_host,
        port=settings.pg_port,
        dbname=settings.pg_db,
        user=settings.pg_user,
        password=settings.pg_password,
    ) as conn:
        with conn.cursor() as cur:
            inserted = 0
            for ch in chunks:
                chunk_id = int(ch["chunk_id"])
                ts_start = float(ch["ts_start"])
                ts_end = float(ch["ts_end"])
                text = str(ch["text"])

                cur.execute(
                    """
                    INSERT INTO chunks (video_id, chunk_id, ts_start, ts_end, text)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (video_id, chunk_id)
                    DO UPDATE SET
                      ts_start = EXCLUDED.ts_start,
                      ts_end   = EXCLUDED.ts_end,
                      text     = EXCLUDED.text
                    """,
                    (video_id, chunk_id, ts_start, ts_end, text),
                )
                inserted += 1

        conn.commit()

    print(f"Indexed {inserted} chunks for video '{video_id}' into pgvector db.")


if __name__ == "__main__":
    main()

