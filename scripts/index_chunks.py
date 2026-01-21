from __future__ import annotations

import json
import os
from pathlib import Path

from dotenv import load_dotenv
import psycopg

load_dotenv()  # ← THIS loads .env into os.environ


def _env(name: str, default: str | None = None) -> str | None:
    return os.getenv(name, default)


def main() -> None:
    host = _env("PGHOST", "localhost")
    port = int(_env("PGPORT", "5432"))
    dbname = _env("PGDATABASE", "videorag")
    user = _env("PGUSER", "postgres")
    password = _env("PGPASSWORD")

    print("🔎 Env check:")
    print(f"PGHOST={host!r}")
    print(f"PGPORT={port!r}")
    print(f"PGDATABASE={dbname!r}")
    print(f"PGUSER={user!r}")
    print(f"PGPASSWORD={'<set>' if password else '<NOT SET>'}")
    print()

    video_id = _env("VIDEO_ID", "").strip()
    if not video_id:
        video_id = input("Video ID (e.g., StanfordCS229): ").strip()

    chunks_path = Path(f"data/derived/{video_id}/chunks.json")
    if not chunks_path.exists():
        raise FileNotFoundError(f"Missing chunks file: {chunks_path}")

    payload = json.loads(chunks_path.read_text(encoding="utf-8"))
    chunks = payload.get("chunks", [])
    if not isinstance(chunks, list) or not chunks:
        raise ValueError("chunks.json has no 'chunks' list (or it's empty).")

    conn_str = f"host={host} port={port} dbname={dbname} user={user} password={password}"

    # ---- Insert (upsert) ----
    with psycopg.connect(conn_str) as conn:
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

    print(f"✅ Indexed {inserted} chunks for video '{video_id}' into Postgres.")


if __name__ == "__main__":
    main()