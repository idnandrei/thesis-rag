from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import tiktoken


# -----------------------------
# Config
# -----------------------------
@dataclass(frozen=True)
class ChunkingConfig:
    chunk_tokens: int = 512
    overlap_tokens: int = 100
    max_tokens: int = 640   # safety cap


def default_chunking_config() -> ChunkingConfig:
    return ChunkingConfig()


# -----------------------------
# IO
# -----------------------------
def _load_events(events_path: Path) -> Tuple[str, List[Dict[str, Any]]]:
    data = json.loads(events_path.read_text(encoding="utf-8"))

    video_id = str(data.get("video_id", "")).strip()
    events = data.get("events", [])

    cleaned: List[Dict[str, Any]] = []
    for idx, e in enumerate(events):
        text = (e.get("text") or "").strip()
        if not text:
            continue

        t_start = float(e.get("t_start", e.get("t", 0.0)))
        t_end = float(e.get("t_end", e.get("t", t_start)))

        cleaned.append(
            {
                "event_id": idx,
                "type": str(e.get("type", "asr")),
                "t_start": t_start,
                "t_end": t_end,
                "text": text,
            }
        )

    cleaned.sort(key=lambda x: (x["t_start"], x["t_end"]))
    return video_id, cleaned


# -----------------------------
# Chunking
# -----------------------------
def chunk_events_to_file(
    *,
    video_id: str,
    events_path: Path,
    derived_root: Path = Path("data/derived"),
    cfg: ChunkingConfig | None = None,
) -> Path:

    if cfg is None:
        cfg = default_chunking_config()

    file_video_id, events = _load_events(events_path)
    if not video_id:
        video_id = file_video_id or events_path.parent.name

    out_dir = derived_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "chunks.json"

    enc = tiktoken.get_encoding("cl100k_base")

    # Precompute token counts per event
    event_tokens = [len(enc.encode(e["text"])) for e in events]

    chunks: List[Dict[str, Any]] = []

    i = 0
    chunk_id = 0

    while i < len(events):
        start_i = i
        token_count = 0
        end_i = i - 1

        while i < len(events):
            next_tokens = token_count + event_tokens[i]

            # Always allow first event
            if i == start_i:
                token_count = next_tokens
                end_i = i
                i += 1
                continue

            if next_tokens <= cfg.chunk_tokens:
                token_count = next_tokens
                end_i = i
                i += 1
                continue

            # ----- closest-to-boundary decision -----
            diff_stop = abs(cfg.chunk_tokens - token_count)
            diff_add = abs(cfg.chunk_tokens - next_tokens)

            if diff_add <= diff_stop and next_tokens <= cfg.max_tokens:
                token_count = next_tokens
                end_i = i
                i += 1

            break

        # Build chunk
        texts = [events[k]["text"] for k in range(start_i, end_i + 1)]
        chunk_text = " ".join(texts)

        ts_start = min(events[k]["t_start"] for k in range(start_i, end_i + 1))
        ts_end = max(events[k]["t_end"] for k in range(start_i, end_i + 1))

        counts: Dict[str, int] = {}
        for k in range(start_i, end_i + 1):
            t = events[k]["type"]
            counts[t] = counts.get(t, 0) + 1

        chunks.append(
            {
                "chunk_id": chunk_id,
                "ts_start": round(ts_start, 3),
                "ts_end": round(ts_end, 3),
                "text": chunk_text,
                "event_range": [start_i, end_i],
                "event_counts": counts,
                "token_count": token_count,
            }
        )
        chunk_id += 1

        # -----------------------------
        # Token-based overlap (event-aligned)
        # -----------------------------
        overlap_budget = cfg.overlap_tokens
        back_tokens = 0
        j = end_i

        while j >= start_i and back_tokens < overlap_budget:
            back_tokens += event_tokens[j]
            j -= 1

        i = max(j + 1, start_i + 1)

    payload = {
        "video_id": video_id,
        "config": {
            "chunk_tokens": cfg.chunk_tokens,
            "overlap_tokens": cfg.overlap_tokens,
            "max_tokens": cfg.max_tokens,
            "tokenizer": "tiktoken::cl100k_base",
        },
        "chunks": chunks,
    }

    out_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return out_path