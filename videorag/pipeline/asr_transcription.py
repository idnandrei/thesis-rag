from __future__ import annotations

import json
from pathlib import Path
from typing import Tuple

import mlx_whisper


def _clean_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def transcribe_video_to_derived(
    *,
    video_id: str,
    input_path: Path,
    derived_root: Path = Path("data/derived"),
    language: str = "en",
    model_name: str = "large-v2", 
) -> Tuple[Path, Path]:
    """
    Transcribe a single video/audio file and write two outputs:

    1) data/derived/<video_id>/transcript_segments.json
       Minimal schema for your RAG pipeline:
       {
         "video_id": "...",
         "segments": [{"start": 12.345, "end": 18.901, "text": "..."}, ...]
       }

    2) data/derived/<video_id>/transcript_raw.txt
       Plain text, one segment per line.
    """
    out_dir = derived_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    formatted_path = out_dir / "transcript_segments.json"
    raw_path = out_dir / "transcript_raw.txt"

    result = mlx_whisper.transcribe(
        str(input_path),
        language=language,
    )

    segments_out = []
    raw_lines = []

    for seg in result.get("segments", []):
        text = _clean_text(seg.get("text", ""))
        if not text:
            continue

        start = float(seg.get("start", 0.0))
        end = float(seg.get("end", 0.0))

        segments_out.append(
            {
                "start": round(start, 3),
                "end": round(end, 3),
                "text": text,
            }
        )
        raw_lines.append(text)

    formatted = {"video_id": video_id, "segments": segments_out}

    formatted_path.write_text(
        json.dumps(formatted, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    raw_path.write_text("\n".join(raw_lines) + "\n", encoding="utf-8")

    return formatted_path, raw_path