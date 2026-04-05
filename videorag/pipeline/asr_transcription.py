from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Tuple, cast

import mlx_whisper

from videorag.config.settings import get_settings


def _clean_text(s: str) -> str:
    return " ".join((s or "").strip().split())


def transcribe_video_to_derived(
    *,
    video_id: str,
    input_path: Path,
    derived_root: Path | None = None,
    language: str | None = None,
    model_name: str | None = None,
) -> Tuple[Path, Path]:
    settings = get_settings()

    derived_root = derived_root or settings.paths.derived_dir
    language = language or settings.asr.language
    model_name = model_name or settings.asr.model_name

    out_dir = derived_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)

    formatted_path = out_dir / "transcript_segments.json"
    raw_path = out_dir / "transcript_raw.txt"

    print(f"[ASR] Loading model: {model_name}")
    print("[ASR] Backend: mlx_whisper")

    result = cast(
        dict[str, Any],
        mlx_whisper.transcribe(
            str(input_path),
            path_or_hf_repo=model_name,
            language=language,
        ),
    )

    segments = cast(list[dict[str, Any]], result.get("segments", []))

    segments_out = []
    raw_lines = []

    for seg in segments:
        text = _clean_text(str(seg.get("text", "")))
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
