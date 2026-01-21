from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional


def _clean_text(s: str) -> str:
    return " ".join((s or "").strip().split())


@dataclass(frozen=True)
class EventBuilderConfig:
    # If true, include an "event_id" (index) field on each event for traceability
    include_event_id: bool = True


def build_events_from_asr(
    *,
    video_id: str,
    transcript_segments_path: Path,
    derived_root: Path = Path("data/derived"),
    cfg: EventBuilderConfig = EventBuilderConfig(),
) -> Path:
    """
    Build a unified events.json from ASR transcript segments (MVP).

    Input:  data/derived/<video_id>/transcript_segments.json
      {
        "video_id": "...",
        "segments": [{"start": 12.345, "end": 18.901, "text": "..."}]
      }

    Output: data/derived/<video_id>/events.json
      {
        "video_id": "...",
        "events": [
          {"event_id": 0, "type": "asr", "t_start": 12.345, "t_end": 18.901, "text": "...", "source": "asr"}
        ]
      }
    """
    data = json.loads(transcript_segments_path.read_text(encoding="utf-8"))

    file_video_id = str(data.get("video_id", "")).strip()
    if not video_id:
        video_id = file_video_id or transcript_segments_path.parent.name

    segments = data.get("segments", [])
    if not isinstance(segments, list):
        raise ValueError("Invalid transcript file: 'segments' must be a list.")

    events: List[Dict[str, Any]] = []
    for seg in segments:
        text = _clean_text(str(seg.get("text", "")))
        if not text:
            continue

        t_start = float(seg.get("start", 0.0))
        t_end = float(seg.get("end", t_start))

        events.append(
            {
                "type": "asr",
                "t_start": round(t_start, 3),
                "t_end": round(t_end, 3),
                "text": text,
                "source": "asr",
            }
        )

    # Ensure stable time order
    events.sort(key=lambda e: (e["t_start"], e["t_end"]))

    if cfg.include_event_id:
        for idx, e in enumerate(events):
            e["event_id"] = idx

    out_dir = derived_root / video_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "events.json"

    payload = {"video_id": video_id, "events": events}
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return out_path