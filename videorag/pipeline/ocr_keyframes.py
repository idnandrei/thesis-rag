from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import easyocr
import numpy as np

from videorag.config.settings import get_settings


def _load_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _to_python_jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _to_python_jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_python_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_to_python_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _resolve_easyocr_gpu_flag(use_gpu: bool) -> bool | str:
    if not use_gpu:
        return False

    try:
        import torch

        if torch.cuda.is_available():
            return True
        if torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass

    return False


def run_ocr_on_keyframes(
    *,
    keyframes_manifest_path: Path,
    output_dir: Path,
    languages: list[str] | None = None,
    min_confidence: float | None = None,
    use_gpu: bool | None = None,
) -> Path:
    settings = get_settings()

    languages = languages or ["en"]
    min_confidence = (
        settings.ocr.min_confidence if min_confidence is None else min_confidence
    )
    use_gpu = settings.ocr.use_gpu if use_gpu is None else use_gpu

    if not keyframes_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing keyframes manifest: {keyframes_manifest_path}"
        )

    manifest = _load_manifest(keyframes_manifest_path)
    frames = manifest.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("Keyframes manifest has no 'frames' list (or it is empty).")

    output_dir.mkdir(parents=True, exist_ok=True)

    easyocr_gpu = _resolve_easyocr_gpu_flag(use_gpu)

    print(f"[OCR] Languages: {languages}")
    print(f"[OCR] Minimum confidence: {min_confidence}")
    print(f"[OCR] EasyOCR GPU setting: {easyocr_gpu}")

    reader = easyocr.Reader(languages, gpu=easyocr_gpu)

    results: dict[str, Any] = {
        "keyframes_manifest_path": str(keyframes_manifest_path),
        "languages": languages,
        "min_confidence": min_confidence,
        "use_gpu": use_gpu,
        "frames": [],
        "summary": {},
    }

    text_lines: list[str] = []
    total_text_items = 0

    for idx, frame in enumerate(frames, start=1):
        image_path = Path(str(frame.get("image_path", "")))
        timestamp_sec = float(
            frame.get("actual_time_sec", frame.get("target_time_sec", 0.0))
        )
        frame_index = frame.get("frame_index")
        sample_index = frame.get("sample_index")

        if not image_path.exists():
            print(f"[OCR] Skipping missing image: {image_path}")
            continue

        print(f"[OCR] ({idx}/{len(frames)}) Processing {image_path.name} ...")

        raw_results = reader.readtext(str(image_path))

        ocr_items = []
        combined_text_parts = []

        for item in raw_results:
            if len(item) != 3:
                continue

            bbox, text, confidence = item
            text = str(text).strip()
            confidence = float(confidence)

            if not text:
                continue
            if confidence < min_confidence:
                continue

            combined_text_parts.append(text)
            ocr_items.append(
                {
                    "bbox": _to_python_jsonable(bbox),
                    "text": text,
                    "confidence": round(confidence, 4),
                }
            )

        combined_text = " ".join(combined_text_parts).strip()

        results["frames"].append(
            {
                "image_path": str(image_path),
                "timestamp_sec": round(timestamp_sec, 3),
                "frame_index": frame_index,
                "sample_index": sample_index,
                "combined_text": combined_text,
                "ocr_items": _to_python_jsonable(ocr_items),
            }
        )

        if combined_text:
            text_lines.append(f"[{round(timestamp_sec, 3)}] {combined_text}")

        total_text_items += len(ocr_items)

    results["summary"] = {
        "frame_count": len(results["frames"]),
        "frames_with_text": sum(1 for f in results["frames"] if f["combined_text"]),
        "total_text_items": total_text_items,
    }

    results_path = output_dir / "ocr_results.json"
    results_path.write_text(
        json.dumps(_to_python_jsonable(results), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    text_path = output_dir / "ocr_text.txt"
    text_path.write_text("\n".join(text_lines) + "\n", encoding="utf-8")

    return results_path
