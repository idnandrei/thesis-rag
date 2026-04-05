from __future__ import annotations

import gc
import json
import re
import time
from pathlib import Path
from typing import Any, cast

import torch
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration


def _load_visual_objects_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _build_crop_lookup(visual_objects_manifest: dict) -> dict[str, dict]:
    crop_lookup: dict[str, dict] = {}

    frames = visual_objects_manifest.get("frames", [])
    if not isinstance(frames, list):
        return crop_lookup

    for frame in frames:
        frame_path = frame.get("frame_path")
        annotated_frame_path = frame.get("annotated_frame_path")
        timestamp_sec = frame.get("timestamp_sec")
        detections = frame.get("detections", [])

        if not isinstance(detections, list):
            continue

        for det in detections:
            crop_path = det.get("crop_path")
            if not crop_path:
                continue

            crop_name = Path(str(crop_path)).name
            crop_lookup[crop_name] = {
                "crop_path": str(crop_path),
                "source_frame_path": frame_path,
                "source_annotated_frame_path": annotated_frame_path,
                "timestamp_sec": timestamp_sec,
                "object_id": det.get("object_id"),
            }

    return crop_lookup


def _extract_frame_stem_from_crop_name(crop_name: str) -> str | None:
    match = re.match(
        r"^(frame_\d+)__obj_\d+\.(jpg|jpeg|png)$",
        crop_name,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1)


def _get_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _clear_torch_backend_state(device: str) -> None:
    gc.collect()

    if device == "cuda" and torch.cuda.is_available():
        torch.cuda.empty_cache()

    if device == "mps" and torch.backends.mps.is_available():
        if hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()
        if hasattr(torch.mps, "synchronize"):
            torch.mps.synchronize()


def _load_model_and_processor(
    model_name: str,
) -> tuple[Qwen3VLForConditionalGeneration, Any, str]:
    device = _get_device()

    print(f"[IMAGE CAPTION] Loading model: {model_name}")
    print(f"[IMAGE CAPTION] Using device: {device}")

    load_start = time.perf_counter()

    model = Qwen3VLForConditionalGeneration.from_pretrained(
        model_name,
        dtype="auto",
    )
    model = model.to(device)
    model.eval()

    processor = cast(Any, AutoProcessor.from_pretrained(model_name))

    load_duration = time.perf_counter() - load_start
    print(f"[IMAGE CAPTION] Model + processor loaded in {load_duration:.2f}s")

    return model, processor, device


def _caption_image(
    *,
    model: Qwen3VLForConditionalGeneration,
    processor: Any,
    image_path: Path,
    device: str,
    max_new_tokens: int,
    do_sample: bool,
) -> tuple[str, int]:
    with Image.open(image_path) as raw_image:
        image = raw_image.convert("RGB")

    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "image": image,
                },
                {
                    "type": "text",
                    "text": (
                        "Describe the main idea of this cropped lecture visual in a concise, grounded, and non-speculative way. "
                        "Focus on what kind of visual it is and the main information it conveys, rather than listing every small detail. "
                        "Mention only the most important visible elements needed to understand the visual at a high level. "
                        "If it is a chart, table, diagram, formula, interface, figure, or other structured visual, summarize its overall content and the main relationships, patterns, or variables it shows. "
                        "Do not guess missing context. "
                        "If some text or detail is unclear, say that it is unclear or partially unreadable instead of guessing."
                    ),
                },
            ],
        }
    ]

    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )
    inputs = inputs.to(device)

    with torch.no_grad():
        generated_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
        )

    generated_ids_trimmed = [
        out_ids[len(in_ids) :]
        for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
    ]

    output_text = processor.batch_decode(
        generated_ids_trimmed,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )

    generated_token_count = 0
    if generated_ids_trimmed:
        generated_token_count = int(generated_ids_trimmed[0].shape[0])

    del inputs
    del generated_ids
    del generated_ids_trimmed
    _clear_torch_backend_state(device)

    return output_text[0].strip(), generated_token_count


def run_image_caption(
    *,
    video_id: str,
    visual_objects_manifest_path: Path,
    caption_source_dir: Path,
    output_dir: Path,
    model_name: str,
    max_new_tokens: int,
    do_sample: bool,
) -> Path:
    if not visual_objects_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing visual objects manifest: {visual_objects_manifest_path}"
        )

    if not caption_source_dir.exists():
        raise FileNotFoundError(f"Missing caption source dir: {caption_source_dir}")

    crop_paths = sorted(
        p
        for p in caption_source_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not crop_paths:
        raise ValueError(f"No crop images found in: {caption_source_dir}")

    print(f"[IMAGE CAPTION] Found {len(crop_paths)} kept crops in {caption_source_dir}")

    visual_objects_manifest = _load_visual_objects_manifest(
        visual_objects_manifest_path
    )
    crop_lookup = _build_crop_lookup(visual_objects_manifest)

    output_dir.mkdir(parents=True, exist_ok=True)

    model: Qwen3VLForConditionalGeneration | None = None
    processor: Any | None = None
    device: str | None = None

    try:
        model, processor, device = _load_model_and_processor(model_name)

        results: dict[str, Any] = {
            "video_id": video_id,
            "model_name": model_name,
            "device": device,
            "caption_source_dir": str(caption_source_dir),
            "generation": {
                "max_new_tokens": max_new_tokens,
                "do_sample": do_sample,
            },
            "items": [],
            "summary": {},
        }

        plain_text_lines: list[str] = []

        total_start = time.perf_counter()
        total_generated_tokens = 0

        for idx, crop_path in enumerate(crop_paths, start=1):
            crop_name = crop_path.name
            metadata = crop_lookup.get(crop_name, {})
            frame_stem = _extract_frame_stem_from_crop_name(crop_name)

            print(
                f"[IMAGE CAPTION] ({idx}/{len(crop_paths)}) Processing {crop_name} ..."
            )
            item_start = time.perf_counter()

            caption, generated_tokens = _caption_image(
                model=model,
                processor=processor,
                image_path=crop_path,
                device=device,
                max_new_tokens=max_new_tokens,
                do_sample=do_sample,
            )

            item_duration = time.perf_counter() - item_start
            total_generated_tokens += generated_tokens

            print(
                f"[IMAGE CAPTION] ({idx}/{len(crop_paths)}) Done in {item_duration:.2f}s | generated_tokens={generated_tokens}"
            )

            item = {
                "crop_name": crop_name,
                "crop_path": str(crop_path),
                "frame_stem": frame_stem,
                "source_frame_path": metadata.get("source_frame_path"),
                "source_annotated_frame_path": metadata.get(
                    "source_annotated_frame_path"
                ),
                "timestamp_sec": metadata.get("timestamp_sec"),
                "object_id": metadata.get("object_id"),
                "generated_tokens": generated_tokens,
                "caption": caption,
            }
            results["items"].append(item)

            ts = metadata.get("timestamp_sec")
            plain_text_lines.append(f"[{ts}] {crop_name}: {caption}")

        total_duration = time.perf_counter() - total_start

        results["summary"] = {
            "captioned_items": len(results["items"]),
            "total_caption_time_sec": round(total_duration, 2),
            "total_generated_tokens": total_generated_tokens,
        }

        results_path = output_dir / "captions.json"
        results_path.write_text(
            json.dumps(results, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

        text_path = output_dir / "captions.txt"
        text_path.write_text("\n".join(plain_text_lines) + "\n", encoding="utf-8")

        print(
            f"[IMAGE CAPTION] Finished {len(results['items'])} items in {total_duration:.2f}s | total_generated_tokens={total_generated_tokens}"
        )
        return results_path

    finally:
        if model is not None:
            del model
        if processor is not None:
            del processor
        if device is not None:
            _clear_torch_backend_state(device)
