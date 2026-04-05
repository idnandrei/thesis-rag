from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO


def _load_visual_objects_manifest(manifest_path: Path) -> dict:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def _build_frame_lookup(manifest: dict) -> dict[str, dict]:
    frames = manifest.get("frames", [])
    lookup: dict[str, dict] = {}

    if isinstance(frames, list):
        for frame in frames:
            frame_path = str(frame.get("frame_path", ""))
            if not frame_path:
                continue
            frame_stem = Path(frame_path).stem
            lookup[frame_stem] = frame

    return lookup


def _extract_frame_stem_from_crop_name(crop_name: str) -> str | None:
    match = re.match(
        r"^(frame_\d+)__obj_\d+\.(jpg|jpeg|png)$",
        crop_name,
        re.IGNORECASE,
    )
    if not match:
        return None
    return match.group(1)


def _find_person_class_ids(model: YOLO) -> set[int]:
    names = model.names
    person_ids: set[int] = set()

    if isinstance(names, dict):
        for cls_id, cls_name in names.items():
            if str(cls_name).strip().lower() == "person":
                person_ids.add(int(cls_id))
    elif isinstance(names, list):
        for cls_id, cls_name in enumerate(names):
            if str(cls_name).strip().lower() == "person":
                person_ids.add(int(cls_id))

    if not person_ids:
        raise RuntimeError("Could not find 'person' class in model.names.")

    return person_ids


def run_person_filter(
    *,
    video_id: str,
    visual_objects_manifest_path: Path,
    crops_dir: Path,
    output_dir: Path,
    model_name: str,
    person_conf_threshold: float,
    person_box_ratio_threshold: float,
) -> Path:
    if not visual_objects_manifest_path.exists():
        raise FileNotFoundError(
            f"Missing visual objects manifest: {visual_objects_manifest_path}"
        )
    if not crops_dir.exists():
        raise FileNotFoundError(f"Missing crops directory: {crops_dir}")

    manifest = _load_visual_objects_manifest(visual_objects_manifest_path)
    frame_lookup = _build_frame_lookup(manifest)

    crop_paths = sorted(
        p for p in crops_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    if not crop_paths:
        raise ValueError(f"No crop images found in: {crops_dir}")

    model = YOLO(model_name)
    person_class_ids = _find_person_class_ids(model)

    kept_crops_dir = output_dir / "kept_crops"
    removed_crops_dir = output_dir / "removed_crops"
    kept_crops_dir.mkdir(parents=True, exist_ok=True)
    removed_crops_dir.mkdir(parents=True, exist_ok=True)

    results: dict[str, Any] = {
        "video_id": video_id,
        "model_name": model_name,
        "person_conf_threshold": person_conf_threshold,
        "person_box_ratio_threshold": person_box_ratio_threshold,
        "person_class_ids": sorted(person_class_ids),
        "crops": [],
        "frames": [],
        "summary": {},
    }

    frame_crop_stats: dict[str, dict[str, int]] = defaultdict(
        lambda: {"total": 0, "kept": 0, "removed": 0}
    )

    kept_crop_count = 0
    removed_crop_count = 0

    for crop_path in crop_paths:
        frame_stem = _extract_frame_stem_from_crop_name(crop_path.name)
        source_frame_info = frame_lookup.get(frame_stem or "")

        object_image = cv2.imread(str(crop_path))
        if object_image is None:
            continue

        image_h, image_w = object_image.shape[:2]
        image_area = float(image_h * image_w) if image_h > 0 and image_w > 0 else 0.0

        prediction = model.predict(
            source=str(crop_path),
            conf=person_conf_threshold,
            verbose=False,
        )[0]

        person_detections = []
        total_person_box_area = 0.0
        total_person_box_ratio = 0.0

        boxes = prediction.boxes
        if boxes is not None and image_area > 0:
            for box in boxes:
                cls_id = int(box.cls.item()) if box.cls is not None else -1
                conf = float(box.conf.item()) if box.conf is not None else 0.0

                if cls_id not in person_class_ids or conf < person_conf_threshold:
                    continue

                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                person_box_area = max(0, x2 - x1) * max(0, y2 - y1)
                person_box_ratio = person_box_area / image_area

                total_person_box_area += person_box_area

                person_detections.append(
                    {
                        "class_id": cls_id,
                        "confidence": round(conf, 4),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_box_ratio": round(person_box_ratio, 4),
                    }
                )

        if image_area > 0:
            total_person_box_ratio = total_person_box_area / image_area
        else:
            total_person_box_ratio = 0.0

        is_person_dominant = (
            len(person_detections) > 0
            and total_person_box_ratio >= person_box_ratio_threshold
        )

        if is_person_dominant:
            dst_crop = removed_crops_dir / crop_path.name
            removed_crop_count += 1
            if frame_stem:
                frame_crop_stats[frame_stem]["removed"] += 1
        else:
            dst_crop = kept_crops_dir / crop_path.name
            kept_crop_count += 1
            if frame_stem:
                frame_crop_stats[frame_stem]["kept"] += 1

        if frame_stem:
            frame_crop_stats[frame_stem]["total"] += 1

        shutil.copy2(crop_path, dst_crop)

        results["crops"].append(
            {
                "crop_path": str(crop_path),
                "crop_name": crop_path.name,
                "source_frame_stem": frame_stem,
                "source_frame_path": (
                    source_frame_info.get("frame_path") if source_frame_info else None
                ),
                "source_annotated_frame_path": (
                    source_frame_info.get("annotated_frame_path")
                    if source_frame_info
                    else None
                ),
                "timestamp_sec": (
                    source_frame_info.get("timestamp_sec")
                    if source_frame_info
                    else None
                ),
                "person_detections": person_detections,
                "total_person_box_ratio": round(total_person_box_ratio, 4),
                "is_person_dominant": is_person_dominant,
                "copied_to": str(dst_crop),
            }
        )

    for frame_stem, stats in sorted(frame_crop_stats.items()):
        frame_info = frame_lookup.get(frame_stem)
        results["frames"].append(
            {
                "frame_stem": frame_stem,
                "source_frame_path": (
                    frame_info.get("frame_path") if frame_info else None
                ),
                "source_annotated_frame_path": (
                    frame_info.get("annotated_frame_path") if frame_info else None
                ),
                "timestamp_sec": (
                    frame_info.get("timestamp_sec") if frame_info else None
                ),
                "total_crops": stats["total"],
                "kept_crops": stats["kept"],
                "removed_crops": stats["removed"],
                "frame_is_kept": stats["kept"] > 0,
            }
        )

    kept_frame_count = sum(1 for frame in results["frames"] if frame["frame_is_kept"])
    removed_frame_count = sum(
        1 for frame in results["frames"] if not frame["frame_is_kept"]
    )

    results["summary"] = {
        "total_crops": len(results["crops"]),
        "kept_crops": kept_crop_count,
        "removed_crops": removed_crop_count,
        "kept_frames": kept_frame_count,
        "removed_frames": removed_frame_count,
        "kept_crops_dir": str(kept_crops_dir),
        "removed_crops_dir": str(removed_crops_dir),
    }

    results_path = output_dir / "filter_results.json"
    results_path.write_text(
        json.dumps(results, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return results_path
