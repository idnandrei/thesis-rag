from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
from ultralytics import YOLO

from videorag.config.settings import get_settings


def _clamp(v: int, lo: int, hi: int) -> int:
    return max(lo, min(hi, v))


def _crop_with_padding(
    img,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    pad_frac: float,
):
    h, w = img.shape[:2]
    bw = x2 - x1
    bh = y2 - y1

    pad_x = int(round(bw * pad_frac))
    pad_y = int(round(bh * pad_frac))

    x1p = _clamp(x1 - pad_x, 0, w - 1)
    y1p = _clamp(y1 - pad_y, 0, h - 1)
    x2p = _clamp(x2 + pad_x, 0, w - 1)
    y2p = _clamp(y2 + pad_y, 0, h - 1)

    return img[y1p:y2p, x1p:x2p], (x1p, y1p, x2p, y2p)


def _draw_detection(
    image,
    *,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    confidence: float,
) -> None:
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

    label = f"object {confidence:.2f}"
    bottom = y1 + 25 if y1 < 30 else y1 - 8

    cv2.putText(
        image,
        label,
        (x1 + 5, bottom),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )


def extract_visual_objects(
    *,
    video_id: str,
    frame_manifest_path: Path,
    model_path: Path | None = None,
    output_dir: Path | None = None,
    conf: float | None = None,
    iou: float | None = None,
    min_area_frac: float | None = None,
    pad_frac: float | None = None,
) -> Path:
    settings = get_settings()

    model_path = model_path or Path(settings.visual_extraction.model_path)
    output_dir = output_dir or (
        settings.paths.derived_dir / video_id / "visual_objects"
    )
    conf = conf if conf is not None else settings.visual_extraction.conf
    iou = iou if iou is not None else settings.visual_extraction.iou
    min_area_frac = (
        min_area_frac
        if min_area_frac is not None
        else settings.visual_extraction.min_area_frac
    )
    pad_frac = (
        pad_frac
        if pad_frac is not None
        else getattr(settings.visual_extraction, "pad_frac", 0.03)
    )

    if not model_path.exists():
        raise FileNotFoundError(f"Missing model file: {model_path}")

    if not frame_manifest_path.exists():
        raise FileNotFoundError(f"Missing frame manifest: {frame_manifest_path}")

    manifest_data = json.loads(frame_manifest_path.read_text(encoding="utf-8"))
    frames = manifest_data.get("frames", [])
    if not isinstance(frames, list) or not frames:
        raise ValueError("Frame manifest has no 'frames' list (or it is empty).")

    output_dir.mkdir(parents=True, exist_ok=True)
    annotated_dir = output_dir / "annotated_frames"
    crops_dir = output_dir / "crops"
    annotated_dir.mkdir(parents=True, exist_ok=True)
    crops_dir.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(model_path))

    output_manifest: dict[str, Any] = {
        "video_id": video_id,
        "model_path": str(model_path),
        "config": {
            "conf": conf,
            "iou": iou,
            "min_area_frac": min_area_frac,
            "pad_frac": pad_frac,
        },
        "frame_manifest_path": str(frame_manifest_path),
        "outputs_dir": str(output_dir),
        "frames": [],
    }

    total_detections = 0
    object_id = 0

    for frame in frames:
        frame_path = Path(str(frame["image_path"]))
        timestamp_sec = float(
            frame.get("actual_time_sec", frame.get("target_time_sec", 0.0))
        )
        frame_index = frame.get("frame_index")
        sample_index = frame.get("sample_index")

        img = cv2.imread(str(frame_path))
        if img is None:
            print(f"Skipping unreadable image: {frame_path}")
            continue

        annotated_img = img.copy()
        h, w = img.shape[:2]
        img_area = float(h * w)

        results = model.predict(
            source=img,
            conf=conf,
            iou=iou,
            verbose=False,
        )
        result = results[0]
        boxes = result.boxes

        detections = []

        if boxes is not None:
            for box in boxes:
                confidence = float(box.conf.item())
                class_id = int(box.cls.item()) if box.cls is not None else 0
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                area = max(0, x2 - x1) * max(0, y2 - y1)
                if img_area == 0 or area / img_area < min_area_frac:
                    continue

                crop_img, padded_bbox = _crop_with_padding(
                    img,
                    x1,
                    y1,
                    x2,
                    y2,
                    pad_frac,
                )

                crop_name = f"{frame_path.stem}__obj_{object_id:06d}.jpg"
                crop_path = crops_dir / crop_name
                cv2.imwrite(str(crop_path), crop_img)

                _draw_detection(
                    annotated_img,
                    x1=x1,
                    y1=y1,
                    x2=x2,
                    y2=y2,
                    confidence=confidence,
                )

                detections.append(
                    {
                        "object_id": object_id,
                        "class_id": class_id,
                        "confidence": round(confidence, 4),
                        "bbox_xyxy": [x1, y1, x2, y2],
                        "bbox_xyxy_padded": list(padded_bbox),
                        "crop_path": str(crop_path),
                    }
                )
                object_id += 1

        annotated_frame_path = annotated_dir / frame_path.name
        cv2.imwrite(str(annotated_frame_path), annotated_img)

        output_manifest["frames"].append(
            {
                "frame_path": str(frame_path),
                "annotated_frame_path": str(annotated_frame_path),
                "timestamp_sec": round(timestamp_sec, 3),
                "frame_index": frame_index,
                "sample_index": sample_index,
                "detections": detections,
            }
        )

        total_detections += len(detections)

    output_manifest["summary"] = {
        "frame_count": len(output_manifest["frames"]),
        "total_detections": total_detections,
    }

    output_manifest_path = output_dir / "visual_objects.json"
    output_manifest_path.write_text(
        json.dumps(output_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return output_manifest_path
