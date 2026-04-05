from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import cv2


def load_manifest(manifest_path: Path) -> dict[str, Any]:
    return json.loads(manifest_path.read_text(encoding="utf-8"))


def write_manifest(output_path: Path, payload: dict[str, Any]) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return output_path


def prepare_gray(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def prepare_color(image_path: Path):
    img = cv2.imread(str(image_path))
    if img is None:
        raise RuntimeError(f"Could not read image: {image_path}")
    return img


def copy_frame_to_kept_dir(
    frame: dict[str, Any], kept_frames_dir: Path
) -> dict[str, Any]:
    frame_copy = dict(frame)

    src = Path(str(frame_copy["image_path"]))
    dst = kept_frames_dir / src.name
    kept_frames_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)

    frame_copy["source_image_path"] = str(src)
    frame_copy["image_path"] = str(dst)

    return frame_copy
