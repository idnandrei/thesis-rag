from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from videorag.config.settings import get_settings


def _load_registry(registry_path: Path) -> dict:
    if registry_path.exists():
        return json.loads(registry_path.read_text(encoding="utf-8"))
    return {"videos": {}}


def _save_registry(registry_path: Path, registry: dict) -> None:
    registry_path.parent.mkdir(parents=True, exist_ok=True)
    registry_path.write_text(
        json.dumps(registry, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize_slug(text: str) -> str:
    text = text.strip().lower()
    text = re.sub(r"[^a-z0-9]+", "-", text)
    text = re.sub(r"-+", "-", text).strip("-")
    return text


def _extract_youtube_id(url: str) -> str:
    parsed = urlparse(url)

    if parsed.netloc in {"youtu.be", "www.youtu.be"}:
        video_id = parsed.path.lstrip("/")
        if video_id:
            return video_id

    if "youtube.com" in parsed.netloc:
        qs = parse_qs(parsed.query)
        if "v" in qs and qs["v"]:
            return qs["v"][0]

        parts = [p for p in parsed.path.split("/") if p]
        if "shorts" in parts:
            idx = parts.index("shorts")
            if idx + 1 < len(parts):
                return parts[idx + 1]

    raise ValueError(f"Could not extract YouTube ID from URL: {url}")


def _find_group_dir(dataset_root: Path, description_slug: str) -> Path:
    matches = []

    for child in sorted(dataset_root.iterdir()):
        if not child.is_dir():
            continue

        folder_slug = re.sub(r"^\d+-", "", child.name)
        if folder_slug == description_slug:
            matches.append(child)

    if not matches:
        raise FileNotFoundError(
            f"Could not find local dataset folder for group '{description_slug}'"
        )

    if len(matches) > 1:
        raise ValueError(
            f"Multiple local dataset folders matched '{description_slug}': "
            f"{[m.name for m in matches]}"
        )

    return matches[0]


def _find_local_video_file(videos_dir: Path, youtube_id: str) -> Path:
    candidates = [
        p for p in videos_dir.iterdir() if p.is_file() and p.stem == youtube_id
    ]

    if not candidates:
        raise FileNotFoundError(
            f"Could not find local video file with stem '{youtube_id}' in {videos_dir}"
        )

    if len(candidates) > 1:
        raise ValueError(
            f"Multiple local files matched '{youtube_id}' in {videos_dir}: "
            f"{[c.name for c in candidates]}"
        )

    return candidates[0]


def main() -> None:
    settings = get_settings()

    dataset_root = Path("dataset/longervideos")
    lecture_dataset_path = dataset_root / "lecture_dataset.json"
    registry_path = settings.paths.registry_path

    if not dataset_root.exists():
        raise FileNotFoundError(f"Missing dataset root: {dataset_root}")
    if not lecture_dataset_path.exists():
        raise FileNotFoundError(f"Missing lecture dataset file: {lecture_dataset_path}")

    lecture_dataset = _load_json(lecture_dataset_path)

    registry = _load_registry(registry_path)
    registry.setdefault("videos", {})

    newly_registered_count = 0
    updated_registered_count = 0
    total_kept_videos = 0

    for _, entry in lecture_dataset.items():
        if not isinstance(entry, dict):
            continue

        if entry.get("type") != "lecture":
            continue

        description = str(entry.get("description", "")).strip()
        if not description:
            raise ValueError("Lecture dataset entry is missing 'description'.")

        description_slug = _normalize_slug(description)
        video_urls = entry.get("video_url", [])
        if not isinstance(video_urls, list) or not video_urls:
            continue

        group_dir = _find_group_dir(dataset_root, description_slug)
        videos_dir = group_dir / "videos"
        if not videos_dir.exists():
            raise FileNotFoundError(f"Missing videos dir: {videos_dir}")

        print(f"\n[GROUP] {description_slug}")
        print(f"[GROUP] Kept videos in trimmed dataset: {len(video_urls)}")

        for idx, video_url in enumerate(video_urls, start=1):
            youtube_id = _extract_youtube_id(video_url)
            input_path = _find_local_video_file(videos_dir, youtube_id)

            video_id = f"{description_slug}_{idx:03d}"
            derived_dir = settings.paths.derived_dir / video_id
            derived_dir.mkdir(parents=True, exist_ok=True)

            existing_entry = registry["videos"].get(video_id)

            base_entry = {
                "title": video_id,
                "group_name": description_slug,
                "group_description": description,
                "group_index": idx,
                "source_youtube_id": youtube_id,
                "source_video_url": video_url,
                "source_local_filename": input_path.name,
                "raw_path": str(input_path),
            }

            if existing_entry is None:
                registry["videos"][video_id] = {
                    **base_entry,
                    "added_at": datetime.now(timezone.utc).isoformat(),
                }
                newly_registered_count += 1
            else:
                registry["videos"][video_id] = {
                    **existing_entry,
                    **base_entry,
                    "added_at": existing_entry.get(
                        "added_at", datetime.now(timezone.utc).isoformat()
                    ),
                }
                updated_registered_count += 1

            total_kept_videos += 1
            print(f"Registered '{video_id}' -> {input_path}")

    _save_registry(registry_path, registry)

    print()
    print(f"Done. Total kept lecture videos registered: {total_kept_videos}")
    print(f"New registry entries: {newly_registered_count}")
    print(f"Existing registry entries preserved/updated: {updated_registered_count}")
    print(f"Registry updated: {registry_path}")


if __name__ == "__main__":
    main()
