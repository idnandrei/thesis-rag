from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

REGISTRY_PATH = Path("data/registry.json")


def load_registry(path: Path = REGISTRY_PATH) -> Dict[str, Any]:
    if not path.exists():
        return {"videos": {}}
    return json.loads(path.read_text(encoding="utf-8"))


def list_video_ids(path: Path = REGISTRY_PATH) -> List[str]:
    data = load_registry(path)
    videos = data.get("videos", {})
    return sorted(videos.keys()) if isinstance(videos, dict) else []
