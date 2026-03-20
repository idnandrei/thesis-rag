from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4


def _format_dt(dt: datetime) -> str:
    return dt.strftime("%d %b %Y, %H:%M:%S")


class RunLogger:
    def __init__(self, run_name: str, log_dir: str | Path = "data/logs") -> None:
        self.run_name = run_name
        self.run_id = uuid4().hex[:8]

        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        now = datetime.now()
        timestamp = now.strftime("%Y%m%d_%H%M%S")
        self.log_path = self.log_dir / f"{self.run_name}_{timestamp}_{self.run_id}.json"

        self.started_at = now
        self.finished_at: datetime | None = None
        self.status = "running"

        self.context: dict[str, Any] = {}
        self.config: dict[str, dict[str, Any]] = {}
        self.stages: list[dict[str, Any]] = []
        self.summary: dict[str, Any] = {}

        self._active_stages: dict[str, dict[str, Any]] = {}

        self._save()

    def set_context(self, **values: Any) -> None:
        self.context.update(values)
        if values:
            print(f"[CONTEXT] {values}")
        else:
            print("[CONTEXT]")
        self._save()

    def config_group(self, group_name: str, config: Any) -> None:
        self.config[group_name] = self._config_to_dict(config)
        print(f"[CONFIG] {group_name} | {self.config[group_name]}")
        self._save()

    def stage_started(self, stage_name: str, **values: Any) -> None:
        stage = {
            "name": stage_name,
            "status": "running",
            "started_at_dt": datetime.now(),
            "finished_at_dt": None,
            "results": dict(values),
        }
        self._active_stages[stage_name] = stage

        if values:
            print(f"[STAGE START] {stage_name} | {values}")
        else:
            print(f"[STAGE START] {stage_name}")

        self._save()

    def stage_finished(self, stage_name: str, **results: Any) -> None:
        stage = self._active_stages.get(stage_name)
        if stage is None:
            raise ValueError(f"Stage '{stage_name}' is not active.")

        stage["status"] = "success"
        stage["finished_at_dt"] = datetime.now()
        stage["results"].update(results)

        self.stages.append(self._serialize_stage(stage))
        del self._active_stages[stage_name]

        if results:
            print(f"[STAGE END] {stage_name} | {results}")
        else:
            print(f"[STAGE END] {stage_name}")

        self._save()

    def stage_failed(self, stage_name: str, **results: Any) -> None:
        stage = self._active_stages.get(stage_name)

        if stage is None:
            stage = {
                "name": stage_name,
                "status": "failed",
                "started_at_dt": None,
                "finished_at_dt": datetime.now(),
                "results": dict(results),
            }
        else:
            stage["status"] = "failed"
            stage["finished_at_dt"] = datetime.now()
            stage["results"].update(results)
            del self._active_stages[stage_name]

        self.stages.append(self._serialize_stage(stage))
        self.status = "failed"

        if results:
            print(f"[STAGE FAIL] {stage_name} | {results}")
        else:
            print(f"[STAGE FAIL] {stage_name}")

        self._save()

    def set_summary(self, **values: Any) -> None:
        self.summary.update(values)
        if values:
            print(f"[SUMMARY] {values}")
        else:
            print("[SUMMARY]")
        self._save()

    def finish(self, status: str = "success", **summary_values: Any) -> None:
        self.finished_at = datetime.now()

        if self.status != "failed":
            self.status = status

        if summary_values:
            self.summary.update(summary_values)

        if summary_values:
            print(f"[RUN END] status={self.status} | {summary_values}")
        else:
            print(f"[RUN END] status={self.status}")

        self._save()

    def _config_to_dict(self, config: Any) -> dict[str, Any]:
        if hasattr(config, "to_log_dict") and callable(config.to_log_dict):
            return dict(config.to_log_dict())

        if is_dataclass(config) and not isinstance(config, type):
            return dict(asdict(config))

        if isinstance(config, dict):
            return dict(config)

        raise TypeError(
            "config must be a dict, a dataclass instance, or have a to_log_dict() method"
        )

    def _serialize_stage(self, stage: dict[str, Any]) -> dict[str, Any]:
        started_at_dt = stage.get("started_at_dt")
        finished_at_dt = stage.get("finished_at_dt")

        duration_seconds = None
        if started_at_dt is not None and finished_at_dt is not None:
            duration_seconds = round(
                (finished_at_dt - started_at_dt).total_seconds(), 4
            )

        return {
            "name": stage["name"],
            "status": stage["status"],
            "started_at": _format_dt(started_at_dt) if started_at_dt else None,
            "finished_at": _format_dt(finished_at_dt) if finished_at_dt else None,
            "duration_seconds": duration_seconds,
            "results": stage.get("results", {}),
        }

    def _build_payload(self) -> dict[str, Any]:
        return {
            "run_name": self.run_name,
            "run_id": self.run_id,
            "started_at": _format_dt(self.started_at),
            "finished_at": _format_dt(self.finished_at) if self.finished_at else None,
            "status": self.status,
            "context": self.context,
            "config": self.config,
            "stages": self.stages,
            "summary": self.summary,
        }

    def _save(self) -> None:
        payload = self._build_payload()
        with self.log_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
            f.write("\n")
