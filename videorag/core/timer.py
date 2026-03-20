from __future__ import annotations

import time
from contextlib import contextmanager
from typing import Any, Iterator


@contextmanager
def timed_block(label: str) -> Iterator[None]:
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed_seconds = time.perf_counter() - start
        print(f"[TIMER] {label}: {elapsed_seconds:.4f}s")


@contextmanager
def timed_logged_block(
    stage_name: str,
    logger: Any,
    **stage_start_values: Any,
) -> Iterator[None]:
    logger.stage_started(stage_name, **stage_start_values)

    start = time.perf_counter()
    try:
        yield
    except Exception as e:
        elapsed_seconds = time.perf_counter() - start
        print(f"[TIMER] {stage_name}: {elapsed_seconds:.4f}s")
        logger.stage_failed(stage_name, error=str(e))
        raise
    else:
        elapsed_seconds = time.perf_counter() - start
        print(f"[TIMER] {stage_name}: {elapsed_seconds:.4f}s")
