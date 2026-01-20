import logging
import time
from contextlib import contextmanager
from typing import Optional, Type


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """
    Configure app-wide logging once and return a named logger you can reuse
    across modules (via logging.getLogger("rag")).
    """
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    return logging.getLogger("rag")


@contextmanager
def process_segment(logger: logging.Logger, name: str):
    """
    Context manager that logs START/END/FAIL with elapsed time.
    Works for any part of the app (not just transcription).
    """
    t0 = time.perf_counter()
    logger.info("START: %s", name)
    try:
        yield
    except Exception:
        dt = time.perf_counter() - t0
        # exception() automatically includes traceback info
        logger.exception("FAIL:  %s (%.2fs)", name, dt)
        raise
    else:
        dt = time.perf_counter() - t0
        logger.info("END:   %s (%.2fs)", name, dt)
