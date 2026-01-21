from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from videorag.config.settings import get_settings

_settings = get_settings()

engine = create_engine(
    _settings.sqlalchemy_url,
    future=True,
    pool_pre_ping=True,
)

SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)


@contextmanager
def db_session() -> Iterator[Session]:
    """
    Context-managed session:
      with db_session() as s:
          ...
    Commits on success, rollbacks on error.
    """
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
