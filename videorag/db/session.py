from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from pgvector.psycopg import register_vector
from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from videorag.config.settings import get_settings

_settings = get_settings()

engine = create_engine(
    _settings.sqlalchemy_url,
    future=True,
    pool_pre_ping=True,
)


@event.listens_for(engine, "connect")
def _on_connect(dbapi_conn, _conn_record) -> None:
    register_vector(dbapi_conn)


SessionLocal = sessionmaker(
    bind=engine,
    autoflush=False,
    autocommit=False,
    future=True,
)


@contextmanager
def db_session() -> Iterator[Session]:
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
