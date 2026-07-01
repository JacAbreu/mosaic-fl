"""engine.py — Fábrica de engine SQLAlchemy (PostgreSQL com pool, SQLite com StaticPool)."""
import sqlalchemy as sa


def _make_engine(url: str) -> sa.Engine:
    if url.startswith("postgresql"):
        return sa.create_engine(url, pool_pre_ping=True, pool_size=5, max_overflow=10)
    from sqlalchemy.pool import StaticPool
    return sa.create_engine(
        url,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
