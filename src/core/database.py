"""
Database configuration for the RAG pipeline.
"""
import asyncio
from typing import AsyncGenerator, Optional

import chromadb
import redis.asyncio as redis
from sqlalchemy import create_engine, MetaData
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine, async_sessionmaker
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from src.core.config import get_settings
from src.core.logging import get_logger

logger = get_logger(__name__)
settings = get_settings()

client = chromadb.CloudClient(
    api_key=settings.CHROMA_API_KEY,
    tenant=settings.CHROMA_TENANT,
    database=settings.CHROMA_DATABASE
)

class Base(DeclarativeBase):
    """Base class for SQLAlchemy models."""
    pass


engine = None
async_engine = None
SessionLocal = None
AsyncSessionLocal = None


def init_db() -> None:
    """Initialize database connection."""
    global engine, async_engine, SessionLocal, AsyncSessionLocal

    engine = create_engine(
        settings.DATABASE_URL,
        echo=settings.DEBUG,
        pool_pre_ping=True
    )

    if settings.DATABASE_URL.startswith("sqlite"):
        async_url = settings.DATABASE_URL.replace("sqlite://", "sqlite+aiosqlite://")
    else:
        async_url = settings.DATABASE_URL

    async_engine = create_async_engine(
        async_url,
        echo=settings.DEBUG,
        pool_pre_ping=True
    )

    SessionLocal = sessionmaker(
        autocommit=False,
        autoflush=False,
        bind=engine
    )

    AsyncSessionLocal = async_sessionmaker(
        bind=async_engine,
        class_=AsyncSession,
        expire_on_commit=False
    )


def get_db() -> Session:
    """Get database session."""
    if SessionLocal is None:
        init_db()

    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


async def get_async_db() -> AsyncGenerator[AsyncSession, None]:
    """Get async database session."""
    if AsyncSessionLocal is None:
        init_db()

    async with AsyncSessionLocal() as session:
        yield session


_redis_client: Optional[redis.Redis] = None


async def get_redis() -> redis.Redis:
    """Get Redis client."""
    global _redis_client

    if _redis_client is None:
        _redis_client = redis.from_url(
            settings.REDIS_URL,
            db=settings.REDIS_DB,
            decode_responses=True,
            retry_on_timeout=True,
            health_check_interval=30
        )

    return _redis_client


async def close_redis() -> None:
    """Close Redis connection."""
    global _redis_client

    if _redis_client:
        await _redis_client.close()
        _redis_client = None


async def create_tables():
    """Create database tables."""
    if async_engine is None:
        init_db()

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)


async def drop_tables():
    """Drop database tables."""
    if async_engine is None:
        init_db()

    async with async_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


async def check_db_health() -> bool:
    """Check database health."""
    try:
        if async_engine is None:
            init_db()

        async with async_engine.begin() as conn:
            await conn.execute("SELECT 1")
        return True
    except Exception as e:
        logger.error(f"Database health check failed: {e}")
        return False


async def check_redis_health() -> bool:
    """Check Redis health."""
    try:
        redis_client = await get_redis()
        await redis_client.ping()
        return True
    except Exception as e:
        logger.error(f"Redis health check failed: {e}")
        return False


class DatabaseManager:
    """Context manager for database operations."""

    def __init__(self):
        self.session = None

    async def __aenter__(self):
        if AsyncSessionLocal is None:
            init_db()

        self.session = AsyncSessionLocal()
        return self.session

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            if exc_type is not None:
                await self.session.rollback()
            else:
                await self.session.commit()
            await self.session.close()