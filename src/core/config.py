"""
Core configuration for the RAG pipeline.
"""
import os
from functools import lru_cache
from typing import Optional, List

from dotenv import load_dotenv
from pydantic import BaseSettings, Field

load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    ENVIRONMENT: str = Field(..., env="ENVIRONMENT")
    DEBUG: bool = Field(..., env="DEBUG")
    API_HOST: str = Field(..., env="API_HOST")
    API_PORT: int = Field(..., env="API_PORT")

    LLM_PROVIDER: str = Field(..., env="LLM_PROVIDER")
    HF_MODEL_NAME: str = Field(..., env="HF_MODEL_NAME")
    LLM_MODEL_PATH: str = Field(..., env="LLM_MODEL_PATH")
    LLM_MAX_CONTEXT: int = Field(..., env="LLM_MAX_CONTEXT")
    LLM_GPU_LAYERS: int = Field(..., env="LLM_GPU_LAYERS")
    LLM_THREADS: int = Field(..., env="LLM_THREADS")
    LLM_VERBOSE: bool = Field(..., env="LLM_VERBOSE")
    LLM_TEMPERATURE: float = Field(..., env="LLM_TEMPERATURE")
    LLM_MAX_TOKENS: int = Field(..., env="LLM_MAX_TOKENS")

    EMBEDDING_MODEL: str = Field(..., env="EMBEDDING_MODEL")
    EMBEDDING_DIMENSION: int = Field(..., env="EMBEDDING_DIMENSION")

    VECTOR_STORE_TYPE: str = Field(..., env="VECTOR_STORE_TYPE")
    CHROMA_PERSIST_DIR: str = Field(..., env="CHROMA_PERSIST_DIR")
    CHROMA_COLLECTION_NAME: str = Field(..., env="CHROMA_COLLECTION_NAME")

    REDIS_URL: str = Field(..., env="REDIS_URL")
    REDIS_DB: int = Field(..., env="REDIS_DB")

    DATABASE_URL: str = Field(..., env="DATABASE_URL")

    LANGSMITH_TRACING: bool = Field(..., env="LANGSMITH_TRACING")
    LANGSMITH_API_KEY: Optional[str] = Field(default=None, env="LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = Field(..., env="LANGSMITH_PROJECT")

    PROMETHEUS_PORT: int = Field(..., env="PROMETHEUS_PORT")
    METRICS_PATH: str = Field(..., env="METRICS_PATH")

    LLM_GUARD_ENABLED: bool = Field(..., env="LLM_GUARD_ENABLED")
    LLM_GUARD_SCANNERS: List[str] = Field(..., env="LLM_GUARD_SCANNERS")

    MAX_UPLOAD_SIZE: int = Field(..., env="MAX_UPLOAD_SIZE")
    ALLOWED_EXTENSIONS: List[str] = Field(..., env="ALLOWED_EXTENSIONS")

    RETRIEVAL_K: int = Field(..., env="RETRIEVAL_K")
    RETRIEVAL_SCORE_THRESHOLD: float = Field(..., env="RETRIEVAL_SCORE_THRESHOLD")

    CHUNK_SIZE: int = Field(..., env="CHUNK_SIZE")
    CHUNK_OVERLAP: int = Field(..., env="CHUNK_OVERLAP")

    RATE_LIMIT_REQUESTS: int = Field(..., env="RATE_LIMIT_REQUESTS")
    RATE_LIMIT_WINDOW: int = Field(..., env="RATE_LIMIT_WINDOW")

    HF_TOKEN: Optional[str] = Field(default=None, env="HF_TOKEN")
    HF_CACHE_DIR: str = Field(..., env="HF_CACHE_DIR")

    # ChromaDB Cloud Settings
    CHROMA_API_KEY: str = Field(..., env="CHROMA_API_KEY")
    CHROMA_TENANT: str = Field(..., env="CHROMA_TENANT")
    CHROMA_DATABASE: str = Field(..., env="CHROMA_DATABASE")

    class Config:
        env_file = ".env"
        case_sensitive = True


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
