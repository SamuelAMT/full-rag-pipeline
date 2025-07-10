"""
LangGraph nodes for the RAG pipeline workflow.
Each node represents a step in the RAG process.
"""

import logging
from typing import Dict, Any, List, Optional
import asyncio

from langsmith import traceable
from pydantic import BaseModel, Field

from src.core.config import get_settings
from src.vectorstore.chroma_store import ChromaVectorStore
from src.vectorstore.embeddings import get_embedding_manager
from src.generation.llm_models import get_llm_manager
from src.generation.prompts import get_prompt_manager, format_context, format_chat_history
from src.safety.content_filter import ContentFilter
from src.monitoring.metrics import get_metrics_collector

logger = logging.getLogger(__name__)


class WorkflowState(BaseModel):
    """State object that flows through the workflow."""

    # Input
    query: str = Field(..., description="Original user query")
    chat_history: List[Dict[str, str]] = Field(default_factory=list, description="Chat history")

    # Processing
    enhanced_query: Optional[str] = Field(None, description="Enhanced search query")
    retrieved_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Retrieved documents")
    reranked_docs: List[Dict[str, Any]] = Field(default_factory=list, description="Reranked documents")
    context: Optional[str] = Field(None, description="Formatted context for LLM")

    # Output
    response: Optional[str] = Field(None, description="Generated response")
    sources: List[Dict[str, Any]] = Field(default_factory=list, description="Source documents")

    # Metadata
    is_safe: bool = Field(True, description="Safety check result")
    error: Optional[str] = Field(None, description="Error message if any")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")


class RAGNodes:
    """Collection of nodes for the RAG workflow."""

    def __init__(self):
        self.settings = get_settings()
        self.vector_store = ChromaVectorStore()
        self.embedding_manager = get_embedding_manager()
        self.llm_manager = get_llm_manager()
        self.prompt_manager = get_prompt_manager()