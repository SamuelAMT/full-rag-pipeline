"""
Pydantic schemas for API requests and responses.
"""
from datetime import datetime
from typing import Dict, List, Optional, Any

from pydantic import BaseModel, Field, validator, field_validator


class ChatMessage(BaseModel):
    """Chat message model."""
    role: str = Field(..., description="Message role (user/assistant)")
    content: str = Field(..., description="Message content")
    timestamp: Optional[datetime] = Field(default=None, description="Message timestamp")

    @field_validator('role')
    def validate_role(cls, v):
        if v not in ['user', 'assistant', 'system']:
            raise ValueError('Role must be one of: user, assistant, system')
        return v


class DocumentSource(BaseModel):
    """Document source information."""
    document_id: str = Field(..., description="Document ID")
    title: Optional[str] = Field(None, description="Document title")
    content: str = Field(..., description="Relevant content snippet")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    similarity_score: Optional[float] = Field(None, description="Similarity score")
    chunk_index: Optional[int] = Field(None, description="Chunk index in document")


class QueryRequest(BaseModel):
    """Query request model."""
    query: str = Field(..., min_length=1, max_length=1000, description="User query")
    chat_history: Optional[List[ChatMessage]] = Field(
        default=None,
        description="Previous chat messages"
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Filters for document retrieval"
    )
    max_tokens: Optional[int] = Field(
        default=None,
        ge=1,
        le=2048,
        description="Maximum tokens in response"
    )
    temperature: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=2.0,
        description="LLM temperature"
    )
    stream: bool = Field(default=False, description="Whether to stream response")

    @field_validator('query')
    def validate_query(cls, v):
        if not v.strip():
            raise ValueError('Query cannot be empty')
        return v.strip()


class QueryResponse(BaseModel):
    """Query response model."""
    query: str = Field(..., description="Original query")
    answer: str = Field(..., description="Generated answer")
    sources: List[DocumentSource] = Field(default_factory=list, description="Source documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    token_usage: Optional[Dict[str, int]] = Field(None, description="Token usage statistics")


class DocumentUploadRequest(BaseModel):
    """Document upload request model."""
    title: Optional[str] = Field(None, description="Document title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    chunk_size: Optional[int] = Field(None, ge=100, le=2000, description="Chunk size override")
    chunk_overlap: Optional[int] = Field(None, ge=0, le=500, description="Chunk overlap override")


class DocumentUploadResponse(BaseModel):
    """Document upload response model."""
    filename: str = Field(..., description="Uploaded filename")
    document_ids: List[str] = Field(..., description="Generated document IDs")
    chunk_count: int = Field(..., description="Number of chunks created")
    status: str = Field(..., description="Upload status")
    processing_time: Optional[float] = Field(None, description="Processing time in seconds")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Upload metadata")


class DocumentInfo(BaseModel):
    """Document information model."""
    document_id: str = Field(..., description="Document ID")
    title: Optional[str] = Field(None, description="Document title")
    filename: Optional[str] = Field(None, description="Original filename")
    chunk_count: int = Field(..., description="Number of chunks")
    created_at: datetime = Field(..., description="Creation timestamp")
    updated_at: Optional[datetime] = Field(None, description="Last update timestamp")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Document metadata")
    size_bytes: Optional[int] = Field(None, description="Document size in bytes")


class DocumentListResponse(BaseModel):
    """Document list response model."""
    documents: List[DocumentInfo] = Field(..., description="List of documents")
    total_count: int = Field(..., description="Total number of documents")
    page: int = Field(default=1, description="Current page")
    page_size: int = Field(default=20, description="Page size")


class HealthCheckResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Health status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Application version")
    services: Dict[str, str] = Field(default_factory=dict, description="Service statuses")


class SystemInfoResponse(BaseModel):
    """System information response model."""
    vector_store_type: str = Field(..., description="Vector store type")
    llm_provider: str = Field(..., description="LLM provider")
    embedding_model: str = Field(..., description="Embedding model")
    environment: str = Field(..., description="Environment")
    version: str = Field(..., description="Application version")
    features: List[str] = Field(default_factory=list, description="Enabled features")


class ErrorResponse(BaseModel):
    """Error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")


class MetricsResponse(BaseModel):
    """Metrics response model."""
    total_requests: int = Field(..., description="Total requests processed")
    average_response_time: float = Field(..., description="Average response time")
    error_rate: float = Field(..., description="Error rate percentage")
    active_documents: int = Field(..., description="Number of active documents")
    cache_hit_rate: Optional[float] = Field(None, description="Cache hit rate")


# Stream response models
class StreamChunk(BaseModel):
    """Stream response chunk model."""
    type: str = Field(..., description="Chunk type")
    data: Dict[str, Any] = Field(..., description="Chunk data")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Chunk timestamp")


class StreamResponse(BaseModel):
    """Stream response model."""
    query: str = Field(..., description="Original query")
    chunks: List[StreamChunk] = Field(..., description="Response chunks")
    final_answer: Optional[str] = Field(None, description="Final assembled answer")
    sources: List[DocumentSource] = Field(default_factory=list, description="Source documents")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Response metadata")