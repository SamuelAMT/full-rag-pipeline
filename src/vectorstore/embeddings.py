"""
Embedding utilities for the RAG pipeline.
Handles text embedding generation using HuggingFace models.
"""

import logging
from typing import List, Optional, Dict, Any
import asyncio
from functools import lru_cache

from InstructorEmbedding import INSTRUCTOR
import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch

from src.core.config import get_settings

logger = logging.getLogger(__name__)


class EmbeddingModel:
    """Base class for embedding models."""

    def __init__(self, model_name: str):
        self.model_name = model_name
        self.model = None
        self.tokenizer = None

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """Embed a list of texts."""
        raise NotImplementedError

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        raise NotImplementedError


class InstructorEmbedding(EmbeddingModel):
    """Instructor embedding model for semantic search."""

    def __init__(self, model_name: str = None):
        model_name = model_name or get_settings().EMBEDDING_MODEL
        super().__init__(model_name)
        self.model = None
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the Instructor model."""
        try:
            logger.info(f"Loading Instructor model: {self.model_name}")
            self.model = INSTRUCTOR(self.model_name)
            logger.info("Instructor model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Instructor model: {e}")
            raise

    async def embed_texts(self, texts: List[str], instruction: str = "Represent the document for retrieval:") -> List[
        List[float]]:
        """Embed a list of texts with instruction."""
        if not self.model:
            raise ValueError("Model not initialized")

        try:
            texts_with_instruction = [[instruction, text] for text in texts]

            # Run embedding in thread pool to avoid blocking
            loop = asyncio.get_event_loop()
            embeddings = await loop.run_in_executor(
                None,
                self.model.encode,
                texts_with_instruction
            )

            return embeddings.tolist()

        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise

    async def embed_query(self, query: str, instruction: str = "Represent the query for retrieval:") -> List[float]:
        """Embed a single query with instruction."""
        if not self.model:
            raise ValueError("Model not initialized")

        try:
            query_with_instruction = [instruction, query]

            loop = asyncio.get_event_loop()
            embedding = await loop.run_in_executor(
                None,
                self.model.encode,
                [query_with_instruction]
            )

            return embedding[0].tolist()

        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise


class HuggingFaceEmbedding(EmbeddingModel):
    """HuggingFace transformer embedding model."""

    def __init__(self, model_name: str = None):
        model_name = model_name or get_settings().EMBEDDING_MODEL
        super().__init__(model_name)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._initialize_model()

    def _initialize_model(self):
        """Initialize the HuggingFace model."""
        try:
            logger.info(f"Loading HuggingFace model: {self.model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()
            logger.info("HuggingFace model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load HuggingFace model: {e}")
            raise

    def _mean_pooling(self, model_output, attention_mask):
        """Mean pooling for sentence embeddings."""
        token_embeddings = model_output[0]
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)

    async def embed_texts(self, texts: List[str], max_length: int = 512) -> List[List[float]]:
        """Embed a list of texts."""
        if not self.model:
            raise ValueError("Model not initialized")

        try:
            encoded_input = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors='pt'
            ).to(self.device)

            # Generate embeddings
            with torch.no_grad():
                model_output = self.model(**encoded_input)

            embeddings = self._mean_pooling(model_output, encoded_input['attention_mask'])

            embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            return embeddings.cpu().numpy().tolist()

        except Exception as e:
            logger.error(f"Error embedding texts: {e}")
            raise

    async def embed_query(self, query: str, max_length: int = 512) -> List[float]:
        """Embed a single query."""
        embeddings = await self.embed_texts([query], max_length)
        return embeddings[0]


class EmbeddingManager:
    """Manager for embedding operations."""

    def __init__(self, embedding_model: str = None):
        self.settings = get_settings()
        self.embedding_model_name = embedding_model or self.settings.EMBEDDING_MODEL
        self.embedding_model = None
        self._initialize_embedding_model()

    def _initialize_embedding_model(self):
        """Initialize the embedding model based on configuration."""
        try:
            if "instructor" in self.embedding_model_name.lower():
                self.embedding_model = InstructorEmbedding(self.embedding_model_name)
            else:
                self.embedding_model = HuggingFaceEmbedding(self.embedding_model_name)

            logger.info(f"Embedding model initialized: {self.embedding_model_name}")

        except Exception as e:
            logger.error(f"Failed to initialize embedding model: {e}")
            raise

    async def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Embed a list of documents."""
        if not documents:
            return []

        try:
            batch_size = self.settings.EMBEDDING_BATCH_SIZE
            all_embeddings = []

            for i in range(0, len(documents), batch_size):
                batch = documents[i:i + batch_size]
                batch_embeddings = await self.embedding_model.embed_texts(batch)
                all_embeddings.extend(batch_embeddings)

            return all_embeddings

        except Exception as e:
            logger.error(f"Error embedding documents: {e}")
            raise

    async def embed_query(self, query: str) -> List[float]:
        """Embed a single query."""
        try:
            return await self.embedding_model.embed_query(query)
        except Exception as e:
            logger.error(f"Error embedding query: {e}")
            raise

    def get_embedding_dimension(self) -> int:
        """Get the dimension of embeddings."""
        model_dimensions = {
            "hkunlp/instructor-large": 768,
            "hkunlp/instructor-base": 768,
            "sentence-transformers/all-MiniLM-L6-v2": 384,
            "sentence-transformers/all-mpnet-base-v2": 768,
        }

        return model_dimensions.get(self.embedding_model_name, 768)

    async def compute_similarity(self, embedding1: List[float], embedding2: List[float]) -> float:
        """Compute cosine similarity between two embeddings."""
        try:
            vec1 = np.array(embedding1)
            vec2 = np.array(embedding2)

            dot_product = np.dot(vec1, vec2)
            norm1 = np.linalg.norm(vec1)
            norm2 = np.linalg.norm(vec2)

            if norm1 == 0 or norm2 == 0:
                return 0.0

            return dot_product / (norm1 * norm2)

        except Exception as e:
            logger.error(f"Error computing similarity: {e}")
            return 0.0


@lru_cache()
def get_embedding_manager() -> EmbeddingManager:
    """Get cached embedding manager instance."""
    return EmbeddingManager()


async def embed_documents(documents: List[str]) -> List[List[float]]:
    """Embed a list of documents."""
    manager = get_embedding_manager()
    return await manager.embed_documents(documents)


async def embed_query(query: str) -> List[float]:
    """Embed a single query."""
    manager = get_embedding_manager()
    return await manager.embed_query(query)


async def compute_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute cosine similarity between two embeddings."""
    manager = get_embedding_manager()
    return await manager.compute_similarity(embedding1, embedding2)