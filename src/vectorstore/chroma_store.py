"""
ChromaDB vector store implementation for the RAG pipeline.
"""
import os
import time
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

import chromadb
from chromadb.config import Settings
from chromadb.utils import embedding_functions

from src.core.config import get_settings
from src.core.logging import get_logger
from src.monitoring.metrics import metrics_collector

logger = get_logger(__name__)
settings = get_settings()


class ChromaVectorStore:
    """ChromaDB vector store implementation."""

    def __init__(self, collection_name: str = None, persist_directory: str = None):
        self.collection_name = collection_name or settings.CHROMA_COLLECTION_NAME
        self.persist_directory = persist_directory or settings.CHROMA_PERSIST_DIR
        self.client = None
        self.collection = None
        self.embedding_function = None

        os.makedirs(self.persist_directory, exist_ok=True)

        self._initialize_client()

    def _initialize_client(self):
        """Initialize ChromaDB client."""
        try:
            self.client = chromadb.PersistentClient(
                path=self.persist_directory,
                settings=Settings(
                    anonymized_telemetry=False,
                    allow_reset=True
                )
            )

            self.embedding_function = embedding_functions.SentenceTransformerEmbeddingFunction(
                model_name=settings.EMBEDDING_MODEL
            )

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "RAG pipeline document collection"}
            )

            logger.info(f"ChromaDB initialized with collection: {self.collection_name}")

        except Exception as e:
            logger.error(f"Failed to initialize ChromaDB: {e}")
            raise

    async def add_documents(self, documents: List[Dict[str, Any]]) -> List[str]:
        """Add documents to the vector store."""
        start_time = time.time()

        try:
            document_ids = []
            texts = []
            metadatas = []
            ids = []

            for doc in documents:
                doc_id = doc.get('id', f"doc_{len(document_ids)}")
                document_ids.append(doc_id)
                ids.append(doc_id)
                texts.append(doc['content'])

                metadata = {
                    'title': doc.get('title', ''),
                    'topic': doc.get('topic', ''),
                    'source': doc.get('source', ''),
                    'created_at': str(doc.get('created_at', datetime.now())),
                    'length': len(doc['content'])
                }

                # Possibility to add additional metadata if available
                if 'metadata' in doc:
                    for key, value in doc['metadata'].items():
                        # ChromaDB only supports str, int, float, bool
                        if isinstance(value, (str, int, float, bool)):
                            metadata[key] = value
                        else:
                            metadata[key] = str(value)

                metadatas.append(metadata)

            self.collection.add(
                ids=ids,
                documents=texts,
                metadatas=metadatas
            )

            duration = time.time() - start_time
            metrics_collector.record_vector_operation("chroma", "add", duration)
            metrics_collector.record_document_indexed("chroma", len(documents))

            logger.info(f"Added {len(documents)} documents to ChromaDB in {duration:.2f}s")
            return document_ids

        except Exception as e:
            logger.error(f"Failed to add documents to ChromaDB: {e}")
            raise

    async def search(self, query: str, k: int = None, filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """Search for similar documents."""
        start_time = time.time()
        k = k or settings.RETRIEVAL_K

        try:
            where_clause = None
            if filters:
                where_clause = {}
                for key, value in filters.items():
                    if isinstance(value, (str, int, float, bool)):
                        where_clause[key] = value
                    elif isinstance(value, list):
                        where_clause[key] = {"$in": value}

            results = self.collection.query(
                query_texts=[query],
                n_results=k,
                where=where_clause,
                include=["documents", "metadatas", "distances"]
            )

            search_results = []
            if results['documents'] and results['documents'][0]:
                for i, (doc, metadata, distance) in enumerate(zip(
                        results['documents'][0],
                        results['metadatas'][0],
                        results['distances'][0]
                )):
                    # Convert distance to similarity score (ChromaDB uses L2 distance)
                    similarity_score = 1 / (1 + distance)

                    if similarity_score >= settings.RETRIEVAL_SCORE_THRESHOLD:
                        search_results.append({
                            'document_id': results['ids'][0][i],
                            'content': doc,
                            'metadata': metadata,
                            'similarity_score': similarity_score,
                            'distance': distance
                        })

            duration = time.time() - start_time
            metrics_collector.record_vector_operation("chroma", "search", duration)
            metrics_collector.record_documents_retrieved("chroma", len(search_results))

            logger.info(f"Found {len(search_results)} documents for query in {duration:.2f}s")
            return search_results

        except Exception as e:
            logger.error(f"Failed to search ChromaDB: {e}")
            raise

    async def delete_document(self, document_id: str) -> bool:
        """Delete a document from the vector store."""
        start_time = time.time()

        try:
            result = self.collection.get(ids=[document_id])
            if not result['ids']:
                logger.warning(f"Document {document_id} not found in ChromaDB")
                return False

            self.collection.delete(ids=[document_id])

            duration = time.time() - start_time
            metrics_collector.record_vector_operation("chroma", "delete", duration)

            logger.info(f"Deleted document {document_id} from ChromaDB in {duration:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to delete document {document_id}: {e}")
            return False

    async def get_document(self, document_id: str) -> Optional[Dict[str, Any]]:
        """Get a specific document by ID."""
        start_time = time.time()

        try:
            result = self.collection.get(
                ids=[document_id],
                include=["documents", "metadatas"]
            )

            if not result['ids']:
                return None

            duration = time.time() - start_time
            metrics_collector.record_vector_operation("chroma", "get", duration)

            return {
                'document_id': result['ids'][0],
                'content': result['documents'][0],
                'metadata': result['metadatas'][0]
            }

        except Exception as e:
            logger.error(f"Failed to get document {document_id}: {e}")
            return None

    async def list_documents(self, limit: int = 100, offset: int = 0) -> List[Dict[str, Any]]:
        """List all documents in the vector store."""
        start_time = time.time()

        try:
            # ChromaDB doesn't support limit/offset directly, so I get all and slice
            result = self.collection.get(include=["documents", "metadatas"])

            if not result['ids']:
                return []

            documents = []
            for i, (doc_id, doc, metadata) in enumerate(zip(
                    result['ids'],
                    result['documents'],
                    result['metadatas']
            )):
                if i >= offset and len(documents) < limit:
                    documents.append({
                        'document_id': doc_id,
                        'title': metadata.get('title', ''),
                        'topic': metadata.get('topic', ''),
                        'created_at': metadata.get('created_at', ''),
                        'length': metadata.get('length', 0),
                        'metadata': metadata
                    })

            duration = time.time() - start_time
            metrics_collector.record_vector_operation("chroma", "list", duration)

            logger.info(f"Listed {len(documents)} documents from ChromaDB in {duration:.2f}s")
            return documents

        except Exception as e:
            logger.error(f"Failed to list documents: {e}")
            return []

    async def get_collection_stats(self) -> Dict[str, Any]:
        """Get collection statistics."""
        try:
            result = self.collection.get()

            total_documents = len(result['ids']) if result['ids'] else 0

            topic_counts = {}
            if result['metadatas']:
                for metadata in result['metadatas']:
                    topic = metadata.get('topic', 'unknown')
                    topic_counts[topic] = topic_counts.get(topic, 0) + 1

            return {
                'total_documents': total_documents,
                'collection_name': self.collection_name,
                'topic_distribution': topic_counts,
                'embedding_model': settings.EMBEDDING_MODEL,
                'persist_directory': self.persist_directory
            }

        except Exception as e:
            logger.error(f"Failed to get collection stats: {e}")
            return {}

    async def clear_collection(self) -> bool:
        """Clear all documents from the collection."""
        try:
            self.client.delete_collection(name=self.collection_name)

            self.collection = self.client.get_or_create_collection(
                name=self.collection_name,
                embedding_function=self.embedding_function,
                metadata={"description": "RAG pipeline document collection"}
            )

            logger.info(f"Cleared collection: {self.collection_name}")
            return True

        except Exception as e:
            logger.error(f"Failed to clear collection: {e}")
            return False

    async def update_document(self, document_id: str, content: str, metadata: Dict[str, Any] = None) -> bool:
        """Update a document in the vector store."""
        start_time = time.time()

        try:
            existing = await self.get_document(document_id)
            if not existing:
                logger.warning(f"Document {document_id} not found for update")
                return False

            update_metadata = existing['metadata'].copy()
            if metadata:
                for key, value in metadata.items():
                    if isinstance(value, (str, int, float, bool)):
                        update_metadata[key] = value
                    else:
                        update_metadata[key] = str(value)

            self.collection.update(
                ids=[document_id],
                documents=[content],
                metadatas=[update_metadata]
            )

            duration = time.time() - start_time
            metrics_collector.record_vector_operation("chroma", "update", duration)

            logger.info(f"Updated document {document_id} in ChromaDB in {duration:.2f}s")
            return True

        except Exception as e:
            logger.error(f"Failed to update document {document_id}: {e}")
            return False

    def health_check(self) -> Dict[str, Any]:
        """Check the health of the vector store."""
        try:
            collection_info = self.collection.get()

            return {
                'status': 'healthy',
                'collection_name': self.collection_name,
                'document_count': len(collection_info['ids']) if collection_info['ids'] else 0,
                'persist_directory': self.persist_directory,
                'embedding_model': settings.EMBEDDING_MODEL
            }

        except Exception as e:
            logger.error(f"ChromaDB health check failed: {e}")
            return {
                'status': 'unhealthy',
                'error': str(e)
            }