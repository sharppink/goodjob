"""
rag package

Provides the vector-store interface, embedding model, and profile loader
used by the RAG retriever agent node.
"""

from rag.vectorstore import VectorStore
from rag.embeddings import EmbeddingModel
from rag.profile_loader import ProfileLoader

__all__ = ["VectorStore", "EmbeddingModel", "ProfileLoader"]
