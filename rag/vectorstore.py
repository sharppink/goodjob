"""
rag/vectorstore.py

Wraps ChromaDB with a simple interface used by the rest of the application.

The collection name is fixed to ``"goodjob_profile"`` so that all profile
chunks live in a single, easily-queried collection.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from rag.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

COLLECTION_NAME = "goodjob_profile"

# Process-wide singleton — prevents multiple PersistentClient instances from
# opening the same SQLite file concurrently (causes segfault on Windows).
_shared_client: Optional[chromadb.PersistentClient] = None


def _get_client(persist_dir: str) -> chromadb.PersistentClient:
    global _shared_client
    if _shared_client is None:
        _shared_client = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _shared_client


class VectorStore:
    """
    Persistent Chroma vector store for user profile chunks.

    Usage
    -----
    >>> vs = VectorStore()
    >>> vs.initialize()
    >>> vs.add_documents(["5 years Python experience", "Led ML team of 4"])
    >>> results = vs.similarity_search("Python machine learning", k=3)
    """

    def __init__(self, persist_dir: Optional[str] = None) -> None:
        """
        Parameters
        ----------
        persist_dir : str, optional
            Directory where Chroma stores its files.  Defaults to
            ``settings.CHROMA_PERSIST_DIR``.
        """
        self._persist_dir = persist_dir or settings.CHROMA_PERSIST_DIR
        self._client: Optional[chromadb.PersistentClient] = None
        self._collection: Optional[chromadb.Collection] = None
        self._embedding_model = EmbeddingModel()

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def initialize(self) -> None:
        """
        Connect to (or create) the persistent Chroma client and collection.

        Safe to call multiple times – idempotent.
        """
        if self._collection is not None:
            return

        logger.info("[VectorStore] Initialising Chroma at '%s'.", self._persist_dir)
        self._client = _get_client(self._persist_dir)
        self._collection = self._client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
        logger.info(
            "[VectorStore] Collection '%s' ready (%d documents).",
            COLLECTION_NAME,
            self._collection.count(),
        )

    # ------------------------------------------------------------------ #
    # CRUD                                                                 #
    # ------------------------------------------------------------------ #

    def add_documents(self, docs: list[str], metadatas: Optional[list[dict]] = None) -> None:
        """
        Embed and insert a list of text chunks into the collection.

        Parameters
        ----------
        docs : list[str]
            Plain-text chunks to add.
        metadatas : list[dict], optional
            Parallel list of metadata dicts (e.g. ``{"source": "cv.pdf"}``).
        """
        self._ensure_initialized()
        if not docs:
            logger.warning("[VectorStore] add_documents called with empty list.")
            return

        embeddings = self._embedding_model.embed_texts(docs)
        ids = [f"doc_{i}_{hash(d) & 0xFFFFFF}" for i, d in enumerate(docs)]
        metadatas = metadatas or [{}] * len(docs)

        self._collection.add(  # type: ignore[union-attr]
            documents=docs,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )
        logger.info("[VectorStore] Added %d documents.", len(docs))

    def similarity_search(self, query: str, k: int = 5) -> list[str]:
        """
        Return the top-k most similar document texts for the given query.

        Parameters
        ----------
        query : str
            The search query.
        k : int
            Number of results to return.

        Returns
        -------
        list[str]
            Ordered list of matching document texts (most similar first).
        """
        self._ensure_initialized()
        query_embedding = self._embedding_model.embed_query(query)

        results = self._collection.query(  # type: ignore[union-attr]
            query_embeddings=[query_embedding],
            n_results=min(k, max(1, self._collection.count())),  # type: ignore[union-attr]
        )
        documents: list[list[str]] = results.get("documents", [[]])
        return documents[0] if documents else []

    def delete_all(self) -> None:
        """
        Delete every document in the collection (useful for re-indexing).
        """
        self._ensure_initialized()
        all_ids = self._collection.get(include=[])["ids"]  # type: ignore[union-attr]
        if all_ids:
            self._collection.delete(ids=all_ids)  # type: ignore[union-attr]
            logger.info("[VectorStore] Deleted %d documents.", len(all_ids))
        else:
            logger.info("[VectorStore] Collection already empty.")

    def count(self) -> int:
        """Return the number of documents currently stored."""
        self._ensure_initialized()
        return self._collection.count()  # type: ignore[union-attr]

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _ensure_initialized(self) -> None:
        if self._collection is None:
            self.initialize()
