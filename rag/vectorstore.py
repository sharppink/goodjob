"""
rag/vectorstore.py

Wraps ChromaDB with a simple interface used by the rest of the application.

The collection name is fixed to ``"goodjob_profile"`` so that all profile
chunks live in a single, easily-queried collection.

계정별 분리: set_current_user(이메일) 이 호출된 컨텍스트에서는 계정마다 따로 저장됩니다.
  - Chroma: 컬렉션 이름에 이메일 해시를 붙임
  - Supabase(SUPABASE_DB_URL): 테이블 하나에 user_key(이메일 해시) 열로 구분 (rag/pg_store.py)
로그인을 쓰지 않으면(로컬 실행·API·테스트) 기존처럼 공용 컬렉션 하나를 씁니다.
"""

from __future__ import annotations

import hashlib
import logging
import os
import uuid
from contextvars import ContextVar
from typing import Any, Optional

import chromadb
from chromadb.config import Settings as ChromaSettings

from config.settings import settings
from rag.embeddings import EmbeddingModel

logger = logging.getLogger(__name__)

COLLECTION_NAME = "goodjob_profile"
PRIVATE_COLLECTION_NAME = "goodjob_profile_local"

# 현재 로그인한 계정 — Streamlit 은 세션마다 스크립트를 별도 스레드에서 실행하므로
# 실행 시작 시 set_current_user 로 정하면 그 실행 안의 모든 VectorStore 가 같은 계정을 씀
_current_user: ContextVar[Optional[str]] = ContextVar("goodjob_current_user", default=None)
_login_required = False
LOCAL_USER_KEY = "local"


def set_current_user(email: Optional[str]) -> None:
    """이후 이 컨텍스트에서 만드는 VectorStore 가 쓸 계정 (None 이면 공용)."""
    _current_user.set(email.strip().lower() if email else None)


def get_current_user() -> Optional[str]:
    return _current_user.get()


def require_login(required: bool = True) -> None:
    """로그인을 쓰는 배포에서 계정 없이 저장소에 접근하면 공용 데이터를 읽지 않고 오류를 내도록 함."""
    global _login_required
    _login_required = required


def user_key(email: Optional[str]) -> str:
    """이메일을 그대로 저장하지 않도록 해시로 바꾼 계정 식별자."""
    if not email:
        return LOCAL_USER_KEY
    return hashlib.sha256(email.strip().lower().encode("utf-8")).hexdigest()[:32]


# Process-wide singleton — prevents multiple PersistentClient instances from
# opening the same SQLite file concurrently (causes segfault on Windows).
# 경로별로 하나씩 유지 — 예전에는 첫 경로의 클라이언트를 모든 경로에 재사용해서
# VectorStore(persist_dir=다른경로) 가 무시되는 버그가 있었음 (PROJECT_DOCS #009)
_shared_clients: dict[str, chromadb.ClientAPI] = {}


def _get_client(persist_dir: str) -> chromadb.ClientAPI:
    # CHROMA_HOST 가 있으면 Chroma 서버에 접속 — 로컬 파일 모드는 단일 프로세스 전용이라
    # 다른 프로세스가 쓴 벡터를 검색하지 못함 (PROJECT_DOCS #024)
    if settings.CHROMA_HOST:
        key = f"http://{settings.CHROMA_HOST}:{settings.CHROMA_PORT}"
        if key not in _shared_clients:
            _shared_clients[key] = chromadb.HttpClient(
                host=settings.CHROMA_HOST,
                port=settings.CHROMA_PORT,
                settings=ChromaSettings(anonymized_telemetry=False),
            )
        return _shared_clients[key]

    key = os.path.abspath(persist_dir)
    if key not in _shared_clients:
        _shared_clients[key] = chromadb.PersistentClient(
            path=persist_dir,
            settings=ChromaSettings(anonymized_telemetry=False),
        )
    return _shared_clients[key]


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

        email = get_current_user()
        if email is None and _login_required:
            raise RuntimeError("로그인한 계정이 없어 프로필 저장소에 접근할 수 없습니다.")

        # 개인정보 보호 모드는 임베딩 차원(bge-m3 1024)이 달라 별도 컬렉션 사용
        from config.settings import settings as _s
        self._collection_name = PRIVATE_COLLECTION_NAME if _s.PRIVACY_MODE else COLLECTION_NAME

        # 보호 모드는 프로필을 PC 밖으로 보내지 않으므로 Supabase 를 쓰지 않음
        # persist_dir 을 직접 지정한 경우(평가·테스트용 임시 DB)도 로컬 Chroma 유지
        if _s.SUPABASE_DB_URL and not _s.PRIVACY_MODE and self._persist_dir == _s.CHROMA_PERSIST_DIR:
            from rag.pg_store import PgCollection
            logger.info("[VectorStore] Using Supabase Postgres (pgvector).")
            self._collection = PgCollection(_s.SUPABASE_DB_URL, self._collection_name, user_key(email))
        else:
            logger.info("[VectorStore] Initialising Chroma at '%s'.", self._persist_dir)
            self._client = _get_client(self._persist_dir)
            name = self._collection_name
            if email is not None:
                name = f"{name}__{user_key(email)[:16]}"
            self._collection = self._client.get_or_create_collection(
                name=name,
                metadata={"hnsw:space": "cosine"},
            )
        logger.info(
            "[VectorStore] Collection '%s' ready (%d documents).",
            self._collection_name,
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
        ids = [f"doc_{uuid.uuid4().hex}" for _ in docs]
        # chromadb 1.x 는 빈 metadata dict 를 거부하므로 기본값을 채움 (PROJECT_DOCS #021)
        metadatas = [m or {"source": "unspecified"} for m in (metadatas or [{}] * len(docs))]

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

    def get_all_documents(self) -> list[str]:
        """Return every stored document text (insertion order not guaranteed)."""
        self._ensure_initialized()
        return self._collection.get(include=["documents"])["documents"]  # type: ignore[union-attr]

    def get_all_entries(self) -> list[dict[str, Any]]:
        """저장된 청크를 메타데이터와 함께 반환 (프로필 확인 화면용, 저장 순서)."""
        self._ensure_initialized()
        data = self._collection.get(include=["documents", "metadatas"])  # type: ignore[union-attr]
        metadatas = data.get("metadatas") or [{}] * len(data["documents"])
        return [
            {"id": i, "document": d, "metadata": m or {}}
            for i, d, m in zip(data["ids"], data["documents"], metadatas)
        ]

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
