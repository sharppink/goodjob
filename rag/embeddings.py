"""
rag/embeddings.py

RAG 파이프라인에서 사용하는 임베딩 모델.

기본: OpenAI text-embedding-3-small (API 기반, 로컬 다운로드 불필요)
대안: BAAI/bge-m3 (로컬, 디스크 2.4GB 필요 시 사용)

USE_OPENAI_EMBEDDINGS=true (기본) → OpenAI API 사용
USE_OPENAI_EMBEDDINGS=false      → 로컬 BGE-M3 사용
"""

from __future__ import annotations

import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

OPENAI_EMBEDDING_MODEL = "text-embedding-3-small"  # 1536차원, 저비용
BGE_MODEL_NAME = "BAAI/bge-m3"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

# Singleton OpenAI client shared with llm/openai_client.py to avoid
# creating multiple httpx pools that trigger GC-related segfaults on Windows.
_shared_openai_embed: Optional[object] = None


def _get_openai_embed_client(api_key: str):
    global _shared_openai_embed
    if _shared_openai_embed is None:
        from openai import OpenAI
        _shared_openai_embed = OpenAI(api_key=api_key)
    return _shared_openai_embed


class EmbeddingModel:
    """
    임베딩 모델 래퍼. OpenAI API 또는 로컬 BGE-M3를 사용합니다.

    Usage
    -----
    >>> em = EmbeddingModel()
    >>> vectors = em.embed_texts(["Python 개발자", "FastAPI 3년 경험"])
    >>> query_vec = em.embed_query("백엔드 개발 경험")
    """

    def __init__(self, use_openai: Optional[bool] = None) -> None:
        """
        Parameters
        ----------
        use_openai : bool, optional
            True면 OpenAI API 사용, False면 로컬 BGE-M3 사용.
            None이면 환경변수 USE_OPENAI_EMBEDDINGS 참조 (기본 True).
        """
        if use_openai is None:
            use_openai = os.getenv("USE_OPENAI_EMBEDDINGS", "true").lower() != "false"
        self._use_openai = use_openai
        self._model = None  # lazy load

        if use_openai:
            logger.info("[EmbeddingModel] OpenAI 임베딩 사용 (model=%s)", OPENAI_EMBEDDING_MODEL)
        else:
            logger.info("[EmbeddingModel] 로컬 BGE-M3 사용")

    # ------------------------------------------------------------------ #
    # Public API                                                           #
    # ------------------------------------------------------------------ #

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        문서 리스트를 임베딩합니다 (인덱싱용).

        Returns
        -------
        list[list[float]]
            각 텍스트의 임베딩 벡터 리스트.
        """
        if not texts:
            return []
        if self._use_openai:
            return self._openai_embed(texts)
        return self._bge_embed(texts)

    def embed_query(self, query: str) -> list[float]:
        """
        검색 쿼리 하나를 임베딩합니다.

        Returns
        -------
        list[float]
            쿼리 임베딩 벡터.
        """
        if self._use_openai:
            return self._openai_embed([query])[0]
        # BGE는 쿼리에 prefix 적용
        prefixed = BGE_QUERY_PREFIX + query
        return self._bge_embed([prefixed])[0]

    @property
    def model_name(self) -> str:
        return OPENAI_EMBEDDING_MODEL if self._use_openai else BGE_MODEL_NAME

    @property
    def dimension(self) -> int:
        """임베딩 벡터 차원 수."""
        return 1536 if self._use_openai else 1024

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _openai_embed(self, texts: list[str]) -> list[list[float]]:
        """OpenAI Embeddings API 호출."""
        try:
            from dotenv import load_dotenv
            load_dotenv()
            client = _get_openai_embed_client(os.getenv("OPENAI_API_KEY", ""))
            response = client.embeddings.create(
                model=OPENAI_EMBEDDING_MODEL,
                input=texts,
            )
            return [item.embedding for item in response.data]
        except ImportError as exc:
            raise ImportError("openai 패키지가 필요합니다: pip install openai") from exc

    def _bge_embed(self, texts: list[str]) -> list[list[float]]:
        """로컬 BGE-M3 임베딩 (sentence-transformers)."""
        if self._model is None:
            logger.info("[EmbeddingModel] BGE-M3 로딩 중 (약 2.4GB)...")
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(BGE_MODEL_NAME)
            logger.info("[EmbeddingModel] BGE-M3 로딩 완료.")
        embeddings = self._model.encode(texts, normalize_embeddings=True)
        return embeddings.tolist()
