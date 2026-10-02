"""
rag/reranker.py

2-stage retrieval: 벡터 검색(1단계) → BGE Reranker 재정렬(2단계).

흐름
----
1. VectorStore.similarity_search(query, k=TOP_K_INITIAL)  → 후보 20개
2. Reranker.rerank(query, candidates)                      → 상위 5개
3. agents/rag_retriever.py 에서 이 결과를 state에 저장

왜 2단계?
---------
임베딩 검색은 빠르지만 의미 정밀도가 낮습니다.
Cross-encoder 기반 reranker는 쿼리-문서 쌍을 직접 비교하므로
훨씬 정확하지만 느립니다. 후보를 줄인 뒤 reranker를 적용하면
정확도와 속도를 모두 잡을 수 있습니다.
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)

# Module-level singleton: CrossEncoder is a large PyTorch model (~500MB).
# Reloading it on every Reranker() instantiation exhausts memory and causes
# segfaults on Windows after ~5 calls. One shared instance is sufficient.
_shared_cross_encoder: Optional[object] = None


def _get_cross_encoder(model_name: str):
    global _shared_cross_encoder
    if _shared_cross_encoder is None:
        from sentence_transformers import CrossEncoder
        logger.info("[Reranker] CrossEncoder 모델 로딩: %s", model_name)
        _shared_cross_encoder = CrossEncoder(model_name)
        logger.info("[Reranker] CrossEncoder 로딩 완료.")
    return _shared_cross_encoder


class Reranker:
    """
    BGE-Reranker-v2-m3 기반 2단계 검색 재정렬기.

    Usage
    -----
    >>> reranker = Reranker()
    >>> top5 = reranker.rerank("Python ML engineer", candidates, top_k=5)
    """

    def __init__(self, model_name: Optional[str] = None) -> None:
        from config.settings import settings
        self._model_name = model_name or settings.RERANKER_MODEL
        self._model = None  # lazy load

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            self._model = _get_cross_encoder(self._model_name)
        except Exception as exc:
            logger.error("[Reranker] 모델 로딩 실패: %s", exc)
            raise

    def rerank(self, query: str, candidates: list[str], top_k: int = 5) -> list[str]:
        """
        후보 문서들을 query와의 관련성으로 재정렬하여 상위 top_k 반환.

        Parameters
        ----------
        query : str
            검색 쿼리 (예: 채용공고의 요구 기술).
        candidates : list[str]
            1단계 벡터 검색에서 가져온 후보 문서들.
        top_k : int
            반환할 최종 문서 수.

        Returns
        -------
        list[str]
            관련성 점수 내림차순으로 정렬된 상위 top_k 문서.
        """
        if not candidates:
            return []

        self._load()

        pairs = [(query, doc) for doc in candidates]
        scores = self._model.predict(pairs)  # type: ignore[union-attr]

        ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
        result = [doc for _, doc in ranked[:top_k]]

        logger.info(
            "[Reranker] %d개 후보 → 상위 %d개 선택 (최고점: %.3f)",
            len(candidates), len(result), ranked[0][0] if ranked else 0,
        )
        return result
