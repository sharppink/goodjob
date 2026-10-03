"""
agents/rag_retriever.py

LangGraph 노드: rag_retriever

역할
----
파싱된 job_requirements를 쿼리로 변환하여
벡터 DB에서 사용자의 관련 경험을 검색합니다.
Reranker가 활성화된 경우 2단계 검색(벡터 → 리랭킹)을 수행합니다.

결과는 state["retrieved_experiences"] 에 저장됩니다.
"""

from __future__ import annotations

import logging
from typing import Optional

from agents.state import GoodJobState

logger = logging.getLogger(__name__)

# 청크 수가 이 값 이하인 작은 프로필은 검색으로 거르지 않고 전부 사용
# (한 장짜리 이력서는 전체를 넣어도 토큰이 적고, 다중 쿼리가 같은 상위 청크만
#  반복 반환해 프로젝트 등 일부 섹션이 누락되는 문제를 막음 — PROJECT_DOCS #008)
SMALL_PROFILE_CHUNKS = 12
MAX_RESULTS = 8


def rag_retriever_node(state: GoodJobState) -> GoodJobState:
    """job_requirements를 기반으로 사용자 경험을 벡터 검색합니다."""
    logger.info("[rag_retriever] RAG 검색 시작.")
    state["current_step"] = "rag_retriever"
    errors: list[str] = state.get("errors") or []

    job_requirements: Optional[dict] = state.get("job_requirements")
    if not job_requirements:
        logger.warning("[rag_retriever] job_requirements 없음 — 검색 건너뜀.")
        errors.append("rag_retriever: job_requirements 없음.")
        state["errors"] = errors
        state["retrieved_experiences"] = []
        return state

    # 벡터 DB에 저장된 문서가 있는지 확인
    from rag.vectorstore import VectorStore
    vs = VectorStore()
    vs.initialize()
    if vs.count() == 0:
        logger.warning("[rag_retriever] 저장된 프로필 없음 — 검색 건너뜀.")
        errors.append("rag_retriever: 프로필이 벡터 DB에 없습니다. 먼저 프로필을 업로드하세요.")
        state["errors"] = errors
        state["retrieved_experiences"] = []
        return state

    total = vs.count()
    if total <= SMALL_PROFILE_CHUNKS:
        docs = vs.get_all_documents()
        logger.info("[rag_retriever] 작은 프로필(%d개 청크) — 전체 사용.", total)
        state["retrieved_experiences"] = docs
        state["errors"] = errors
        return state

    # job_requirements → 검색 쿼리 생성 (다각도 쿼리)
    queries = _build_queries(job_requirements)
    logger.info("[rag_retriever] 생성된 쿼리 %d개", len(queries))

    # ProfileLoader.search() 사용 → Reranker 자동 적용
    from rag.profile_loader import ProfileLoader
    loader = ProfileLoader()

    all_results: list[str] = []
    seen: set[str] = set()

    for query in queries:
        results = loader.search(query, k=3)
        for r in results:
            if r not in seen:
                seen.add(r)
                all_results.append(r)

    # 쿼리별 결과의 합집합에서 상위 MAX_RESULTS 개 유지
    final = all_results[:MAX_RESULTS]
    logger.info("[rag_retriever] 최종 검색 결과: %d개 경험 청크", len(final))

    state["retrieved_experiences"] = final
    state["errors"] = errors
    return state


# ------------------------------------------------------------------ #
# Helpers                                                              #
# ------------------------------------------------------------------ #

def _build_queries(req: dict) -> list[str]:
    """
    job_requirements에서 다각도 검색 쿼리를 생성합니다.

    단일 쿼리보다 여러 관점의 쿼리를 사용하면 RAG 리콜이 향상됩니다.
    """
    queries: list[str] = []

    required = req.get("required_skills") or []
    preferred = req.get("preferred_skills") or []
    responsibilities = req.get("responsibilities") or []
    keywords = req.get("keywords") or []

    # 쿼리 1: 필수 기술 중심
    if required:
        queries.append("기술 경험: " + ", ".join(required[:8]))

    # 쿼리 2: 주요 업무 중심
    if responsibilities:
        queries.append("업무 경험: " + " / ".join(responsibilities[:4]))

    # 쿼리 3: 키워드 조합
    if keywords:
        queries.append(" ".join(keywords[:10]))

    # 쿼리 4: 우대 기술 (없으면 생략)
    if preferred:
        queries.append("우대 기술: " + ", ".join(preferred[:5]))

    # 최소 1개 보장
    if not queries:
        queries.append("소프트웨어 개발 경험")

    return queries
