"""
agents/job_recommender.py

역방향 매칭 노드: 사용자 프로필에 맞는 채용공고를 자동으로 탐색·랭킹합니다.

흐름
----
1. Tavily API로 키워드 기반 공고 다수 수집
2. 각 공고를 Structured Output(JobRequirements)으로 구조화
3. RAG로 사용자 관련 경험 검색
4. fit_analyzer 로직으로 점수화
5. 내림차순 정렬 → ranked_matches 반환

결과는 state["ranked_matches"] 에 저장됩니다.
"""

from __future__ import annotations

import logging
from typing import Any

from agents.state import GoodJobState

logger = logging.getLogger(__name__)

MAX_CANDIDATES   = 10   # 수집할 최대 공고 수
MAX_RANKED       = 5    # 반환할 최종 순위 수
SCORE_THRESHOLD  = 0.3  # 이 점수 미만은 결과에서 제외


# ------------------------------------------------------------------ #
# Node                                                                 #
# ------------------------------------------------------------------ #

def job_recommender_node(state: GoodJobState) -> GoodJobState:
    """
    사용자 프로필 기반으로 채용공고를 자동 탐색하고 적합도 순으로 랭킹합니다.

    state 입력
    ----------
    recommendation_query : 검색 키워드 (예: "Python 백엔드", "AI 엔지니어")

    state 출력
    ----------
    job_candidates  : 수집·구조화된 공고 목록
    ranked_matches  : 적합도 상위 N개 랭킹 결과
    """
    logger.info("[job_recommender] 공고 추천 시작.")
    state["current_step"] = "job_recommender"
    errors: list[str] = state.get("errors") or []

    query: str = state.get("recommendation_query") or "백엔드 개발자"

    # ── 1. 프로필 벡터 DB 확인 ──────────────────────────────────────
    from rag.vectorstore import VectorStore
    vs = VectorStore()
    vs.initialize()
    if vs.count() == 0:
        errors.append("job_recommender: 프로필이 벡터 DB에 없습니다. 먼저 프로필을 등록하세요.")
        state["errors"] = errors
        state["ranked_matches"] = []
        return state

    # ── 2. 채용공고 수집 ─────────────────────────────────────────────
    candidates = _collect_postings(query, max_count=MAX_CANDIDATES)
    if not candidates:
        errors.append("job_recommender: 공고 검색 결과가 없습니다.")
        state["errors"] = errors
        state["job_candidates"] = []
        state["ranked_matches"] = []
        return state

    logger.info("[job_recommender] 수집된 공고 %d개 → 구조화 시작", len(candidates))

    # ── 3. Structured Output으로 공고 구조화 (병렬) ──────────────────
    candidates = _structure_postings(candidates)
    state["job_candidates"] = candidates

    # ── 4. 각 공고별 적합도 계산 (병렬) ─────────────────────────────
    ranked = _rank_candidates(candidates)

    # ── 5. 필터링·정렬·순위 부여 ─────────────────────────────────────
    ranked = [r for r in ranked if r["fit_score"] >= SCORE_THRESHOLD]
    ranked.sort(key=lambda x: x["fit_score"], reverse=True)
    for i, item in enumerate(ranked[:MAX_RANKED], 1):
        item["rank"] = i

    logger.info("[job_recommender] 최종 랭킹: %d개", len(ranked[:MAX_RANKED]))
    state["ranked_matches"] = ranked[:MAX_RANKED]
    state["errors"] = errors
    return state


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

def _collect_postings(query: str, max_count: int = MAX_CANDIDATES) -> list[dict[str, Any]]:
    """Tavily API로 채용공고 텍스트를 수집합니다."""
    try:
        from search.company_searcher import CompanySearcher
        searcher = CompanySearcher()
        result = searcher.search_job_postings(company_name=query, role="")
        items = result.get("results", [])[:max_count]

        postings = []
        for item in items:
            content = item.get("content", "").strip()
            if len(content) < 100:
                continue
            postings.append({
                "raw_text": content,
                "title":    item.get("title", ""),
                "company":  _extract_company(item.get("title", ""), item.get("url", "")),
                "url":      item.get("url", ""),
                "requirements": {},
            })
        return postings
    except Exception as exc:
        logger.error("[job_recommender] 공고 수집 실패: %s", exc)
        return []


def _extract_company(title: str, url: str) -> str:
    """공고 제목/URL에서 회사명을 간단히 추출합니다."""
    import re
    # URL 도메인에서 회사명 추출 시도
    domain_match = re.search(r"//(?:www\.)?([^./]+)", url)
    if domain_match:
        domain = domain_match.group(1)
        # 주요 채용 플랫폼은 제외
        if domain not in ("wanted", "saramin", "jobkorea", "linkedin", "jumpit", "programmers"):
            return domain.capitalize()
    # 제목에서 [] 또는 () 안의 회사명 추출
    bracket = re.search(r"[\[\(]([^\]\)]{2,20})[\]\)]", title)
    if bracket:
        return bracket.group(1)
    return title[:20] if title else "미확인"


def _structure_postings(candidates: list[dict]) -> list[dict]:
    """각 공고를 Structured Output으로 순차 구조화합니다."""
    from agents.job_parser import parse_posting_structured

    for item in candidates:
        try:
            item["requirements"] = parse_posting_structured(item["raw_text"])
        except Exception as exc:
            logger.warning("[job_recommender] 구조화 실패 (%s): %s", item.get("title", ""), exc)
            item["requirements"] = {}

    return candidates


def _rank_candidates(candidates: list[dict]) -> list[dict]:
    """각 공고에 대해 RAG 검색 + 적합도 계산을 순차 수행합니다.

    Chroma DB(SQLite)는 멀티스레드 환경에서 세그폴트 위험이 있어 순차 처리합니다.
    구조화(Structured Output)는 I/O 바운드이므로 병렬, 점수화는 순차.
    """
    from rag.profile_loader import ProfileLoader
    from agents.fit_analyzer import fit_analyzer_node

    loader = ProfileLoader()
    ranked = []

    for item in candidates:
        req = item.get("requirements", {})
        if not req:
            ranked.append({**item, "fit_score": 0.0, "fit_feedback": "구조화 실패"})
            continue
        try:
            skills_query = "기술 경험: " + ", ".join(req.get("required_skills", [])[:6])
            experiences = loader.search(skills_query, k=4)

            mock_state: GoodJobState = {
                "job_requirements":      req,
                "retrieved_experiences": experiences,
                "errors":                [],
            }
            result_state = fit_analyzer_node(mock_state)
            ranked.append({
                **item,
                "fit_score":    result_state.get("fit_score", 0.0),
                "fit_feedback": result_state.get("fit_feedback", ""),
            })
        except Exception as exc:
            logger.warning("[job_recommender] 점수 계산 오류 (%s): %s", item.get("title", ""), exc)
            ranked.append({**item, "fit_score": 0.0, "fit_feedback": f"오류: {exc}"})

    return ranked
