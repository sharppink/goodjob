"""
search/pipeline.py

회사 입력(텍스트/이미지) → 채용공고 수집의 통합 진입점.

사용 흐름
---------
1. 회사명 텍스트 입력  OR  이미지 업로드
2. 이미지면 Vision으로 회사명 추출
3. Tavily로 회사 정보 + 채용공고 검색
4. 채용공고 URL이 있으면 Playwright로 상세 내용 수집
5. 최종 job_posting_text 반환 → LangGraph 파이프라인으로 전달
"""

from __future__ import annotations

import asyncio
import logging
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

MIN_POSTING_CHARS = 200   # 이보다 짧은 검색 결과는 공고 본문으로 보기 어려움
MAX_POSTING_CHECKS = 5    # 공고 여부를 LLM 으로 판정할 최대 검색 결과 수


def find_job_postings(
    company_name: str, role: str = "개발자", max_checks: int = MAX_POSTING_CHECKS
) -> dict[str, Any]:
    """
    회사·직군으로 검색한 페이지 중 개별 채용공고로 판정된 것만 반환합니다 (PROJECT_DOCS #028).

    예전 자동 검색은 검색 결과 여러 개(블로그 기사·Q&A 포함)를 합친 텍스트를 공고로 넘겼음.
    이제는 페이지마다 목록 페이지·짧은 글을 먼저 거르고, job_parser 의 is_job_posting 판정을
    통과한 페이지만 후보로 돌려줍니다. 입력한 회사의 공고가 앞에 옵니다.

    Returns
    -------
    dict
        postings : [{title, url, raw_text, requirements}]  — 공고로 판정된 페이지
        checked  : 공고 여부를 판정한 페이지 수
        found    : 검색된 페이지 수
    """
    from agents.job_parser import parse_posting_structured
    from agents.job_recommender import _is_listing_title, _is_listing_url
    from search.company_searcher import CompanySearcher

    results = CompanySearcher().search_job_posting_results(company_name, role)
    candidates = [
        r for r in results
        if len((r.get("content") or "").strip()) >= MIN_POSTING_CHARS
        and not _is_listing_url(r.get("url", ""))
        and not _is_listing_title(r.get("title", ""))
    ][:max_checks]

    postings = []
    for r in candidates:
        title = r.get("title", "")
        raw_text = f"{title}\n\n{r['content'].strip()}" if title else r["content"].strip()
        req = parse_posting_structured(raw_text)
        if not req.get("is_job_posting"):
            logger.info("[find_job_postings] 공고 아님 제외: %s", r.get("url", ""))
            continue
        postings.append({"title": title, "url": r.get("url", ""), "raw_text": raw_text, "requirements": req})

    # 입력한 회사의 공고를 앞으로 (검색 결과에 다른 회사 공고가 섞이는 경우가 많음)
    postings.sort(key=lambda p: not _same_company(company_name, p["requirements"].get("company_name", "")))
    return {"postings": postings, "checked": len(candidates), "found": len(results)}


def _same_company(a: str, b: str) -> bool:
    import re
    norm = lambda s: re.sub(r"\(주\)|㈜|주식회사|[\W_]+", "", s or "").lower()  # noqa: E731
    a, b = norm(a), norm(b)
    return bool(a and b and (a in b or b in a))


class JobSearchPipeline:
    """
    회사명 또는 이미지를 받아 채용공고 텍스트를 반환합니다.

    Usage
    -----
    >>> pipeline = JobSearchPipeline()

    >>> # 텍스트 입력
    >>> result = pipeline.run(company_name="카카오", role="백엔드 개발자")

    >>> # 이미지 입력
    >>> result = pipeline.run(image_path="screenshot.png", role="데이터 엔지니어")

    >>> print(result["job_posting_text"])  # LangGraph에 넘길 공고 텍스트
    """

    def __init__(self) -> None:
        from search.company_searcher import CompanySearcher
        from search.image_parser import ImageParser
        from search.job_scraper import JobScraper
        self._searcher = CompanySearcher()
        self._parser = ImageParser()
        self._scraper = JobScraper()

    # ------------------------------------------------------------------ #
    # 통합 실행                                                            #
    # ------------------------------------------------------------------ #

    def run(
        self,
        company_name: Optional[str] = None,
        image_path: Optional[str | Path] = None,
        role: str = "개발자",
        fetch_detail: bool = True,
    ) -> dict[str, Any]:
        """
        채용공고 수집 파이프라인을 실행합니다.

        Parameters
        ----------
        company_name : str, optional
            회사명 텍스트.
        image_path : str | Path, optional
            회사 로고 또는 채용공고 이미지 경로.
        role : str
            검색할 직무 (예: "백엔드 개발자", "ML 엔지니어").
        fetch_detail : bool
            True면 첫 번째 공고 URL의 상세 내용도 수집.

        Returns
        -------
        dict
            Keys: company_name, role, summary, job_posting_text, urls, raw
        """
        # 1단계: 회사명 확보
        if not company_name and image_path:
            logger.info("[Pipeline] 이미지에서 회사명 추출 중...")
            company_name = self._parser.extract_company_name(image_path)
            logger.info("[Pipeline] 추출된 회사명: '%s'", company_name)

        if not company_name:
            return {"error": "회사명을 입력하거나 이미지를 업로드해주세요."}

        # 2단계: 이미지가 채용공고 자체인 경우 → 바로 파싱
        image_posting_text = ""
        if image_path:
            logger.info("[Pipeline] 이미지 채용공고 파싱 중...")
            parsed = self._parser.parse_job_posting_image(image_path)
            image_posting_text = parsed.get("raw_text", "")

        # 3단계: Tavily 검색
        logger.info("[Pipeline] Tavily 검색 중: %s / %s", company_name, role)
        search_result = self._searcher.search_all(company_name, role)

        # 4단계: 첫 번째 공고 URL 상세 수집 (Playwright)
        detail_text = ""
        urls = search_result.get("urls", [])
        if fetch_detail and urls:
            first_url = urls[0]
            logger.info("[Pipeline] 공고 상세 수집: %s", first_url)
            try:
                detail_text = asyncio.run(self._scraper.fetch_job_detail(first_url))
            except Exception as exc:
                logger.warning("[Pipeline] 상세 수집 실패: %s", exc)

        # 5단계: 최종 공고 텍스트 조합
        parts = []
        if image_posting_text:
            parts.append(f"[이미지 채용공고]\n{image_posting_text}")
        if detail_text:
            parts.append(f"[웹 채용공고 상세]\n{detail_text[:3000]}")
        if search_result.get("combined_text"):
            parts.append(f"[검색 결과 요약]\n{search_result['combined_text'][:2000]}")

        job_posting_text = "\n\n---\n\n".join(parts) if parts else search_result.get("summary", "")

        return {
            "company_name":     company_name,
            "role":             role,
            "summary":          search_result.get("summary", ""),
            "job_posting_text": job_posting_text,
            "urls":             urls,
            "raw":              search_result,
        }
