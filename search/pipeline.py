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
