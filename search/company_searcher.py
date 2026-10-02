"""
search/company_searcher.py

Tavily API로 회사 정보 및 채용공고를 검색하고
OpenAI로 핵심 내용을 요약합니다.
"""

from __future__ import annotations

import logging
from typing import Any, Optional

from config.settings import settings

logger = logging.getLogger(__name__)


class CompanySearcher:
    """
    회사 정보와 채용공고를 검색합니다.

    Usage
    -----
    >>> searcher = CompanySearcher()
    >>> result = searcher.search_all("카카오", role="백엔드 개발자")
    >>> print(result["summary"])
    """

    def __init__(self, api_key: Optional[str] = None) -> None:
        self._api_key = api_key or settings.TAVILY_API_KEY
        self._client: Optional[object] = None

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    def search_company_info(self, company_name: str) -> dict[str, Any]:
        """회사 문화, 기술 스택, 팀 정보를 검색합니다."""
        queries = [
            f"{company_name} 개발팀 기술스택 개발문화 2024",
            f"{company_name} engineering culture tech stack",
        ]
        results = []
        for q in queries:
            results.extend(self._search(q, max_results=3))

        aggregated = self._aggregate(results, context="company_info")
        aggregated["summary"] = self._summarize(
            aggregated["combined_text"],
            prompt_hint="회사의 기술 스택, 개발 문화, 팀 규모를 중심으로 요약해줘.",
        )
        return aggregated

    def search_job_postings(self, company_name: str, role: str = "개발자") -> dict[str, Any]:
        """특정 회사의 채용공고를 검색합니다."""
        queries = [
            f"{company_name} {role} 채용 공고 2024 site:wanted.co.kr OR site:saramin.co.kr",
            f"{company_name} {role} job posting 2024",
        ]
        results = []
        for q in queries:
            results.extend(self._search(q, max_results=4))

        # 중복 URL 제거
        seen_urls: set[str] = set()
        unique = []
        for r in results:
            if r.get("url") not in seen_urls:
                seen_urls.add(r.get("url", ""))
                unique.append(r)

        aggregated = self._aggregate(unique, context="job_postings")
        aggregated["summary"] = self._summarize(
            aggregated["combined_text"],
            prompt_hint="채용공고에서 필수 기술, 우대 조건, 주요 업무를 항목별로 요약해줘.",
        )
        return aggregated

    def fetch_url_content(self, url: str) -> str:
        """
        특정 채용공고 URL의 전체 내용을 가져옵니다.

        Tavily extract API를 사용해 URL 본문을 추출합니다.
        """
        try:
            client = self._get_client()
            response = client.extract(urls=[url])  # type: ignore[union-attr]
            results = response.get("results", [])
            if results:
                return results[0].get("raw_content", "")
            return ""
        except Exception as exc:
            logger.error("[CompanySearcher] URL 추출 실패 (%s): %s", url, exc)
            return ""

    def search_all(self, company_name: str, role: str = "개발자") -> dict[str, Any]:
        """
        회사 정보와 채용공고를 한 번에 검색하고 통합 요약을 생성합니다.

        Returns
        -------
        dict
            Keys: company_info, job_postings, combined_text, summary, urls
        """
        logger.info("[CompanySearcher] 통합 검색 시작: %s / %s", company_name, role)
        company_info = self.search_company_info(company_name)
        job_postings = self.search_job_postings(company_name, role)

        combined = f"{company_info['combined_text']}\n\n{job_postings['combined_text']}"
        urls = [r["url"] for r in job_postings["results"] if r.get("url")]

        overall_summary = self._summarize(
            combined,
            prompt_hint=(
                f"{company_name}의 {role} 채용 관련 핵심 정보를 정리해줘: "
                "1) 회사 특징 2) 필수 기술 3) 우대 조건 4) 주요 업무"
            ),
        )

        return {
            "company_name": company_name,
            "role": role,
            "company_info": company_info,
            "job_postings": job_postings,
            "combined_text": combined,
            "summary": overall_summary,
            "urls": urls,
        }

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    def _get_client(self) -> object:
        if self._client is None:
            from tavily import TavilyClient
            self._client = TavilyClient(api_key=self._api_key)
        return self._client

    def _search(self, query: str, max_results: int = 5) -> list[dict[str, Any]]:
        try:
            client = self._get_client()
            response = client.search(  # type: ignore[union-attr]
                query=query,
                max_results=max_results,
                search_depth="advanced",
            )
            return response.get("results", [])
        except Exception as exc:
            logger.error("[CompanySearcher] 검색 실패 ('%s'): %s", query, exc)
            return []

    def _aggregate(
        self, results: list[dict[str, Any]], context: str = "general"
    ) -> dict[str, Any]:
        cleaned = []
        texts = []
        for item in results:
            title = item.get("title", "")
            url = item.get("url", "")
            content = item.get("content", "")
            score = item.get("score", 0.0)
            cleaned.append({"title": title, "url": url, "content": content, "score": score})
            if content:
                texts.append(f"[{title}]\n{content}")

        return {
            "context": context,
            "count": len(cleaned),
            "results": cleaned,
            "combined_text": "\n\n".join(texts),
            "summary": "",
        }

    @staticmethod
    def _summarize(text: str, prompt_hint: str = "") -> str:
        """OpenAI로 검색 결과를 요약합니다."""
        if not text.strip():
            return ""
        try:
            from llm.openai_client import OpenAIClient, FAST_MODEL
            client = OpenAIClient(model=FAST_MODEL)
            prompt = (
                f"다음은 채용 관련 검색 결과입니다.\n\n{text[:3000]}\n\n"
                f"지시사항: {prompt_hint}"
            )
            return client.generate(prompt, max_tokens=600)
        except Exception as exc:
            logger.warning("[CompanySearcher] 요약 생성 실패: %s", exc)
            return text[:500]
