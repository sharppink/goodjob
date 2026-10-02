"""
search/job_scraper.py

Playwright로 채용공고 URL의 전체 내용을 가져옵니다.

주요 기능
---------
- fetch_job_detail(url)  : 어떤 채용공고 URL이든 전체 텍스트 추출
- scrape_wanted()        : 원티드 검색 결과 목록
- scrape_saramin()       : 사람인 검색 결과 목록
- scrape_all()           : 3개 사이트 동시 검색
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

logger = logging.getLogger(__name__)

NAV_TIMEOUT = 20_000
MAX_RESULTS = 10


class JobScraper:
    """
    Playwright 기반 채용사이트 스크래퍼.

    Usage
    -----
    >>> scraper = JobScraper()
    >>> # URL에서 채용공고 전체 텍스트 추출 (가장 많이 사용)
    >>> text = asyncio.run(scraper.fetch_job_detail("https://www.wanted.co.kr/..."))

    >>> # 회사명으로 공고 목록 검색
    >>> results = asyncio.run(scraper.scrape_wanted("카카오"))
    """

    # ------------------------------------------------------------------ #
    # 핵심: 채용공고 URL → 전체 텍스트
    # ------------------------------------------------------------------ #

    async def fetch_job_detail(self, url: str) -> str:
        """
        채용공고 URL에서 전체 텍스트를 추출합니다.

        원티드, 사람인, 잡코리아, 링크드인 등 어떤 URL이든 동작합니다.

        Parameters
        ----------
        url : str
            채용공고 URL.

        Returns
        -------
        str
            페이지의 주요 텍스트 (공고 내용).
        """
        logger.info("[JobScraper] 공고 상세 수집: %s", url)
        try:
            async with self._browser_context() as page:
                await page.goto(url, timeout=NAV_TIMEOUT)
                await page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)

                # 사이트별 주요 콘텐츠 셀렉터 시도
                selectors = [
                    # 원티드
                    "div.JobDescription_JobDescription__VWfcb",
                    # 사람인
                    "div.jv_detail",
                    # 잡코리아
                    "div.cont_duty",
                    # 링크드인
                    "div.job-view-layout",
                    # 일반 폴백
                    "main", "article", "div#content", "div.content",
                ]

                for selector in selectors:
                    el = await page.query_selector(selector)
                    if el:
                        text = await el.inner_text()
                        if len(text.strip()) > 100:
                            logger.info("[JobScraper] 셀렉터 '%s' 로 추출 성공", selector)
                            return text.strip()

                # 최후 폴백: body 전체 텍스트
                body = await page.query_selector("body")
                if body:
                    return (await body.inner_text())[:5000]
                return ""

        except Exception as exc:
            logger.error("[JobScraper] 상세 수집 실패 (%s): %s", url, exc)
            return ""

    # ------------------------------------------------------------------ #
    # 사이트별 목록 검색
    # ------------------------------------------------------------------ #

    async def scrape_wanted(self, company_name: str) -> list[dict[str, Any]]:
        """원티드에서 회사명으로 채용공고 목록을 수집합니다."""
        logger.info("[JobScraper] 원티드 검색: %s", company_name)
        url = f"https://www.wanted.co.kr/search?query={company_name}&tab=position"
        try:
            async with self._browser_context() as page:
                await page.goto(url, timeout=NAV_TIMEOUT)
                await page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
                return await self._parse_wanted(page)
        except Exception as exc:
            logger.error("[JobScraper] 원티드 오류: %s", exc)
            return []

    async def scrape_saramin(self, company_name: str) -> list[dict[str, Any]]:
        """사람인에서 회사명으로 채용공고 목록을 수집합니다."""
        logger.info("[JobScraper] 사람인 검색: %s", company_name)
        url = (
            f"https://www.saramin.co.kr/zf_user/search/recruit"
            f"?searchType=search&searchword={company_name}"
        )
        try:
            async with self._browser_context() as page:
                await page.goto(url, timeout=NAV_TIMEOUT)
                await page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
                return await self._parse_saramin(page)
        except Exception as exc:
            logger.error("[JobScraper] 사람인 오류: %s", exc)
            return []

    async def scrape_jobkorea(self, company_name: str) -> list[dict[str, Any]]:
        """잡코리아에서 회사명으로 채용공고 목록을 수집합니다."""
        logger.info("[JobScraper] 잡코리아 검색: %s", company_name)
        url = f"https://www.jobkorea.co.kr/Search/?stext={company_name}&tabType=recruit"
        try:
            async with self._browser_context() as page:
                await page.goto(url, timeout=NAV_TIMEOUT)
                await page.wait_for_load_state("networkidle", timeout=NAV_TIMEOUT)
                return await self._parse_jobkorea(page)
        except Exception as exc:
            logger.error("[JobScraper] 잡코리아 오류: %s", exc)
            return []

    async def scrape_all(self, company_name: str) -> dict[str, list[dict[str, Any]]]:
        """3개 사이트를 동시에 검색합니다."""
        wanted, saramin, jobkorea = await asyncio.gather(
            self.scrape_wanted(company_name),
            self.scrape_saramin(company_name),
            self.scrape_jobkorea(company_name),
            return_exceptions=True,
        )
        return {
            "wanted":   wanted   if not isinstance(wanted, Exception)   else [],
            "saramin":  saramin  if not isinstance(saramin, Exception)  else [],
            "jobkorea": jobkorea if not isinstance(jobkorea, Exception) else [],
        }

    # ------------------------------------------------------------------ #
    # 파서
    # ------------------------------------------------------------------ #

    @staticmethod
    async def _parse_wanted(page: Any) -> list[dict[str, Any]]:
        results = []
        try:
            # 원티드는 JS 렌더링 후 카드 로딩 대기
            await page.wait_for_selector("ul.List_List__FqS3v li", timeout=8000)
            cards = await page.query_selector_all("ul.List_List__FqS3v li")
            for card in cards[:MAX_RESULTS]:
                title_el  = await card.query_selector("strong")
                company_el = await card.query_selector("span.CompanyNameArea_CompanyNameArea__ImYEX")
                link_el   = await card.query_selector("a")
                title   = (await title_el.inner_text()).strip()   if title_el   else ""
                company = (await company_el.inner_text()).strip() if company_el else ""
                href    = await link_el.get_attribute("href")     if link_el    else ""
                if title:
                    results.append({
                        "title":   title,
                        "company": company,
                        "url":     f"https://www.wanted.co.kr{href}" if href else "",
                        "source":  "wanted",
                    })
        except Exception as exc:
            logger.warning("[JobScraper] 원티드 파싱 실패: %s", exc)
        return results

    @staticmethod
    async def _parse_saramin(page: Any) -> list[dict[str, Any]]:
        results = []
        try:
            items = await page.query_selector_all("div.item_recruit")
            for item in items[:MAX_RESULTS]:
                title_el    = await item.query_selector("h2.job_tit a")
                company_el  = await item.query_selector("strong.corp_name a")
                deadline_el = await item.query_selector("span.deadlines")
                title    = (await title_el.inner_text()).strip()    if title_el    else ""
                company  = (await company_el.inner_text()).strip()  if company_el  else ""
                href     = await title_el.get_attribute("href")     if title_el    else ""
                deadline = (await deadline_el.inner_text()).strip() if deadline_el else ""
                if title:
                    results.append({
                        "title":    title,
                        "company":  company,
                        "url":      f"https://www.saramin.co.kr{href}" if href else "",
                        "deadline": deadline,
                        "source":   "saramin",
                    })
        except Exception as exc:
            logger.warning("[JobScraper] 사람인 파싱 실패: %s", exc)
        return results

    @staticmethod
    async def _parse_jobkorea(page: Any) -> list[dict[str, Any]]:
        results = []
        try:
            items = await page.query_selector_all("div.list-post")
            for item in items[:MAX_RESULTS]:
                title_el    = await item.query_selector("a.title")
                company_el  = await item.query_selector("a.name")
                deadline_el = await item.query_selector("div.date em")
                title    = (await title_el.inner_text()).strip()    if title_el    else ""
                company  = (await company_el.inner_text()).strip()  if company_el  else ""
                href     = await title_el.get_attribute("href")     if title_el    else ""
                deadline = (await deadline_el.inner_text()).strip() if deadline_el else ""
                if title:
                    results.append({
                        "title":    title,
                        "company":  company,
                        "url":      f"https://www.jobkorea.co.kr{href}" if href else "",
                        "deadline": deadline,
                        "source":   "jobkorea",
                    })
        except Exception as exc:
            logger.warning("[JobScraper] 잡코리아 파싱 실패: %s", exc)
        return results

    # ------------------------------------------------------------------ #
    # 브라우저 컨텍스트
    # ------------------------------------------------------------------ #

    @staticmethod
    def _browser_context():
        from contextlib import asynccontextmanager

        @asynccontextmanager
        async def _ctx():
            from playwright.async_api import async_playwright
            async with async_playwright() as p:
                browser = await p.chromium.launch(headless=True)
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    locale="ko-KR",
                )
                page = await context.new_page()
                try:
                    yield page
                finally:
                    await browser.close()

        return _ctx()
