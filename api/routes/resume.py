"""
api/routes/resume.py

Endpoints for generating and retrieving tailored resumes.

Routes
------
POST /resume/generate       – Kick off the LangGraph pipeline and return results.
GET  /resume/{session_id}   – Retrieve a previously generated resume by session ID.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from starlette.concurrency import run_in_threadpool
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

from api.session_store import session_store


# ------------------------------------------------------------------ #
# Request / Response models                                            #
# ------------------------------------------------------------------ #

class ResumeGenerateRequest(BaseModel):
    company_name: str = Field(..., description="Name of the target company.")
    job_posting_text: Optional[str] = Field(
        None,
        description="Raw job posting text (paste directly).",
    )
    job_url: Optional[str] = Field(
        None,
        description="URL of the job posting (scraped automatically if provided).",
    )
    generate_interview: bool = Field(
        True,
        description="Also generate interview questions + answer tips after the resume.",
    )


class ResumeGenerateResponse(BaseModel):
    session_id: str
    company_name: str
    fit_score: float
    fit_feedback: str
    matched_skills: list[str]
    missing_skills: list[str]
    resume_final: str
    resume_draft: str
    errors: list[str]
    current_step: str
    interview_questions: list[dict[str, Any]]


class ResumeGetResponse(BaseModel):
    session_id: str
    company_name: str
    fit_score: float
    fit_feedback: str
    resume_final: str
    interview_questions: list[dict[str, Any]] = []
    created_at: Optional[str] = None


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@router.post(
    "/generate",
    response_model=ResumeGenerateResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Generate a tailored resume via the LangGraph pipeline",
)
async def generate_resume(body: ResumeGenerateRequest) -> ResumeGenerateResponse:
    """
    Run the full GoodJob agent pipeline:

    1. Parse job requirements from the provided text or URL.
    2. Retrieve relevant profile experiences via RAG.
    3. Analyse fit score and gap analysis.
    4. Generate a tailored resume draft.
    5. Self-review and produce the final polished resume.

    Returns a session ID you can use to retrieve the result later.
    """
    logger.info("[resume/generate] Request for company: %s", body.company_name)

    # ---- Resolve job posting text ----
    job_posting_raw = body.job_posting_text or ""
    if not job_posting_raw and body.job_url:
        job_posting_raw = await _fetch_from_url(body.job_url)

    if not job_posting_raw.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=(
                f"Could not fetch job posting text from '{body.job_url}'."
                if body.job_url
                else "Provide either 'job_posting_text' or 'job_url'."
            ),
        )

    # ---- Build initial state ----
    from agents.state import GoodJobState
    initial_state: GoodJobState = {
        "company_name": body.company_name,
        "job_posting_raw": job_posting_raw,
        "messages": [],
        "errors": [],
        "current_step": "start",
        "generate_interview": body.generate_interview,
    }

    # ---- Run LangGraph pipeline ----
    try:
        from agents.graph import app as agent_app
        # LLM 5회 내외 호출 (수십 초) → 스레드풀에서 실행
        final_state: GoodJobState = await run_in_threadpool(agent_app.invoke, initial_state)
    except Exception as exc:  # noqa: BLE001
        logger.error("[resume/generate] Pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline failed: {exc}",
        ) from exc

    # ---- Persist to session store ----
    session_id = str(uuid.uuid4())
    record = {
        "session_id": session_id,
        "company_name": body.company_name,
        "fit_score": final_state.get("fit_score") or 0.0,
        "fit_feedback": final_state.get("fit_feedback") or "",
        "matched_skills": final_state.get("matched_skills") or [],
        "missing_skills": final_state.get("missing_skills") or [],
        "interview_questions": final_state.get("interview_questions") or [],
        "resume_final": final_state.get("resume_final") or "",
        "resume_draft": final_state.get("resume_draft") or "",
        "errors": final_state.get("errors") or [],
        "current_step": final_state.get("current_step") or "done",
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    session_store.save(session_id, record)
    logger.info("[resume/generate] Session created: %s (%s)", session_id, session_store.backend)

    return ResumeGenerateResponse(
        session_id=session_id,
        **{k: v for k, v in record.items() if k not in ("session_id", "created_at")},
    )


@router.get(
    "/{session_id}",
    response_model=ResumeGetResponse,
    status_code=status.HTTP_200_OK,
    summary="Retrieve a previously generated resume",
)
async def get_resume(session_id: str) -> ResumeGetResponse:
    """
    Fetch the results of a completed resume generation session by its ID.
    """
    record = session_store.get(session_id)
    if record is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Session '{session_id}' not found.",
        )

    return ResumeGetResponse(
        session_id=session_id,
        company_name=record["company_name"],
        fit_score=record["fit_score"],
        fit_feedback=record["fit_feedback"],
        resume_final=record["resume_final"],
        interview_questions=record.get("interview_questions", []),
        created_at=record.get("created_at"),
    )


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

async def _fetch_from_url(url: str) -> str:
    """
    채용공고 URL에서 본문 텍스트를 가져옵니다.

    1차: Playwright(JobScraper) — JS 렌더링 사이트(원티드 등) 대응
    2차: httpx + HTML 태그 제거 — Playwright 실패 시 폴백

    실패하면 빈 문자열을 반환합니다.
    """
    text = await run_in_threadpool(_scrape_with_playwright, url)
    if text.strip():
        return text

    logger.info("[resume/generate] Playwright 결과 없음 → httpx 폴백: %s", url)
    try:
        import httpx
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            return _html_to_text(response.text)[:5000]
    except Exception as exc:  # noqa: BLE001
        logger.warning("[resume/generate] Failed to fetch URL '%s': %s", url, exc)
        return ""


def _scrape_with_playwright(url: str) -> str:
    """
    별도 스레드에서 새 이벤트 루프로 Playwright 실행.

    uvicorn 이벤트 루프(Windows 에서는 Selector 루프일 수 있음)는
    Playwright 브라우저 서브프로세스를 띄우지 못하므로 분리 실행합니다.
    """
    import asyncio
    import sys

    from search.job_scraper import JobScraper

    try:
        if sys.platform == "win32":
            loop = asyncio.ProactorEventLoop()
        else:
            loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(JobScraper().fetch_job_detail(url))
        finally:
            loop.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("[resume/generate] Playwright 수집 실패 (%s): %s", url, exc)
        return ""


def _html_to_text(html: str) -> str:
    """script/style 을 제외한 HTML 본문 텍스트를 줄 단위로 추출합니다."""
    from html.parser import HTMLParser

    class _Extractor(HTMLParser):
        _SKIP = {"script", "style", "noscript", "svg", "head"}

        def __init__(self) -> None:
            super().__init__()
            self.parts: list[str] = []
            self._skip_depth = 0

        def handle_starttag(self, tag, attrs):
            if tag in self._SKIP:
                self._skip_depth += 1

        def handle_endtag(self, tag):
            if tag in self._SKIP and self._skip_depth:
                self._skip_depth -= 1

        def handle_data(self, data):
            if not self._skip_depth and data.strip():
                self.parts.append(data.strip())

    parser = _Extractor()
    parser.feed(html)
    return "\n".join(parser.parts)
