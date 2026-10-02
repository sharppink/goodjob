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
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# In-memory session store (replace with Redis / DB in production)
_session_store: dict[str, dict[str, Any]] = {}


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


class ResumeGenerateResponse(BaseModel):
    session_id: str
    company_name: str
    fit_score: float
    fit_feedback: str
    resume_final: str
    resume_draft: str
    errors: list[str]
    current_step: str


class ResumeGetResponse(BaseModel):
    session_id: str
    company_name: str
    fit_score: float
    fit_feedback: str
    resume_final: str
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
            detail="Provide either 'job_posting_text' or 'job_url'.",
        )

    # ---- Build initial state ----
    from agents.state import GoodJobState
    initial_state: GoodJobState = {
        "company_name": body.company_name,
        "job_posting_raw": job_posting_raw,
        "messages": [],
        "errors": [],
        "current_step": "start",
    }

    # ---- Run LangGraph pipeline ----
    try:
        from agents.graph import app as agent_app
        final_state: GoodJobState = agent_app.invoke(initial_state)
    except Exception as exc:  # noqa: BLE001
        logger.error("[resume/generate] Pipeline error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Agent pipeline failed: {exc}",
        ) from exc

    # ---- Persist to session store ----
    session_id = str(uuid.uuid4())
    _session_store[session_id] = {
        "session_id": session_id,
        "company_name": body.company_name,
        "fit_score": final_state.get("fit_score") or 0.0,
        "fit_feedback": final_state.get("fit_feedback") or "",
        "resume_final": final_state.get("resume_final") or "",
        "resume_draft": final_state.get("resume_draft") or "",
        "errors": final_state.get("errors") or [],
        "current_step": final_state.get("current_step") or "done",
    }
    logger.info("[resume/generate] Session created: %s", session_id)

    return ResumeGenerateResponse(
        session_id=session_id,
        **{k: v for k, v in _session_store[session_id].items() if k != "session_id"},
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
    record = _session_store.get(session_id)
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
    )


# ------------------------------------------------------------------ #
# Internal helpers                                                     #
# ------------------------------------------------------------------ #

async def _fetch_from_url(url: str) -> str:
    """
    Attempt to fetch job posting text from a URL using httpx.

    Returns empty string on failure.
    """
    try:
        import httpx
        async with httpx.AsyncClient(follow_redirects=True, timeout=15.0) as client:
            response = await client.get(url)
            response.raise_for_status()
            # Very naive text extraction – real-world use should parse HTML
            return response.text
    except Exception as exc:  # noqa: BLE001
        logger.warning("[resume/generate] Failed to fetch URL '%s': %s", url, exc)
        return ""
