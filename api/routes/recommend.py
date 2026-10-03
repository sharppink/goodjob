"""
api/routes/recommend.py

Endpoint for reverse matching: find job postings that fit the stored profile.

Routes
------
POST /recommend  – Collect postings for a keyword via Tavily, score fit, return ranking.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

logger = logging.getLogger(__name__)
router = APIRouter()


class RecommendRequest(BaseModel):
    query: str = Field(..., description="Search keyword or role (e.g. 'Python 백엔드').")


class RecommendResponse(BaseModel):
    query: str
    ranked_matches: list[dict[str, Any]]
    errors: list[str]


@router.post(
    "",
    response_model=RecommendResponse,
    status_code=status.HTTP_200_OK,
    summary="Recommend job postings that fit the stored profile",
)
async def recommend_jobs(body: RecommendRequest) -> RecommendResponse:
    """
    Run ``job_recommender_node``: Tavily search → structure each posting →
    fit score against the stored profile → top matches above the fit threshold.

    Takes roughly 30–90 seconds (one LLM call per candidate posting).
    """
    logger.info("[recommend] query=%s", body.query)
    try:
        from agents.job_recommender import job_recommender_node
        state = await run_in_threadpool(
            job_recommender_node, {"recommendation_query": body.query, "errors": []}
        )
    except Exception as exc:  # noqa: BLE001
        logger.error("[recommend] Error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Recommendation failed: {exc}",
        ) from exc

    matches = [
        {
            "rank": m.get("rank"),
            "company": m.get("company", ""),
            "title": m.get("title", ""),
            "url": m.get("url", ""),
            "fit_score": m.get("fit_score", 0.0),
            "fit_feedback": m.get("fit_feedback", ""),
            "job_title": (m.get("requirements") or {}).get("job_title", ""),
        }
        for m in state.get("ranked_matches") or []
    ]
    return RecommendResponse(query=body.query, ranked_matches=matches, errors=state.get("errors") or [])
