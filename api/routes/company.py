"""
api/routes/company.py

Endpoints for company and job posting discovery.

Routes
------
POST /company/search  – Search for a company and its open positions via Tavily.
POST /company/image   – Upload a screenshot; extract company name + posting text.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)
router = APIRouter()

# Accepted image MIME types
_ALLOWED_IMAGE_TYPES = {
    "image/png", "image/jpeg", "image/jpg",
    "image/gif", "image/webp",
}


# ------------------------------------------------------------------ #
# Request / Response models                                            #
# ------------------------------------------------------------------ #

class CompanySearchRequest(BaseModel):
    company_name: str = Field(..., description="Name of the company to research.")
    role: str = Field(
        default="engineer",
        description="Job role keyword (e.g. 'backend engineer', 'data scientist').",
    )


class CompanySearchResponse(BaseModel):
    company_name: str
    company_info: dict[str, Any]
    job_postings: dict[str, Any]


class ImageParseResponse(BaseModel):
    company_name: str
    job_title: str
    raw_text: str
    required_skills: list[str]
    location: str
    employment_type: str


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@router.post(
    "/search",
    response_model=CompanySearchResponse,
    status_code=status.HTTP_200_OK,
    summary="Search company info and job postings",
)
async def search_company(body: CompanySearchRequest) -> CompanySearchResponse:
    """
    Use Tavily to gather company background information and current job listings.

    - ``company_info`` : general overview, culture, tech stack
    - ``job_postings`` : current open positions
    """
    logger.info("[company/search] Searching for: %s (%s)", body.company_name, body.role)

    try:
        from search.company_searcher import CompanySearcher
        searcher = CompanySearcher()
        company_info = searcher.search_company_info(body.company_name)
        job_postings = searcher.search_job_postings(body.company_name, role=body.role)
    except Exception as exc:  # noqa: BLE001
        logger.error("[company/search] Error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_502_BAD_GATEWAY,
            detail=f"Search failed: {exc}",
        ) from exc

    return CompanySearchResponse(
        company_name=body.company_name,
        company_info=company_info,
        job_postings=job_postings,
    )


@router.post(
    "/image",
    response_model=ImageParseResponse,
    status_code=status.HTTP_200_OK,
    summary="Parse job posting from screenshot image",
)
async def parse_company_image(
    file: UploadFile = File(..., description="Screenshot of a job posting (PNG/JPEG/WEBP)."),
) -> ImageParseResponse:
    """
    Upload a job posting screenshot and use Claude Vision to extract:
    company name, job title, required skills, and raw text.
    """
    content_type = file.content_type or ""
    if content_type not in _ALLOWED_IMAGE_TYPES:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type '{content_type}'. Accepted: PNG, JPEG, WEBP, GIF.",
        )

    logger.info("[company/image] Received image: %s (%s)", file.filename, content_type)

    suffix = Path(file.filename or "upload.png").suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        from search.image_parser import ImageParser
        parser = ImageParser()
        result = parser.parse_job_posting_image(tmp_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("[company/image] Error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Image parsing failed: {exc}",
        ) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return ImageParseResponse(
        company_name=result.get("company_name", ""),
        job_title=result.get("job_title", ""),
        raw_text=result.get("raw_text", ""),
        required_skills=result.get("required_skills", []),
        location=result.get("location", ""),
        employment_type=result.get("employment_type", ""),
    )
