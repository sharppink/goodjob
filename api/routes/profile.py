"""
api/routes/profile.py

Endpoints for managing the user's profile data.

Routes
------
POST /profile/upload   – Upload a PDF resume and index it into the RAG store.
POST /profile/manual   – Submit profile data as a JSON body.
GET  /profile/status   – Return how many chunks are currently indexed.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Any, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile, status
from pydantic import BaseModel, Field

from rag.profile_loader import ProfileLoader
from rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)
router = APIRouter()


# ------------------------------------------------------------------ #
# Request / Response models                                            #
# ------------------------------------------------------------------ #

class ManualProfileRequest(BaseModel):
    """Schema for manually entering a user profile."""

    name: str = Field(..., description="Full name of the candidate.")
    email: Optional[str] = Field(None, description="Contact email address.")
    phone: Optional[str] = Field(None, description="Contact phone number.")
    summary: Optional[str] = Field(None, description="Short professional summary.")
    skills: list[str] = Field(default_factory=list, description="List of technical skills.")
    experience: Optional[str] = Field(
        None,
        description="Work experience (free-form text or structured).",
    )
    education: Optional[str] = Field(None, description="Education history.")
    projects: Optional[str] = Field(None, description="Notable projects.")
    certifications: list[str] = Field(default_factory=list, description="Certifications held.")
    languages: list[str] = Field(default_factory=list, description="Spoken languages.")


class ProfileUploadResponse(BaseModel):
    message: str
    chunks_stored: int


class ProfileStatusResponse(BaseModel):
    total_chunks: int
    status: str


# ------------------------------------------------------------------ #
# Endpoints                                                            #
# ------------------------------------------------------------------ #

@router.post(
    "/upload",
    response_model=ProfileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Upload a PDF resume",
)
async def upload_pdf_profile(
    file: UploadFile = File(..., description="PDF resume file."),
) -> ProfileUploadResponse:
    """
    Parse and index a PDF resume into the RAG vector store.

    - Accepts: ``multipart/form-data`` with a ``file`` field (PDF only).
    - Returns: Number of indexed text chunks.
    """
    if not file.filename or not file.filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are accepted.",
        )

    logger.info("[profile/upload] Received file: %s", file.filename)

    # Write the upload to a temp file (ProfileLoader needs a file path)
    with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        loader = ProfileLoader()
        chunks = loader.load_from_pdf(tmp_path)
    except Exception as exc:  # noqa: BLE001
        logger.error("[profile/upload] Error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process PDF: {exc}",
        ) from exc
    finally:
        Path(tmp_path).unlink(missing_ok=True)

    return ProfileUploadResponse(
        message="PDF profile indexed successfully.",
        chunks_stored=chunks,
    )


@router.post(
    "/manual",
    response_model=ProfileUploadResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Submit profile data manually (JSON)",
)
async def submit_manual_profile(body: ManualProfileRequest) -> ProfileUploadResponse:
    """
    Accept a structured profile as JSON and index it into the RAG vector store.
    """
    logger.info("[profile/manual] Received profile for: %s", body.name)

    try:
        loader = ProfileLoader()
        chunks = loader.load_from_dict(body.model_dump(exclude_none=True))
    except Exception as exc:  # noqa: BLE001
        logger.error("[profile/manual] Error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to process profile: {exc}",
        ) from exc

    return ProfileUploadResponse(
        message="Manual profile indexed successfully.",
        chunks_stored=chunks,
    )


@router.get(
    "/status",
    response_model=ProfileStatusResponse,
    summary="Get current RAG index status",
)
async def get_profile_status() -> ProfileStatusResponse:
    """Return the number of profile chunks currently indexed in the vector store."""
    try:
        vs = VectorStore()
        vs.initialize()
        count = vs.count()
        status_str = "ready" if count > 0 else "empty"
    except Exception as exc:  # noqa: BLE001
        logger.error("[profile/status] Error: %s", exc)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Could not reach vector store: {exc}",
        ) from exc

    return ProfileStatusResponse(total_chunks=count, status=status_str)
