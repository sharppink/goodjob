"""
search/image_parser.py

OpenAI Vision(gpt-4o)으로 이미지에서 회사명과 채용공고 정보를 추출합니다.

지원 이미지
-----------
- 회사 로고 → 회사명 추출
- 채용공고 스크린샷 → 구조화된 공고 정보 추출
- 명함, 배너 등
"""

from __future__ import annotations

import base64
import json
import logging
import re
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class ImageParser:
    """
    이미지에서 회사명 및 채용 정보를 추출합니다.

    Usage
    -----
    >>> parser = ImageParser()
    >>> company = parser.extract_company_name("screenshot.png")
    >>> posting = parser.parse_job_posting_image("job_posting.png")
    """

    def __init__(self) -> None:
        from llm.openai_client import OpenAIClient
        self._client = OpenAIClient()

    # ------------------------------------------------------------------ #
    # Public methods                                                       #
    # ------------------------------------------------------------------ #

    def extract_company_name(self, image_path: str | Path) -> str:
        """
        이미지에서 회사명만 추출합니다.

        Parameters
        ----------
        image_path : str | Path
            이미지 파일 경로 (PNG, JPEG, WEBP, GIF).

        Returns
        -------
        str
            추출된 회사명. 찾지 못하면 빈 문자열.
        """
        logger.info("[ImageParser] 회사명 추출: %s", image_path)
        prompt = (
            "이 이미지를 보고 회사명만 반환해줘. "
            "다른 설명 없이 회사명 텍스트만 출력해. "
            "회사명을 찾을 수 없으면 'UNKNOWN'을 반환해."
        )
        result = self._client.generate_with_vision(prompt=prompt, image_path=str(image_path))
        company = result.strip()
        if company == "UNKNOWN":
            return ""
        return company

    def parse_job_posting_image(self, image_path: str | Path) -> dict[str, Any]:
        """
        채용공고 이미지를 파싱하여 구조화된 정보를 반환합니다.

        Parameters
        ----------
        image_path : str | Path
            채용공고 이미지 파일 경로.

        Returns
        -------
        dict
            Keys: company_name, job_title, raw_text,
                  required_skills, preferred_skills,
                  responsibilities, location, employment_type
        """
        logger.info("[ImageParser] 채용공고 이미지 파싱: %s", image_path)
        prompt = (
            "이 채용공고 이미지의 모든 텍스트를 읽고 아래 JSON 형식으로 추출해줘.\n"
            "다른 설명 없이 JSON만 반환해.\n\n"
            "{\n"
            '  "company_name": "회사명",\n'
            '  "job_title": "직무명",\n'
            '  "raw_text": "이미지의 전체 텍스트",\n'
            '  "required_skills": ["필수기술1", "필수기술2"],\n'
            '  "preferred_skills": ["우대조건1", "우대조건2"],\n'
            '  "responsibilities": ["주요업무1", "주요업무2"],\n'
            '  "location": "근무지",\n'
            '  "employment_type": "정규직/계약직/인턴"\n'
            "}"
        )
        raw = self._client.generate_with_vision(prompt=prompt, image_path=str(image_path))
        return self._safe_parse_json(raw)

    def extract_text_from_image(self, image_path: str | Path) -> str:
        """
        이미지에서 텍스트만 추출합니다 (OCR 대용).

        Returns
        -------
        str
            이미지에서 읽은 전체 텍스트.
        """
        logger.info("[ImageParser] 이미지 텍스트 추출: %s", image_path)
        prompt = "이 이미지에 있는 모든 텍스트를 그대로 추출해줘. 형식은 유지하고 추가 설명은 하지 마."
        return self._client.generate_with_vision(prompt=prompt, image_path=str(image_path))

    # ------------------------------------------------------------------ #
    # Internal                                                             #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _safe_parse_json(raw: str) -> dict[str, Any]:
        """JSON 파싱 시도 후 실패하면 기본값 반환."""
        # ```json ... ``` 블록 제거
        clean = re.sub(r"```(?:json)?", "", raw).strip().rstrip("`").strip()
        try:
            return json.loads(clean)
        except (json.JSONDecodeError, TypeError):
            return {
                "company_name": "",
                "job_title": "",
                "raw_text": raw,
                "required_skills": [],
                "preferred_skills": [],
                "responsibilities": [],
                "location": "",
                "employment_type": "",
            }
