"""
rag/profile_loader.py

사용자 프로필을 여러 소스에서 불러와 RAG 벡터 저장소에 인덱싱합니다.

지원 입력
---------
- PDF 파일    : PyMuPDF로 텍스트 추출
- 딕셔너리   : 직접 입력 폼 (이름, 기술, 경력 등)

청킹 전략
---------
일반 문자 분할 대신 이력서 섹션 기반으로 청킹합니다.
("경력", "기술", "학력", "프로젝트" 등의 헤더를 기준으로 분할)
섹션이 없으면 문장 경계 기반 오버랩 청킹으로 폴백합니다.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any

from rag.vectorstore import VectorStore

logger = logging.getLogger(__name__)

CHUNK_SIZE = 600
CHUNK_OVERLAP = 80

# 한국어/영어 이력서 섹션 헤더 패턴
SECTION_HEADERS = re.compile(
    r"^(?:"
    r"경[력험]|학력|기술|스킬|skill|프로젝트|project|자격[증]?|certificate"
    r"|수상|award|활동|activity|자기소개|introduction|summary|education"
    r"|experience|work|employment|language|외국어|봉사|volunteer"
    r").*$",
    re.IGNORECASE | re.MULTILINE,
)


class ProfileLoader:
    """
    사용자 프로필을 RAG 벡터 저장소에 로드합니다.

    Usage
    -----
    >>> loader = ProfileLoader()
    >>> loader.load_from_pdf("resume.pdf")

    >>> loader.load_from_dict({
    ...     "이름": "홍길동",
    ...     "기술": ["Python", "FastAPI", "LangGraph"],
    ...     "경력": "카카오 백엔드 3년",
    ... })
    """

    def __init__(self, vectorstore: VectorStore | None = None) -> None:
        # vectorstore 를 넘기면 그 저장소를 사용 (평가 스크립트의 임시 DB 등)
        self._vectorstore = vectorstore or VectorStore()
        self._vectorstore.initialize()

    # ------------------------------------------------------------------ #
    # Public loaders                                                       #
    # ------------------------------------------------------------------ #

    def load_from_pdf(self, file_path: str | Path, clear_existing: bool = False) -> int:
        """
        PDF 이력서를 파싱하여 벡터 저장소에 인덱싱합니다.

        Parameters
        ----------
        file_path : str | Path
            PDF 파일 경로.
        clear_existing : bool
            True면 기존 프로필 데이터를 모두 삭제 후 재인덱싱.

        Returns
        -------
        int
            저장된 청크 수.
        """
        file_path = Path(file_path)
        if not file_path.exists():
            raise FileNotFoundError(f"PDF를 찾을 수 없습니다: {file_path}")

        if clear_existing:
            self.clear_profile()

        logger.info("[ProfileLoader] PDF 파싱 시작: %s", file_path)
        text = self._extract_pdf_text(file_path)

        if not text.strip():
            logger.warning("[ProfileLoader] PDF에서 텍스트를 추출할 수 없습니다.")
            return 0

        return self.chunk_and_store(
            text,
            metadata={"source": str(file_path), "type": "pdf", "filename": file_path.name},
        )

    def load_from_dict(self, data: dict[str, Any], clear_existing: bool = False) -> int:
        """
        딕셔너리 형태의 프로필을 벡터 저장소에 인덱싱합니다.

        Parameters
        ----------
        data : dict
            프로필 필드 (이름, 기술, 경력, 학력, 프로젝트 등 자유 형식).
        clear_existing : bool
            True면 기존 데이터를 삭제 후 재인덱싱.

        Returns
        -------
        int
            저장된 청크 수.
        """
        if clear_existing:
            self.clear_profile()

        logger.info("[ProfileLoader] 딕셔너리에서 프로필 로드.")
        text = self._dict_to_text(data)
        return self.chunk_and_store(
            text,
            metadata={"source": "manual_input", "type": "dict"},
        )

    def load_star_experiences(self, stars: list[dict[str, Any]]) -> int:
        """
        STAR 구조로 변환된 경험을 기존 프로필에 추가합니다 (기존 데이터 유지).

        경험 하나를 청크 하나로 저장해 상황·행동·결과가 함께 검색되도록 합니다.

        Returns
        -------
        int
            저장된 청크 수.
        """
        from agents.star_converter import star_to_text

        docs = [star_to_text(s) for s in stars if s.get("title")]
        if not docs:
            return 0
        metas = [{"source": "star", "type": "star", "section": "STAR 경험"} for _ in docs]
        self._vectorstore.add_documents(docs, metadatas=metas)
        logger.info("[ProfileLoader] STAR 경험 %d개 저장 완료.", len(docs))
        return len(docs)

    def chunk_and_store(self, text: str, metadata: dict[str, Any] | None = None) -> int:
        """
        텍스트를 청킹하여 벡터 저장소에 저장합니다.

        섹션 헤더가 감지되면 섹션 단위로 분할하고,
        없으면 오버랩 문자 분할을 사용합니다.

        Returns
        -------
        int
            저장된 청크 수.
        """
        chunks, chunk_metas = self._smart_split(text, base_metadata=metadata or {})

        if not chunks:
            logger.warning("[ProfileLoader] 청크가 생성되지 않았습니다.")
            return 0

        self._vectorstore.add_documents(chunks, metadatas=chunk_metas)
        logger.info("[ProfileLoader] %d개 청크 저장 완료.", len(chunks))
        return len(chunks)

    def clear_profile(self) -> None:
        """벡터 저장소의 모든 프로필 데이터를 삭제합니다."""
        logger.info("[ProfileLoader] 기존 프로필 데이터 삭제.")
        self._vectorstore.delete_all()

    def get_stored_count(self) -> int:
        """현재 저장된 청크 수를 반환합니다."""
        return self._vectorstore.count()

    def search(self, query: str, k: int = 5) -> list[str]:
        """
        저장된 프로필에서 쿼리와 유사한 경험을 검색합니다.

        Parameters
        ----------
        query : str
            검색할 키워드 또는 문장.
        k : int
            반환할 결과 수.
        """
        from config.settings import settings
        from rag.reranker import Reranker

        # 1단계: 벡터 검색 (후보 수집)
        initial_k = settings.RAG_TOP_K_INITIAL if settings.RERANKER_ENABLED else k
        candidates = self._vectorstore.similarity_search(query, k=initial_k)

        if not candidates:
            return []

        # 2단계: Reranker로 재정렬 (활성화된 경우)
        if settings.RERANKER_ENABLED and len(candidates) > k:
            reranker = Reranker()
            return reranker.rerank(query, candidates, top_k=k)

        return candidates[:k]

    # ------------------------------------------------------------------ #
    # Internal helpers                                                     #
    # ------------------------------------------------------------------ #

    def _smart_split(
        self,
        text: str,
        base_metadata: dict[str, Any],
    ) -> tuple[list[str], list[dict]]:
        """
        섹션 헤더 기반 분할을 시도하고, 없으면 오버랩 분할로 폴백합니다.
        """
        sections = self._split_by_sections(text)

        if len(sections) >= 2:
            logger.info("[ProfileLoader] 섹션 기반 청킹 적용 (%d개 섹션)", len(sections))
            chunks, metas = [], []
            for section_name, section_text in sections:
                sub_chunks = self._overlap_split(section_text)
                for chunk in sub_chunks:
                    chunks.append(chunk)
                    metas.append({**base_metadata, "section": section_name})
            return chunks, metas

        # 섹션 없음 → 오버랩 분할
        logger.info("[ProfileLoader] 오버랩 분할 적용.")
        chunks = self._overlap_split(text)
        metas = [base_metadata] * len(chunks)
        return chunks, metas

    @staticmethod
    def _split_by_sections(text: str) -> list[tuple[str, str]]:
        """
        섹션 헤더를 기준으로 텍스트를 분할합니다.

        Returns
        -------
        list of (section_name, section_text) tuples.
        """
        lines = text.split("\n")
        sections: list[tuple[str, str]] = []
        current_header = "intro"
        current_lines: list[str] = []

        for line in lines:
            if SECTION_HEADERS.match(line.strip()):
                if current_lines:
                    body = "\n".join(current_lines).strip()
                    if body:
                        sections.append((current_header, body))
                # 섹션 이름은 "경력: 스타트업 A ..." 의 앞부분만 사용
                current_header = line.strip().split(":", 1)[0][:30]
                # 헤더 줄도 본문에 포함 — "Experience: 스타트업 A ..." 처럼
                # 헤더와 내용이 한 줄에 있는 경우 내용이 버려지는 버그 방지
                current_lines = [line]
            else:
                current_lines.append(line)

        # 마지막 섹션 추가
        if current_lines:
            body = "\n".join(current_lines).strip()
            if body:
                sections.append((current_header, body))

        return sections

    @staticmethod
    def _overlap_split(
        text: str,
        chunk_size: int = CHUNK_SIZE,
        overlap: int = CHUNK_OVERLAP,
    ) -> list[str]:
        """문장 경계를 고려한 오버랩 문자 분할."""
        if not text.strip():
            return []
        if len(text) <= chunk_size:
            return [text.strip()]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            if end < len(text):
                boundary = text.rfind(". ", start, end)
                if boundary != -1 and boundary > start + overlap:
                    end = boundary + 1
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            start = end - overlap

        return chunks

    @staticmethod
    def _extract_pdf_text(file_path: Path) -> str:
        """PyMuPDF(fitz)로 PDF 전체 텍스트 추출."""
        try:
            import fitz  # PyMuPDF
        except ImportError as exc:
            raise ImportError("PyMuPDF가 필요합니다: pip install pymupdf") from exc

        pages: list[str] = []
        with fitz.open(str(file_path)) as doc:
            for page in doc:
                pages.append(page.get_text())

        return "\n\n".join(pages)

    @staticmethod
    def _dict_to_text(data: dict[str, Any]) -> str:
        """딕셔너리를 이력서 형식의 텍스트로 변환합니다."""
        lines: list[str] = []
        for key, value in data.items():
            # 빈 값(빈 문자열/리스트/딕셔너리, None)은 의미 없는 청크가 되므로 제외
            if value is None or (isinstance(value, (str, list, dict)) and not value):
                continue
            heading = key.replace("_", " ").title()
            if isinstance(value, list):
                # 리스트: 각 항목을 줄바꿈으로 나열
                items = "\n  - ".join(str(v) for v in value)
                lines.append(f"{heading}:\n  - {items}")
            elif isinstance(value, dict):
                # 중첩 딕셔너리: 재귀적으로 변환
                sub_lines = []
                for k, v in value.items():
                    sub_lines.append(f"  {k}: {v}")
                lines.append(f"{heading}:\n" + "\n".join(sub_lines))
            else:
                lines.append(f"{heading}: {value}")
        return "\n\n".join(lines)
