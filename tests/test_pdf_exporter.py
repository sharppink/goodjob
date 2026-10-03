"""frontend/pdf_exporter.py — Markdown 이력서 → PDF."""

from __future__ import annotations

import pytest

pdf_exporter = pytest.importorskip("frontend.pdf_exporter")

RESUME_MD = "# 이력서\n## 경력\n- **Python** 백엔드 4년\n- `FastAPI` 서버 개발\n\n---\n일반 문단"


@pytest.fixture(autouse=True)
def reset_font_cache(monkeypatch):
    monkeypatch.setattr(pdf_exporter, "_registered_fonts", None)


def test_pdf_without_korean_font_falls_back_to_helvetica(monkeypatch):
    """한국어 폰트가 없는 환경(Linux 컨테이너 등)에서도 PDF 생성이 실패하면 안 됨 (#023)."""
    monkeypatch.setattr(pdf_exporter, "_FONT_CANDIDATES", [("/nonexistent.ttf", "/nonexistent.ttf")])
    pdf = pdf_exporter.markdown_to_pdf_bytes(RESUME_MD, company_name="테스트컴퍼니")
    assert pdf.startswith(b"%PDF-")
    assert pdf_exporter._registered_fonts == ("Helvetica", "Helvetica-Bold")


def test_pdf_with_available_font():
    pdf = pdf_exporter.markdown_to_pdf_bytes(RESUME_MD, company_name="테스트컴퍼니")
    assert pdf.startswith(b"%PDF-")
    assert pdf_exporter._registered_fonts is not None
