"""
frontend/pdf_exporter.py

Markdown 이력서 텍스트를 PDF 바이트로 변환합니다.
reportlab 사용, 한국어는 Windows 맑은고딕 / 나눔고딕 자동 감지.

Usage
-----
    from frontend.pdf_exporter import markdown_to_pdf_bytes
    pdf_bytes = markdown_to_pdf_bytes(markdown_text, company_name="카카오")
"""

from __future__ import annotations

import io
import os
import re
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    HRFlowable,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
)

# ------------------------------------------------------------------ #
# 폰트 등록                                                           #
# ------------------------------------------------------------------ #

_FONT_NAME      = "KoreanFont"
_FONT_BOLD_NAME = "KoreanFontBold"

_FONT_CANDIDATES = [
    # (일반, 볼드)
    (r"C:\Windows\Fonts\NanumGothic.ttf",  r"C:\Windows\Fonts\NanumGothicBold.ttf"),
    (r"C:\Windows\Fonts\malgun.ttf",        r"C:\Windows\Fonts\malgunbd.ttf"),
    (r"C:\Windows\Fonts\HANDotum.ttf",      r"C:\Windows\Fonts\HANDotumB.ttf"),
    (r"C:\Windows\Fonts\batang.ttc",        r"C:\Windows\Fonts\batang.ttc"),
]

_font_registered = False


def _register_fonts() -> None:
    global _font_registered
    if _font_registered:
        return

    for regular, bold in _FONT_CANDIDATES:
        if Path(regular).exists():
            try:
                pdfmetrics.registerFont(TTFont(_FONT_NAME, regular))
                bold_path = bold if Path(bold).exists() else regular
                pdfmetrics.registerFont(TTFont(_FONT_BOLD_NAME, bold_path))
                _font_registered = True
                return
            except Exception:
                continue

    # 폴백: 기본 Helvetica (한국어 깨지지만 오류는 없음)
    _font_registered = True


# ------------------------------------------------------------------ #
# 스타일 정의                                                          #
# ------------------------------------------------------------------ #

def _build_styles() -> dict:
    _register_fonts()
    f  = _FONT_NAME
    fb = _FONT_BOLD_NAME

    return {
        "h1": ParagraphStyle(
            "H1", fontName=fb, fontSize=20, leading=26,
            textColor=colors.HexColor("#1a1d2e"),
            alignment=TA_CENTER, spaceAfter=4,
        ),
        "h2": ParagraphStyle(
            "H2", fontName=fb, fontSize=13, leading=18,
            textColor=colors.HexColor("#1e40af"),
            spaceBefore=10, spaceAfter=3,
            borderPadding=(0, 0, 2, 0),
        ),
        "h3": ParagraphStyle(
            "H3", fontName=fb, fontSize=11, leading=16,
            textColor=colors.HexColor("#374151"),
            spaceBefore=6, spaceAfter=2,
        ),
        "body": ParagraphStyle(
            "Body", fontName=f, fontSize=10, leading=15,
            textColor=colors.HexColor("#374151"),
            spaceAfter=3,
        ),
        "bullet": ParagraphStyle(
            "Bullet", fontName=f, fontSize=10, leading=15,
            textColor=colors.HexColor("#374151"),
            leftIndent=12, spaceAfter=2,
            bulletIndent=4,
        ),
        "caption": ParagraphStyle(
            "Caption", fontName=f, fontSize=9, leading=13,
            textColor=colors.HexColor("#6b7280"),
            alignment=TA_CENTER, spaceAfter=8,
        ),
    }


# ------------------------------------------------------------------ #
# Markdown 파서 (경량)                                                 #
# ------------------------------------------------------------------ #

def _parse_markdown(md: str, styles: dict) -> list:
    """Markdown 텍스트를 reportlab Flowable 목록으로 변환합니다."""
    flowables = []
    lines = md.splitlines()
    i = 0

    while i < len(lines):
        line = lines[i]
        stripped = line.strip()

        # 빈 줄
        if not stripped:
            flowables.append(Spacer(1, 3))
            i += 1
            continue

        # 코드 펜스 (``` 로 감싼 블록) — 이력서엔 거의 없지만 방어
        if stripped.startswith("```"):
            i += 1
            while i < len(lines) and not lines[i].strip().startswith("```"):
                i += 1
            i += 1
            continue

        # H1  (#)
        if stripped.startswith("# ") and not stripped.startswith("## "):
            text = _inline(stripped[2:])
            flowables.append(Paragraph(text, styles["h1"]))
            flowables.append(HRFlowable(
                width="100%", thickness=1.5,
                color=colors.HexColor("#1e40af"), spaceAfter=6,
            ))
            i += 1
            continue

        # H2  (##)
        if stripped.startswith("## ") and not stripped.startswith("### "):
            text = _inline(stripped[3:])
            flowables.append(Spacer(1, 4))
            flowables.append(Paragraph(text, styles["h2"]))
            flowables.append(HRFlowable(
                width="100%", thickness=0.5,
                color=colors.HexColor("#bfdbfe"), spaceAfter=4,
            ))
            i += 1
            continue

        # H3  (###)
        if stripped.startswith("### "):
            text = _inline(stripped[4:])
            flowables.append(Paragraph(text, styles["h3"]))
            i += 1
            continue

        # 수평선  (--- 또는 ***)
        if re.match(r"^[-*]{3,}$", stripped):
            flowables.append(HRFlowable(
                width="100%", thickness=0.4,
                color=colors.HexColor("#e5e7eb"), spaceAfter=4,
            ))
            i += 1
            continue

        # 불릿 (- 또는 *)
        if stripped.startswith(("- ", "* ", "· ")):
            text = _inline(stripped[2:])
            flowables.append(Paragraph(f"• {text}", styles["bullet"]))
            i += 1
            continue

        # 들여쓴 불릿 (공백 2~4개 + -)
        if re.match(r"^\s{2,4}[-*] ", line):
            text = _inline(stripped[2:])
            flowables.append(Paragraph(f"  ◦ {text}", styles["bullet"]))
            i += 1
            continue

        # 일반 텍스트
        text = _inline(stripped)
        flowables.append(Paragraph(text, styles["body"]))
        i += 1

    return flowables


def _inline(text: str) -> str:
    """인라인 마크다운(볼드, 이탤릭, 코드)을 reportlab XML로 변환합니다."""
    # **bold**
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    # *italic* 또는 _italic_
    text = re.sub(r"\*(.+?)\*",   r"<i>\1</i>", text)
    text = re.sub(r"_(.+?)_",     r"<i>\1</i>", text)
    # `code`
    text = re.sub(r"`(.+?)`",     r"<font name='Courier'>\1</font>", text)
    # [링크](url) → 텍스트만 표시
    text = re.sub(r"\[(.+?)\]\(.+?\)", r"\1", text)
    # & → &amp; (XML 이스케이프)
    text = text.replace("&", "&amp;")
    return text


# ------------------------------------------------------------------ #
# 공개 API                                                             #
# ------------------------------------------------------------------ #

def markdown_to_pdf_bytes(
    markdown_text: str,
    company_name: str = "",
    author: str = "GoodJob",
) -> bytes:
    """
    Markdown 이력서 문자열을 PDF 바이트로 반환합니다.

    Parameters
    ----------
    markdown_text : str
        Markdown 형식의 이력서 텍스트.
    company_name : str
        PDF 파일명/메타데이터에 사용할 회사명.
    author : str
        PDF 문서 메타데이터 작성자.

    Returns
    -------
    bytes
        PDF 파일 바이트.
    """
    # 마크다운 코드펜스(```markdown ... ```)로 감싸진 경우 제거
    md = re.sub(r"^```(?:markdown)?\s*", "", markdown_text.strip())
    md = re.sub(r"\s*```$", "", md)

    styles = _build_styles()
    flowables = _parse_markdown(md, styles)

    # 하단 여백에 생성 정보
    if company_name:
        flowables.append(Spacer(1, 12))
        flowables.append(HRFlowable(
            width="100%", thickness=0.4,
            color=colors.HexColor("#e5e7eb"),
        ))
        flowables.append(Paragraph(
            f"Generated by GoodJob AI · {company_name} 지원용",
            styles["caption"],
        ))

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=18 * mm,
        title=f"이력서_{company_name}" if company_name else "이력서",
        author=author,
    )
    doc.build(flowables)
    return buf.getvalue()
