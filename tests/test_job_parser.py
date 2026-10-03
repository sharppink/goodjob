"""agents/job_parser.py — 공고 정제·정규화·로컬 응답 파싱·노드 분기."""

from __future__ import annotations

import json

import pytest

from agents import job_parser
from agents.job_parser import (
    MAX_POSTING_CHARS,
    clean_posting_text,
    job_parser_node,
    normalize_remote_policy,
    normalize_requirements,
    parse_local_response,
)
from conftest import SAMPLE_POSTING, SAMPLE_REQUIREMENTS


# ------------------------------------------------------------------ #
# clean_posting_text (PROJECT_DOCS #014)                              #
# ------------------------------------------------------------------ #

def test_clean_removes_base64_images_and_link_urls():
    raw = (
        "회사 소개 ![logo](data:image/png;base64,iVBORw0KGgoAAA==)\n"
        "지원: [쏘카](/company/123) 채용\n"
        "inline data:image/png;base64,AAAABBBB== 끝"
    )
    out = clean_posting_text(raw)
    assert "base64" not in out
    assert "/company/123" not in out
    assert "쏘카" in out


def test_clean_collapses_blank_lines_and_truncates():
    assert clean_posting_text("a\n\n\n\nb") == "a\n\nb"
    assert len(clean_posting_text("가" * (MAX_POSTING_CHARS + 500))) == MAX_POSTING_CHARS


# ------------------------------------------------------------------ #
# normalize_remote_policy / normalize_requirements (#019)             #
# ------------------------------------------------------------------ #

@pytest.mark.parametrize("value, expected", [
    ("", ""),
    ("전면 재택근무", "재택"),
    ("Remote", "재택"),
    ("재택근무 가능", "하이브리드"),
    ("주 2일 재택", "하이브리드"),
    ("하이브리드 근무", "하이브리드"),
    ("사무실 출근", "출근"),
    ("출근 + 재택 병행", "하이브리드"),
    ("자율 복장", ""),
    ("유연근무제", ""),
])
def test_normalize_remote_policy(value, expected):
    assert normalize_remote_policy(value) == expected


def test_normalize_requirements_blanks_unknown_values():
    req = {
        "salary_range": "정보 없음",
        "location": "N/A",
        "company_culture": "수평적 문화",
        "preferred_skills": ["Docker", "N/A", "-"],
        "remote_policy": "재택근무 가능",
    }
    out = normalize_requirements(req)
    assert out["salary_range"] == ""
    assert out["location"] == ""
    assert out["company_culture"] == "수평적 문화"
    assert out["preferred_skills"] == ["Docker"]
    assert out["remote_policy"] == "하이브리드"
    assert req["location"] == "N/A", "원본 dict 를 바꾸면 안 됨"


# ------------------------------------------------------------------ #
# parse_local_response                                                 #
# ------------------------------------------------------------------ #

def test_parse_local_response_strips_code_fence_and_fills_defaults():
    raw = "```json\n" + json.dumps({"job_title": "백엔드", "experience_years": "3"}) + "\n```"
    out = parse_local_response(raw)
    assert out["job_title"] == "백엔드"
    assert out["experience_years"] == 3
    assert out["required_skills"] == []
    assert out["is_job_posting"] is True
    assert out["job_type"] == "정규직"


def test_parse_local_response_raises_on_local_error_marker():
    with pytest.raises(RuntimeError):
        parse_local_response("[LocalLLM error: connection refused]")


def test_parse_local_response_raises_on_invalid_json():
    with pytest.raises(ValueError):
        parse_local_response("이건 JSON 이 아닙니다")


def test_parse_with_local_llm_rejects_chinese_leak(monkeypatch):
    """Qwen 중국어 혼입은 예외로 올려서 다른 파서로 넘어가게 함 (#018)."""
    import llm.local_llm

    leaked = dict(SAMPLE_REQUIREMENTS, company_culture="我们公司有很好的文化。")
    monkeypatch.setattr(llm.local_llm.LocalLLM, "generate",
                        lambda self, **kw: json.dumps(leaked, ensure_ascii=False))
    with pytest.raises(ValueError, match="중국어"):
        job_parser.parse_with_local_llm(SAMPLE_POSTING)


# ------------------------------------------------------------------ #
# job_parser_node                                                      #
# ------------------------------------------------------------------ #

def test_node_empty_posting_returns_empty_requirements(fake_llm):
    state = job_parser_node({"job_posting_raw": "   "})
    assert state["job_requirements"]["is_job_posting"] is False
    assert state["errors"]
    assert fake_llm.calls == []


def test_node_parses_with_openai_structured_output(fake_llm):
    fake_llm.structured["JobRequirements"] = dict(SAMPLE_REQUIREMENTS, remote_policy="재택 가능")
    state = job_parser_node({"job_posting_raw": SAMPLE_POSTING, "errors": []})
    req = state["job_requirements"]
    assert req["required_skills"] == ["Python", "FastAPI"]
    assert req["remote_policy"] == "하이브리드", "OpenAI 결과도 정규화돼야 함"
    assert state["errors"] == []


def test_node_openai_failure_records_error(fake_llm):
    fake_llm.structured["JobRequirements"] = RuntimeError("rate limit")
    state = job_parser_node({"job_posting_raw": SAMPLE_POSTING})
    assert state["job_requirements"]["required_skills"] == []
    assert any("rate limit" in e for e in state["errors"])


def test_node_local_failure_falls_back_to_openai(fake_llm, monkeypatch):
    """로컬 파서가 실패해도 빈 결과로 끝내지 않고 OpenAI 로 재파싱."""
    from config.settings import settings

    monkeypatch.setattr(settings, "PARSE_WITH_LOCAL_LLM", True)
    monkeypatch.setattr(job_parser._router, "_local_available", True)

    def broken_local(raw_text, model=None):
        raise ValueError("로컬 모델 출력에 중국어가 섞임")

    monkeypatch.setattr(job_parser, "parse_with_local_llm", broken_local)
    fake_llm.structured["JobRequirements"] = SAMPLE_REQUIREMENTS

    state = job_parser_node({"job_posting_raw": SAMPLE_POSTING})
    assert state["job_requirements"]["job_title"] == "백엔드 개발자"
    assert fake_llm.count("structured", "JobRequirements") == 1
    assert any("OpenAI 로 대체" in e for e in state["errors"])
