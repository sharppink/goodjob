"""finetune/metrics.py — 파인튜닝 평가 지표."""

from __future__ import annotations

import json

import pytest

from finetune.metrics import has_chinese_leak, parse_prediction, score_all, score_one, set_f1

GOLD = {
    "is_job_posting": True,
    "company_name": "쏘카",
    "job_title": "백엔드 개발자",
    "required_skills": ["Python", "FastAPI"],
    "preferred_skills": ["Docker"],
    "keywords": ["Python", "MSA"],
    "experience_years": 2,
}


def test_parse_prediction_handles_fence_and_surrounding_text():
    assert parse_prediction('```json\n{"a": 1}\n```') == {"a": 1}
    assert parse_prediction('결과입니다: {"a": 1} 이상') == {"a": 1}
    assert parse_prediction("not json") is None
    assert parse_prediction("[1, 2]") is None


@pytest.mark.parametrize("pred, gold, expected", [
    ([], [], 1.0),
    (["Python"], [], 0.0),
    ([], ["Python"], 0.0),
    (["python", " FastAPI "], ["Python", "FastAPI"], 1.0),
    (["Python"], ["Python", "FastAPI"], pytest.approx(2 / 3)),
])
def test_set_f1_basic(pred, gold, expected):
    assert set_f1(pred, gold) == expected


def test_set_f1_fuzzy_matches_spelling_variants():
    """표기만 다른 같은 기술은 일치로 침 (#015)."""
    assert set_f1(["playwright를 통한 e2e 테스트"], ["playwright를 통해 e2e test"]) == 1.0


def test_set_f1_does_not_match_java_to_javascript():
    assert set_f1(["Java"], ["JavaScript"]) == 0.0


def test_set_f1_is_one_to_one():
    """예측 하나가 정답 여러 개와 매칭되면 안 됨."""
    assert set_f1(["Python"], ["Python", "python3"]) == pytest.approx(2 / 3)


def test_has_chinese_leak():
    assert has_chinese_leak({"a": ["정상 한국어", "我们的公司文化"]})
    assert has_chinese_leak("문장 끝에 중국어 마침표。")
    assert not has_chinese_leak({"a": "漢字 두 글자 정도는 허용"})


def test_score_one_perfect_and_invalid():
    perfect = score_one(json.dumps(GOLD, ensure_ascii=False), GOLD)
    assert all(v == 1.0 for v in perfect.values()), perfect

    invalid = score_one("broken", GOLD)
    assert all(v == 0.0 for v in invalid.values())


def test_score_one_contains_match_for_titles():
    pred = dict(GOLD, job_title="백엔드 개발자 (Python)", company_name="쏘카(SOCAR)")
    scores = score_one(json.dumps(pred, ensure_ascii=False), GOLD)
    assert scores["job_title_match"] == 1.0
    assert scores["company_name_match"] == 1.0


def test_score_all_averages():
    good = json.dumps(GOLD, ensure_ascii=False)
    result = score_all([good, "broken"], [GOLD, GOLD])
    assert result["json_valid"] == 0.5
    with pytest.raises(AssertionError):
        score_all([good], [GOLD, GOLD])
