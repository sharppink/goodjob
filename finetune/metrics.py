"""
finetune/metrics.py

공고 파싱 모델 평가 지표 (순수 Python — Colab·로컬 공통 사용).

정답(label)은 gpt-4o Structured Output 결과(teacher), 예측(pred)은 평가 대상 모델의
JSON 문자열입니다.

지표
----
- json_valid          : 예측이 JSON 으로 파싱되는 비율
- is_job_posting_acc  : 공고 여부 판정 정확도
- required_skills_f1  : 필수 기술 집합 F1 (대소문자·공백 무시)
- preferred_skills_f1 : 우대 기술 집합 F1
- keywords_f1         : 키워드 집합 F1
- experience_years_acc: 최소 경력 연수 일치율
- job_title_match     : 직무명 일치(한쪽이 다른 쪽을 포함하면 일치)
- company_name_match  : 회사명 일치(포함 관계 허용)

JSON 파싱에 실패한 예측은 모든 필드 점수를 0 으로 계산합니다.
"""

from __future__ import annotations

import json
import re
from typing import Any

SET_FIELDS = ("required_skills", "preferred_skills", "keywords")


def parse_prediction(text: str) -> dict[str, Any] | None:
    """모델 출력 문자열에서 JSON 객체를 추출합니다. 실패 시 None."""
    clean = re.sub(r"```(?:json)?", "", text or "").strip().rstrip("`").strip()
    try:
        data = json.loads(clean)
    except json.JSONDecodeError:
        # 앞뒤에 설명이 붙은 경우: 첫 { 부터 마지막 } 까지 재시도
        start, end = clean.find("{"), clean.rfind("}")
        if start == -1 or end <= start:
            return None
        try:
            data = json.loads(clean[start:end + 1])
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def _norm(s: Any) -> str:
    return re.sub(r"\s+", "", str(s)).lower()


FUZZY_THRESHOLD = 0.75


def _similar(a: str, b: str) -> bool:
    """
    정규화 문자열 기준: 같거나, 포함 관계이면서 짧은 쪽이 긴 쪽의 절반 이상이거나
    ("java" ⊂ "javascript" 같은 다른 기술의 오매칭 방지), 유사도 ≥ 0.75.
    """
    from difflib import SequenceMatcher
    if a == b:
        return True
    short, long_ = sorted((a, b), key=len)
    if len(short) >= 2 and short in long_ and len(short) / len(long_) >= 0.5:
        return True
    return SequenceMatcher(None, a, b).ratio() >= FUZZY_THRESHOLD


def set_f1(pred: list | None, gold: list | None) -> float:
    """
    목록 F1 (퍼지 매칭, 1:1 대응).

    완전 일치만 세면 "playwright를 통한 e2e 테스트" vs "playwright를 통해 e2e test"
    처럼 같은 뜻도 0점이 되어 모델 차이가 아닌 표기 차이를 측정하게 됨 (PROJECT_DOCS #015).
    """
    p = list(dict.fromkeys(_norm(x) for x in (pred or []) if str(x).strip()))
    g = list(dict.fromkeys(_norm(x) for x in (gold or []) if str(x).strip()))
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    unmatched = list(g)
    tp = 0
    for item in p:
        hit = next((x for x in unmatched if _similar(item, x)), None)
        if hit is not None:
            unmatched.remove(hit)
            tp += 1
    if tp == 0:
        return 0.0
    precision, recall = tp / len(p), tp / len(g)
    return 2 * precision * recall / (precision + recall)


def _contains_match(pred: Any, gold: Any) -> float:
    p, g = _norm(pred or ""), _norm(gold or "")
    if not p and not g:
        return 1.0
    if not p or not g:
        return 0.0
    return 1.0 if (p in g or g in p) else 0.0


def score_one(pred_text: str, gold: dict[str, Any]) -> dict[str, float]:
    pred = parse_prediction(pred_text)
    if pred is None:
        return {"json_valid": 0.0, "is_job_posting_acc": 0.0,
                **{f"{f}_f1": 0.0 for f in SET_FIELDS},
                "experience_years_acc": 0.0, "job_title_match": 0.0, "company_name_match": 0.0}

    def _int(v: Any) -> int:
        try:
            return int(v or 0)
        except (TypeError, ValueError):
            return -1

    return {
        "json_valid": 1.0,
        "is_job_posting_acc": float(bool(pred.get("is_job_posting", True)) == bool(gold.get("is_job_posting"))),
        **{f"{f}_f1": set_f1(pred.get(f), gold.get(f)) for f in SET_FIELDS},
        "experience_years_acc": float(_int(pred.get("experience_years")) == _int(gold.get("experience_years"))),
        "job_title_match": _contains_match(pred.get("job_title"), gold.get("job_title")),
        "company_name_match": _contains_match(pred.get("company_name"), gold.get("company_name")),
    }


def score_all(pred_texts: list[str], golds: list[dict[str, Any]]) -> dict[str, float]:
    """예측/정답 리스트 → 지표별 평균 (소수 3자리)."""
    assert len(pred_texts) == len(golds), "예측과 정답 개수가 다릅니다."
    rows = [score_one(p, g) for p, g in zip(pred_texts, golds)]
    if not rows:
        return {}
    return {k: round(sum(r[k] for r in rows) / len(rows), 3) for k in rows[0]}


def format_table(results: dict[str, dict[str, float]]) -> str:
    """{"base": {...}, "finetuned": {...}} → 비교 표 문자열."""
    names = list(results)
    metrics = list(next(iter(results.values())))
    lines = ["metric".ljust(22) + "".join(n.rjust(12) for n in names)]
    for m in metrics:
        lines.append(m.ljust(22) + "".join(f"{results[n][m]:12.3f}" for n in names))
    return "\n".join(lines)
