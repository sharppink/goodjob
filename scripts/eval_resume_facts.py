"""
scripts/eval_resume_facts.py

이력서 사실성 측정 (PROJECT_DOCS #025) — 실제 OpenAI 를 호출합니다 (1회당 gpt-4o 2번).

같은 샘플 프로필·공고로 resume_writer → reviewer 를 N번 실행하고,
"경력" 섹션 bullet 에 프로필 경력 원문에 없는 기술·업무가 들어갔는지 셉니다.

    python -m scripts.eval_resume_facts            # 5회
    python -m scripts.eval_resume_facts --runs 3 --out result.json

판정 규칙 (샘플 프로필 기준)
- 경력 원문: "FastAPI 기반 REST API 설계 및 개발, 응답속도 40% 개선, Redis 캐시 도입으로 DB 부하 30% 감소"
- 경력 bullet 에 아래 단어가 나오면 위반: 원문 경력에 없는 기술·업무 (보유 기술 목록이나 프로젝트,
  공고 업무 문구에서 끌어온 것)
- 원문에 없는 수치(40%, 30% 외의 %)도 위반
"""

from __future__ import annotations

import argparse
import json
import re
import sys

sys.stdout.reconfigure(encoding="utf-8")

EXPERIENCES = [
    "Name: 홍길동",
    "Summary: Python 백엔드 개발자 3년차",
    "Skills:\n  - Python\n  - FastAPI\n  - PostgreSQL\n  - Docker\n  - Redis\n  - LangGraph",
    "Experience: 스타트업 A (2022~2025) 백엔드 개발자 - FastAPI 기반 REST API 설계 및 개발, "
    "응답속도 40% 개선, Redis 캐시 도입으로 DB 부하 30% 감소",
    "Projects: LangGraph 기반 이력서 생성 AI 에이전트 개발 (RAG, Chroma)",
]

REQUIREMENTS = {
    "is_job_posting": True,
    "company_name": "테스트랩",
    "job_title": "Python 백엔드 개발자",
    "required_skills": ["Python", "FastAPI", "PostgreSQL"],
    "preferred_skills": ["LangChain", "LangGraph", "Docker", "AWS"],
    "experience_years": 2,
    "responsibilities": [
        "FastAPI 기반 REST API 설계 및 개발",
        "LLM 연동 서비스 백엔드 개발",
        "PostgreSQL 데이터 모델링 및 쿼리 최적화",
    ],
    "company_culture": "",
    "keywords": ["Python", "FastAPI", "REST API", "LLM", "PostgreSQL", "데이터 모델링",
                 "쿼리 최적화", "LangChain", "LangGraph", "Docker", "AWS"],
}

# 경력(스타트업 A) 원문에 없는 기술·업무 — 경력 bullet 에 나오면 위반
UNSUPPORTED_IN_CAREER = ["PostgreSQL", "Docker", "LangGraph", "LangChain", "AWS", "LLM",
                         "쿼리 최적화", "데이터 모델링", "컨테이너", "배포", "에이전트"]
ALLOWED_PERCENTS = {"40", "30"}


def career_section(md: str) -> str:
    """'경력' 헤더부터 다음 같은 수준 이상의 헤더 전까지."""
    lines = md.splitlines()
    out, level = [], None
    for line in lines:
        m = re.match(r"^(#{1,6})\s*(.*)", line)
        if m:
            depth, title = len(m.group(1)), m.group(2)
            if level is None and "경력" in title and "보완" not in title:
                level = depth
                continue
            if level is not None and depth <= level:
                break
        if level is not None:
            out.append(line)
    return "\n".join(out)


def check(md: str) -> list[str]:
    career = career_section(md)
    if not career.strip():
        return ["경력 섹션을 찾지 못함"]
    problems = []
    for line in career.splitlines():
        if not line.strip().startswith(("-", "*")):
            continue
        hits = [w for w in UNSUPPORTED_IN_CAREER if w.lower() in line.lower()]
        if hits:
            problems.append(f"원문에 없는 업무 {hits}: {line.strip()}")
        for pct in re.findall(r"(\d+)\s*%", line):
            if pct not in ALLOWED_PERCENTS:
                problems.append(f"원문에 없는 수치 {pct}%: {line.strip()}")
    return problems


def run_once() -> dict:
    from agents.resume_writer import resume_writer_node
    from agents.reviewer import reviewer_node

    state = {
        "company_name": "테스트랩",
        "job_requirements": REQUIREMENTS,
        "retrieved_experiences": EXPERIENCES,
        "fit_score": 0.85,
        "fit_feedback": "Python·FastAPI 경험이 직무와 잘 맞음. LangChain·AWS 경험은 부족.",
        "matched_skills": ["Python", "FastAPI", "PostgreSQL", "LangGraph", "Docker"],
        "missing_skills": ["LangChain", "AWS"],
        "errors": [],
    }
    state = resume_writer_node(state)
    state = reviewer_node(state)
    return {
        "draft": state["resume_draft"],
        "final": state["resume_final"],
        "draft_problems": check(state["resume_draft"]),
        "final_problems": check(state["resume_final"]),
        "errors": state.get("errors") or [],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--out", default="")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    results = []
    for i in range(args.runs):
        r = run_once()
        results.append(r)
        print(f"[{i + 1}/{args.runs}] 초안 위반 {len(r['draft_problems'])}건, 최종 위반 {len(r['final_problems'])}건"
              + (f", 오류 {r['errors']}" if r["errors"] else ""))
        for p in r["final_problems"]:
            print("    -", p)

    clean = sum(1 for r in results if not r["final_problems"])
    total = sum(len(r["final_problems"]) for r in results)
    print(f"\n최종 이력서: {clean}/{args.runs}회 위반 없음, 위반 bullet 총 {total}건")
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
