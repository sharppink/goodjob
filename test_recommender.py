"""
test_recommender.py — 역방향 매칭 (공고 추천) 테스트
실행: python test_recommender.py
"""
import sys
sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

import time

# ── 1. 프로필 로드 ───────────────────────────────────────────────────
print("=" * 55)
print("  GoodJob — 공고 추천 테스트")
print("=" * 55)

print("\n[ 사전 준비: 프로필 벡터 DB 저장 ]")
from rag.profile_loader import ProfileLoader
loader = ProfileLoader()
loader.clear_profile()
count = loader.load_from_dict({
    "이름": "홍길동",
    "기술": ["Python", "FastAPI", "Django", "Docker",
             "PostgreSQL", "Redis", "LangGraph", "AWS EC2", "Git"],
    "경력": [
        "스타트업 A (2021~2024) 백엔드 개발자 - FastAPI 서버 개발, "
        "트래픽 처리 성능 40% 개선, MSA 구조 설계",
    ],
    "프로젝트": [
        "AI 채용 분석 시스템 - LangGraph + RAG 파이프라인, GPT-4o 연동",
    ],
    "학력": "한국대학교 컴퓨터공학과 졸업 (2020)",
    "자격증": ["정보처리기사", "AWS Solutions Architect Associate"],
})
print(f"  프로필 저장: {count}개 청크")

# ── 2. Structured Output 단독 테스트 ────────────────────────────────
print("\n[ 테스트 1: Structured Output 파싱 ]")
from agents.job_parser import parse_posting_structured

sample_posting = """
[네이버] 백엔드 엔지니어 채용
필수: Python 3년 이상, FastAPI, PostgreSQL, Docker
우대: Redis, Kafka, AWS, LangChain
근무: 경기 성남시 / 재택 40%
고용형태: 정규직
연봉: 5000~8000만원
"""
r = parse_posting_structured(sample_posting)
print(f"  직무명     : {r['job_title']}")
print(f"  필수기술   : {r['required_skills']}")
print(f"  위치       : {r['location']}")
print(f"  재택       : {r['remote_policy']}")
print(f"  급여       : {r['salary_range']}")
so_ok = bool(r["job_title"] and r["required_skills"])
print(f"  결과       : {'[OK]' if so_ok else '[FAIL]'}")

# ── 3. 역방향 매칭 전체 테스트 ──────────────────────────────────────
print("\n[ 테스트 2: 역방향 매칭 (공고 추천) ]")
print("  키워드: 'Python 백엔드 개발자'")
print("  (Tavily 검색 + Structured Output + RAG 점수화)")
print("  잠시 기다려 주세요…\n")

from agents.job_recommender import job_recommender_node
start = time.time()

state = {
    "recommendation_query": "Python 백엔드 개발자",
    "errors": [],
}
result = job_recommender_node(state)
elapsed = time.time() - start

candidates = result.get("job_candidates", [])
matches    = result.get("ranked_matches", [])
errors     = result.get("errors", [])

print(f"  소요 시간  : {elapsed:.1f}초")
print(f"  수집 공고  : {len(candidates)}개")
print(f"  구조화 성공: {sum(1 for c in candidates if c.get('requirements'))}개")
print(f"  랭킹 결과  : {len(matches)}개")

if errors:
    print(f"\n  [경고] {errors}")

if matches:
    print()
    print("─" * 55)
    print("  추천 결과")
    print("─" * 55)
    for m in matches:
        score = m["fit_score"]
        grade = "높음" if score >= 0.7 else "보통" if score >= 0.4 else "낮음"
        req   = m.get("requirements", {})
        print(f"\n  [{m['rank']}위] {m['title'][:45]}")
        print(f"       회사    : {m['company']}")
        print(f"       적합도  : {score:.0%} ({grade})")
        print(f"       필수기술: {req.get('required_skills', [])[:4]}")
        loc = req.get("location", "-")
        rem = req.get("remote_policy", "-")
        sal = req.get("salary_range", "-")
        print(f"       위치    : {loc} | 재택: {rem}")
        if sal and sal != "-":
            print(f"       급여    : {sal}")
else:
    print("\n  매칭된 공고 없음 (적합도 30% 미만이거나 검색 결과 없음)")

# ── 결과 요약 ────────────────────────────────────────────────────────
print("\n\n" + "=" * 55)
print("  결과 요약")
print("=" * 55)
results = {
    "structured_output": so_ok,
    "job_recommender":   len(matches) > 0,
}
for name, ok in results.items():
    status = "[OK]   정상" if ok else "[FAIL] 실패"
    print(f"  {name:<25} {status}")
print()
