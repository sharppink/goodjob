"""
LangGraph 에이전트 파이프라인 엔드투엔드 테스트
실행: python test_agents.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv()

# ── 샘플 데이터 ──────────────────────────────────────────────────
SAMPLE_JOB_POSTING = """
[카카오] 백엔드 개발자 채용

■ 담당 업무
- Python/FastAPI 기반 REST API 설계 및 개발
- MSA 환경에서 서비스 간 통신 구조 설계
- 대용량 트래픽 처리를 위한 성능 최적화
- 코드 리뷰 및 기술 문서 작성

■ 필수 조건
- Python 백엔드 개발 경력 3년 이상
- FastAPI 또는 Django REST Framework 실무 경험
- MySQL, PostgreSQL 등 RDBMS 설계 및 최적화 경험
- Docker, Kubernetes 기반 컨테이너 운영 경험

■ 우대 사항
- LangChain, LangGraph 등 AI/LLM 연동 경험
- Redis, Kafka 등 메시지 큐 활용 경험
- AWS 또는 GCP 클라우드 서비스 운영 경험
- 오픈소스 기여 경험

■ 근무 조건
- 위치: 판교 / 재택 50% 병행
- 고용형태: 정규직
- 자율 출퇴근제 운영
"""

SAMPLE_PROFILE = {
    "이름": "홍길동",
    "기술": ["Python", "FastAPI", "Django", "Docker", "PostgreSQL", "Redis",
             "LangGraph", "RAG", "AWS EC2", "Git"],
    "경력": [
        "스타트업 A (2021~2024) 백엔드 개발자 - FastAPI 서버 개발, "
        "트래픽 처리 성능 40% 개선, MSA 구조 설계",
        "스타트업 B (2020~2021) 풀스택 개발자 - Django + React 서비스 개발",
    ],
    "프로젝트": [
        "AI 채용 분석 시스템 개발 - LangGraph + RAG 파이프라인 구축, "
        "GPT-4o 연동, Chroma 벡터 DB 활용",
        "실시간 알림 서비스 - WebSocket + Redis Pub/Sub, 동시 접속자 1만명 처리",
    ],
    "학력": "한국대학교 컴퓨터공학과 졸업 (2016~2020)",
    "자격증": ["정보처리기사", "AWS Solutions Architect Associate"],
}


def setup_profile():
    """테스트용 프로필을 벡터 DB에 저장합니다."""
    from rag.profile_loader import ProfileLoader
    loader = ProfileLoader()
    loader.clear_profile()
    count = loader.load_from_dict(SAMPLE_PROFILE)
    print(f"  프로필 저장: {count}개 청크")
    return count > 0


def test_job_parser():
    print("\n[ 노드 1: job_parser ]")
    from agents.job_parser import job_parser_node
    state = {"job_posting_raw": SAMPLE_JOB_POSTING, "errors": []}
    state = job_parser_node(state)

    req = state.get("job_requirements", {})
    print(f"  직무명: {req.get('job_title')}")
    print(f"  필수 기술 ({len(req.get('required_skills', []))}개): "
          f"{', '.join(req.get('required_skills', [])[:5])}")
    print(f"  우대 기술 ({len(req.get('preferred_skills', []))}개): "
          f"{', '.join(req.get('preferred_skills', [])[:3])}")
    print(f"  키워드 ({len(req.get('keywords', []))}개): "
          f"{', '.join(req.get('keywords', [])[:5])}")
    return bool(req.get("required_skills")), state


def test_rag_retriever(state):
    print("\n[ 노드 2: rag_retriever ]")
    from agents.rag_retriever import rag_retriever_node
    state = rag_retriever_node(state)

    experiences = state.get("retrieved_experiences", [])
    print(f"  검색된 경험: {len(experiences)}개")
    for i, exp in enumerate(experiences[:2], 1):
        print(f"  [{i}] {exp[:80]}...")
    return len(experiences) > 0, state


def test_fit_analyzer(state):
    print("\n[ 노드 3: fit_analyzer ]")
    from agents.fit_analyzer import fit_analyzer_node
    state = fit_analyzer_node(state)

    score = state.get("fit_score", 0)
    feedback = state.get("fit_feedback", "")
    print(f"  적합도 점수: {score:.0%}")
    print(f"  피드백 (앞 200자): {feedback[:200]}")
    return score > 0, state


def test_resume_writer(state):
    print("\n[ 노드 4: resume_writer ]")
    from agents.resume_writer import resume_writer_node
    state["company_name"] = "카카오"
    state = resume_writer_node(state)

    draft = state.get("resume_draft", "")
    print(f"  초안 길이: {len(draft)}자")
    print(f"  앞 300자:\n{draft[:300]}")
    return len(draft) > 100, state


def test_reviewer(state):
    print("\n[ 노드 5: reviewer ]")
    from agents.reviewer import reviewer_node
    state = reviewer_node(state)

    final = state.get("resume_final", "")
    print(f"  최종본 길이: {len(final)}자")
    print(f"\n{'='*50}")
    print("  최종 이력서 (앞 500자)")
    print('='*50)
    print(final[:500])
    return len(final) > 100, state


def test_full_graph():
    print("\n[ 전체 그래프: graph.app.invoke() ]")
    from agents.graph import app

    initial_state = {
        "company_name": "카카오",
        "job_posting_raw": SAMPLE_JOB_POSTING,
        "errors": [],
    }
    final_state = app.invoke(initial_state)
    print(f"  완료 단계: {final_state.get('current_step')}")
    print(f"  적합도: {final_state.get('fit_score', 0):.0%}")
    print(f"  최종 이력서 길이: {len(final_state.get('resume_final', ''))}자")
    print(f"  에러: {final_state.get('errors', [])}")
    return bool(final_state.get("resume_final")), final_state


if __name__ == "__main__":
    print("=" * 50)
    print("  GoodJob LangGraph 에이전트 테스트")
    print("=" * 50)

    print("\n[ 사전 준비: 프로필 벡터 DB 저장 ]")
    setup_profile()

    results = {}

    ok, state = test_job_parser()
    results["job_parser"] = ok

    ok, state = test_rag_retriever(state)
    results["rag_retriever"] = ok

    ok, state = test_fit_analyzer(state)
    results["fit_analyzer"] = ok

    # 적합도가 낮으면 이후 노드 건너뜀
    if state.get("fit_score", 0) >= 0.3:
        ok, state = test_resume_writer(state)
        results["resume_writer"] = ok

        ok, state = test_reviewer(state)
        results["reviewer"] = ok
    else:
        print(f"\n  [주의] 적합도 {state.get('fit_score', 0):.0%} — "
              f"이력서 생성 건너뜀 (임계값 30% 미만)")
        results["resume_writer"] = None
        results["reviewer"] = None

    print("\n\n" + "=" * 50)
    print("  결과 요약")
    print("=" * 50)
    for name, ok in results.items():
        if ok is True:
            status = "[OK]   정상"
        elif ok is False:
            status = "[FAIL] 실패"
        else:
            status = "[SKIP] 건너뜀"
        print(f"  {name:<20} {status}")
    print()
