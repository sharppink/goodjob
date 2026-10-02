"""
RAG 파이프라인 동작 확인 스크립트
실행: python test_rag.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)  # 로그 최소화

from dotenv import load_dotenv
load_dotenv()


def test_dict_input():
    print("\n[ 테스트 1: 딕셔너리 직접 입력 ]")
    from rag.profile_loader import ProfileLoader

    loader = ProfileLoader()
    loader.clear_profile()

    sample_profile = {
        "이름": "홍길동",
        "기술": ["Python", "FastAPI", "LangGraph", "RAG", "Docker", "SQL"],
        "경력": [
            "카카오 백엔드 개발자 3년 (2021~2024) - FastAPI 서버 개발, MSA 설계",
            "스타트업 풀스택 개발 1년 (2020~2021) - React + Django",
        ],
        "학력": "한국대학교 컴퓨터공학과 (2016~2020)",
        "프로젝트": [
            "AI 채용 분석 시스템 - LangGraph + RAG 활용",
            "실시간 채팅 서비스 - WebSocket + Redis",
        ],
        "자격증": ["정보처리기사", "AWS Solutions Architect"],
    }

    count = loader.load_from_dict(sample_profile)
    print(f"  저장된 청크 수: {count}개")
    print(f"  총 저장 문서: {loader.get_stored_count()}개")
    return count > 0


def test_search():
    print("\n[ 테스트 2: 벡터 검색 (Reranker 없이) ]")
    from rag.vectorstore import VectorStore
    from rag.embeddings import EmbeddingModel

    print("  BGE-M3 임베딩 모델 로딩 중... (첫 실행 시 다운로드 필요, 약 2.4GB)")
    vs = VectorStore()
    vs.initialize()

    em = EmbeddingModel()
    query = "Python 백엔드 개발 경험"
    print(f"  검색 쿼리: '{query}'")

    query_vec = em.embed_query(query)
    print(f"  임베딩 벡터 차원: {len(query_vec)}")

    results = vs.similarity_search(query, k=3)
    print(f"  검색 결과 {len(results)}개:")
    for i, r in enumerate(results, 1):
        print(f"    [{i}] {r[:80]}...")
    return len(results) > 0


def test_profile_loader_search():
    print("\n[ 테스트 3: ProfileLoader.search() (Reranker 포함) ]")
    from rag.profile_loader import ProfileLoader

    loader = ProfileLoader()
    query = "FastAPI 서버 개발 경험"
    print(f"  검색 쿼리: '{query}'")

    results = loader.search(query, k=3)
    print(f"  최종 결과 {len(results)}개:")
    for i, r in enumerate(results, 1):
        print(f"    [{i}] {r[:80]}...")
    return True


if __name__ == "__main__":
    print("=" * 50)
    print("  GoodJob RAG 파이프라인 테스트")
    print("=" * 50)

    results = {}

    results["딕셔너리 입력"] = test_dict_input()

    try:
        results["벡터 검색"] = test_search()
        results["ProfileLoader 검색"] = test_profile_loader_search()
    except Exception as e:
        print(f"\n  [주의] 임베딩 모델 관련 오류: {e}")
        print("  sentence-transformers 및 BGE-M3 모델이 필요합니다.")
        results["벡터 검색"] = False
        results["ProfileLoader 검색"] = False

    print("\n" + "=" * 50)
    print("  결과 요약")
    print("=" * 50)
    for name, ok in results.items():
        status = "[OK]   정상" if ok else "[FAIL] 실패"
        print(f"  {name:<22} {status}")
    print()
