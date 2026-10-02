"""
search 파이프라인 동작 확인 스크립트
실행: python test_search.py
"""

import sys
sys.stdout.reconfigure(encoding="utf-8")

import logging
logging.basicConfig(level=logging.WARNING)

from dotenv import load_dotenv
load_dotenv()


def test_company_search():
    print("\n[ 테스트 1: 회사 정보 + 채용공고 검색 (Tavily) ]")
    from search.company_searcher import CompanySearcher

    searcher = CompanySearcher()
    result = searcher.search_all("카카오", role="백엔드 개발자")

    print(f"  검색 결과 수: {result['job_postings']['count']}개")
    print(f"  수집된 URL 수: {len(result['urls'])}개")
    if result["urls"]:
        print(f"  첫 번째 URL: {result['urls'][0]}")
    print(f"  요약 (앞 200자):\n    {result['summary'][:200]}")
    return result["job_postings"]["count"] > 0


def test_url_fetch():
    print("\n[ 테스트 2: Tavily URL 내용 추출 ]")
    from search.company_searcher import CompanySearcher

    searcher = CompanySearcher()
    # 공개된 원티드 채용공고 URL로 테스트
    test_url = "https://www.wanted.co.kr/jobsfeed"
    content = searcher.fetch_url_content(test_url)
    print(f"  추출된 텍스트 길이: {len(content)}자")
    if content:
        print(f"  앞 150자: {content[:150]}")
    return len(content) > 0


def test_pipeline():
    print("\n[ 테스트 3: 통합 파이프라인 (회사명 텍스트 입력) ]")
    from search.pipeline import JobSearchPipeline

    pipeline = JobSearchPipeline()
    result = pipeline.run(
        company_name="네이버",
        role="백엔드 개발자",
        fetch_detail=False,   # Playwright 스킵 (속도 우선)
    )

    print(f"  회사명: {result['company_name']}")
    print(f"  직무: {result['role']}")
    print(f"  공고 텍스트 길이: {len(result['job_posting_text'])}자")
    print(f"  URL 수: {len(result['urls'])}개")
    print(f"  요약 (앞 200자):\n    {result['summary'][:200]}")
    return len(result["job_posting_text"]) > 0


if __name__ == "__main__":
    print("=" * 50)
    print("  GoodJob Search 파이프라인 테스트")
    print("=" * 50)

    results = {}
    results["회사 검색 (Tavily)"] = test_company_search()
    results["URL 추출"]           = test_url_fetch()
    results["통합 파이프라인"]     = test_pipeline()

    print("\n" + "=" * 50)
    print("  결과 요약")
    print("=" * 50)
    for name, ok in results.items():
        status = "[OK]   정상" if ok else "[FAIL] 실패"
        print(f"  {name:<22} {status}")
    print()
