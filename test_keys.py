"""
API 키 동작 확인 스크립트
실행: python test_keys.py
"""

import os
import sys

# Windows 콘솔 UTF-8 출력
sys.stdout.reconfigure(encoding="utf-8")

from dotenv import load_dotenv
load_dotenv()


def check_openai():
    print("\n[ OpenAI API ]")
    key = os.getenv("OPENAI_API_KEY", "")
    if not key or key == "sk-your-openai-key-here":
        print("  [FAIL] 키가 입력되지 않았습니다.")
        return False
    try:
        from openai import OpenAI
        client = OpenAI(api_key=key)
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": "Say 'API works' in Korean only."}],
            max_tokens=20,
        )
        result = response.choices[0].message.content
        print(f"  [OK] 연결 성공 -- 응답: {result}")
        return True
    except Exception as e:
        print(f"  [FAIL] 오류: {e}")
        return False


def check_langsmith():
    print("\n[ LangSmith API ]")
    key = os.getenv("LANGCHAIN_API_KEY", "")
    if not key or key == "ls__your-langsmith-key-here":
        print("  [FAIL] 키가 입력되지 않았습니다.")
        return False
    try:
        from langsmith import Client
        client = Client(api_key=key)
        projects = list(client.list_projects())
        print(f"  [OK] 연결 성공 -- 프로젝트 수: {len(projects)}개")
        return True
    except Exception as e:
        print(f"  [FAIL] 오류: {e}")
        return False


def check_tavily():
    print("\n[ Tavily API ]")
    key = os.getenv("TAVILY_API_KEY", "")
    if not key or key == "tvly-your-key-here":
        print("  [SKIP] 키 미입력 (선택 사항)")
        return None
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=key)
        result = client.search("OpenAI", max_results=1)
        title = result["results"][0]["title"][:40]
        print(f"  [OK] 연결 성공 -- 검색 결과: {title}...")
        return True
    except Exception as e:
        print(f"  [FAIL] 오류: {e}")
        return False


if __name__ == "__main__":
    print("=" * 40)
    print("  GoodJob API 키 확인")
    print("=" * 40)

    results = {
        "OpenAI":    check_openai(),
        "LangSmith": check_langsmith(),
        "Tavily":    check_tavily(),
    }

    print("\n" + "=" * 40)
    print("  결과 요약")
    print("=" * 40)
    for name, ok in results.items():
        if ok is True:
            status = "[OK]   정상"
        elif ok is False:
            status = "[FAIL] 실패"
        else:
            status = "[SKIP] 미입력"
        print(f"  {name:<12} {status}")
    print()
