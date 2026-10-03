"""
scripts/docker_smoke.py

docker compose 로 띄운 컨테이너 안에서 실행하는 동작 확인 스크립트 (CI 용).
표준입력으로 넘겨 실행합니다 (이미지에는 scripts/ 가 들어가지 않음):

    docker compose exec -T api      python - write    < scripts/docker_smoke.py
    docker compose exec -T frontend python - read     < scripts/docker_smoke.py
    docker compose exec -T frontend python - pdf      < scripts/docker_smoke.py
    docker compose exec -T api      python - chromium < scripts/docker_smoke.py

write/read : API 컨테이너가 저장한 벡터를 Streamlit 컨테이너가 검색 (Chroma 서버 공유, #024)
             OpenAI 를 쓰지 않도록 고정 벡터 사용. 빈 DB 에서만 동작 (실제 프로필 보호)
pdf        : 이력서 PDF 가 한국어 폰트(나눔고딕)로 생성되는지
chromium   : Playwright Chromium 이 비루트 사용자로 실행되는지
"""

from __future__ import annotations

import sys

MARKER = "docker-smoke-test-document"
VECTOR = [1.0, 0.0, 0.0]


class _FixedEmbedding:
    def embed_texts(self, texts):
        return [VECTOR for _ in texts]

    def embed_query(self, query):
        return VECTOR


def _store():
    import rag.vectorstore as vectorstore

    vectorstore.EmbeddingModel = lambda *a, **k: _FixedEmbedding()
    vs = vectorstore.VectorStore()
    vs.initialize()
    return vs


def write() -> None:
    vs = _store()
    if vs.count() != 0:
        sys.exit(f"벡터 DB 가 비어 있지 않음({vs.count()}개) — 실제 프로필 보호를 위해 중단")
    vs.add_documents([MARKER], metadatas=[{"source": "smoke"}])
    print("write ok:", vs.count())


def read() -> None:
    found = _store().similarity_search("query", k=1)
    assert found == [MARKER], f"다른 컨테이너가 저장한 문서를 찾지 못함: {found}"
    print("read ok:", found)


def pdf() -> None:
    from frontend.pdf_exporter import markdown_to_pdf_bytes
    import frontend.pdf_exporter as exporter

    data = markdown_to_pdf_bytes("# 이력서\n## 경력\n- 백엔드 개발", company_name="테스트")
    assert data.startswith(b"%PDF-")
    assert exporter._registered_fonts == ("KoreanFont", "KoreanFontBold"), exporter._registered_fonts
    print("pdf ok:", len(data), "bytes, font:", exporter._registered_fonts)


def chromium() -> None:
    from playwright.sync_api import sync_playwright

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        page = browser.new_page()
        page.set_content("<h1>채용공고</h1>")
        text = page.inner_text("h1")
        browser.close()
    assert text == "채용공고", text
    print("chromium ok")


if __name__ == "__main__":
    {"write": write, "read": read, "pdf": pdf, "chromium": chromium}[sys.argv[1]]()
