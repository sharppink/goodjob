"""rag/* — 청킹, 벡터 저장소, 프로필 로더 (임시 Chroma + 가짜 임베딩)."""

from __future__ import annotations

import os

import pytest

from conftest import SAMPLE_PROFILE
from rag.profile_loader import CHUNK_SIZE, ProfileLoader
from rag.vectorstore import VectorStore


# ------------------------------------------------------------------ #
# 청킹                                                                 #
# ------------------------------------------------------------------ #

def test_dict_to_text_skips_empty_values():
    text = ProfileLoader._dict_to_text({"이름": "A", "경력": [], "학력": "", "메모": None,
                                        "기술": ["Python", "SQL"]})
    assert "경력" not in text and "학력" not in text and "메모" not in text
    assert "기술:\n  - Python\n  - SQL" in text


def test_split_by_sections_keeps_header_line_content():
    """'Experience: 스타트업 A' 처럼 헤더와 내용이 한 줄이어도 내용이 사라지면 안 됨 (#007)."""
    text = "홍길동\nExperience: 스타트업 A 백엔드 3년\nSkills: Python, FastAPI"
    sections = dict(ProfileLoader._split_by_sections(text))
    assert "스타트업 A 백엔드 3년" in sections["Experience"]
    assert "Python, FastAPI" in sections["Skills"]
    assert sections["intro"] == "홍길동"


def test_overlap_split_respects_size_and_overlap():
    text = ". ".join(f"문장 {i} 내용입니다" for i in range(200))
    chunks = ProfileLoader._overlap_split(text)
    assert len(chunks) > 1
    assert all(len(c) <= CHUNK_SIZE for c in chunks)
    assert chunks[0][-20:].strip()[-5:] in text
    assert ProfileLoader._overlap_split("   ") == []
    assert ProfileLoader._overlap_split("짧은 글") == ["짧은 글"]


def test_every_profile_field_is_stored(loaded_profile):
    """모든 섹션이 청크로 저장돼야 함 — 경력이 빠져 이력서 환각이 생겼던 버그 (#007)."""
    docs = "\n".join(VectorStore().get_all_documents())
    for value in ("테스트 지원자", "FastAPI 서버 개발", "LangGraph", "테스트대학교"):
        assert value in docs


# ------------------------------------------------------------------ #
# VectorStore / ProfileLoader                                          #
# ------------------------------------------------------------------ #

def test_vectorstore_uses_temp_dir_not_real_db():
    from config.settings import settings
    from conftest import REAL_CHROMA_DIR

    vs = VectorStore()
    assert os.path.abspath(vs._persist_dir) != str(REAL_CHROMA_DIR)
    assert os.path.abspath(vs._persist_dir) == os.path.abspath(settings.CHROMA_PERSIST_DIR)


def test_explicit_persist_dir_is_not_ignored(tmp_path):
    """다른 경로를 넘기면 그 경로의 DB 를 써야 함 — 첫 경로에 고정되던 버그 (#009)."""
    default_vs = VectorStore()
    default_vs.add_documents(["기본 저장소 문서"])

    other_vs = VectorStore(persist_dir=str(tmp_path / "other"))
    assert other_vs.count() == 0
    other_vs.add_documents(["다른 저장소 문서"])
    assert default_vs.count() == 1


def test_add_search_and_delete():
    vs = VectorStore()
    vs.add_documents(["Python FastAPI 백엔드", "React 프론트엔드", "Kubernetes 운영"],
                     metadatas=[{"i": 0}, {"i": 1}, {"i": 2}])
    assert vs.count() == 3
    assert vs.similarity_search("FastAPI 백엔드", k=1) == ["Python FastAPI 백엔드"]
    assert len(vs.similarity_search("아무거나", k=10)) == 3, "k 가 문서 수보다 커도 오류 없이 동작"
    vs.delete_all()
    assert vs.count() == 0


def test_add_documents_appends_without_id_collision():
    vs = VectorStore()
    vs.add_documents(["문서"] * 5)
    vs.add_documents(["문서"] * 5)
    assert vs.count() == 10


def test_store_without_metadata_does_not_crash():
    """chromadb 1.x 는 빈 metadata dict 를 거부 — 기본값으로 채워야 함 (#021)."""
    assert ProfileLoader().chunk_and_store("섹션 없는 짧은 자기소개") == 1
    VectorStore().add_documents(["메타데이터 없음"], metadatas=[{}])
    assert VectorStore().count() == 2


def test_load_from_dict_with_clear_replaces_profile(loaded_profile):
    before = loaded_profile.get_stored_count()
    loaded_profile.load_from_dict({"이름": "새 지원자", "기술": ["Go"]}, clear_existing=True)
    after = VectorStore().get_all_documents()
    assert before > 0
    assert all("테스트 지원자" not in d for d in after)


def test_profile_search_returns_top_k(loaded_profile):
    results = loaded_profile.search("FastAPI PostgreSQL 백엔드", k=2)
    assert len(results) == 2
    assert "FastAPI" in results[0]


def test_load_from_pdf(tmp_path):
    fitz = pytest.importorskip("fitz")
    pdf_path = tmp_path / "resume.pdf"
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Experience: Startup backend 3 years\nSkills: Python, FastAPI")
    doc.save(str(pdf_path))
    doc.close()

    loader = ProfileLoader()
    assert loader.load_from_pdf(pdf_path) >= 1
    assert "Startup backend" in "\n".join(VectorStore().get_all_documents())

    with pytest.raises(FileNotFoundError):
        loader.load_from_pdf(tmp_path / "missing.pdf")
