"""
frontend/app.py

GoodJob — AI 채용 매칭 & 이력서 생성 Streamlit 앱
백엔드(FastAPI) 없이 LangGraph 에이전트를 직접 호출합니다.

실행:
    streamlit run frontend/app.py
"""

from __future__ import annotations

import sys
import os
import time
import tempfile
from pathlib import Path
from typing import Any

# ── 프로젝트 루트를 sys.path에 추가 ─────────────────────────────────
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from dotenv import load_dotenv
load_dotenv(ROOT / ".env")

import streamlit as st

# ------------------------------------------------------------------ #
# 페이지 설정 (반드시 첫 번째 st 호출)                                 #
# ------------------------------------------------------------------ #

st.set_page_config(
    page_title="GoodJob · AI 이력서 생성기",
    page_icon="💼",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ------------------------------------------------------------------ #
# 커스텀 CSS                                                           #
# ------------------------------------------------------------------ #

st.markdown("""
<style>
/* 전체 배경 */
[data-testid="stAppViewContainer"] {
    background: #f8f9fb;
}
/* 사이드바 */
[data-testid="stSidebar"] {
    background: #1a1d2e;
}
[data-testid="stSidebar"] * {
    color: #e0e0e0 !important;
}
/* 카드 스타일 */
.gj-card {
    background: white;
    border-radius: 12px;
    padding: 24px;
    box-shadow: 0 2px 8px rgba(0,0,0,0.07);
    margin-bottom: 16px;
}
/* 점수 배지 */
.score-badge {
    font-size: 3.5rem;
    font-weight: 700;
    text-align: center;
    padding: 12px 0;
}
.score-high  { color: #22c55e; }
.score-mid   { color: #f59e0b; }
.score-low   { color: #ef4444; }
/* 노드 진행 */
.node-done   { color: #22c55e; font-weight: 600; }
.node-active { color: #3b82f6; font-weight: 600; }
.node-wait   { color: #9ca3af; }
/* 기술 태그 */
.skill-tag {
    display: inline-block;
    background: #e0f2fe;
    color: #0369a1;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.82rem;
}
.skill-tag-missing {
    display: inline-block;
    background: #fee2e2;
    color: #991b1b;
    border-radius: 20px;
    padding: 2px 10px;
    margin: 2px;
    font-size: 0.82rem;
}
/* 구분선 */
hr { border-color: #e5e7eb; }
</style>
""", unsafe_allow_html=True)


# ------------------------------------------------------------------ #
# 세션 상태 초기화                                                     #
# ------------------------------------------------------------------ #

_DEFAULTS: dict[str, Any] = {
    "page": "프로필 등록",
    "profile_indexed": False,
    "profile_chunk_count": 0,
    "company_name": "",
    "job_posting_text": "",
    "job_requirements": {},
    "fit_score": None,
    "fit_feedback": "",
    "resume_draft": "",
    "resume_final": "",
    "pipeline_errors": [],
    "pipeline_ran": False,
    # 공고 추천
    "rec_query": "",
    "rec_running": False,
    "ranked_matches": [],
    "rec_errors": [],
}

for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v


# ------------------------------------------------------------------ #
# 사이드바 내비게이션                                                   #
# ------------------------------------------------------------------ #

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown("## 💼 GoodJob")
        st.caption("AI 기반 채용 매칭 & 이력서 생성")
        st.divider()

        pages = ["프로필 등록", "공고 입력", "분석 & 이력서", "공고 추천"]
        icons  = ["👤",          "🏢",        "🚀",          "🔎"]

        selected = st.radio(
            "메뉴",
            options=pages,
            format_func=lambda p: f"{icons[pages.index(p)]}  {p}",
            index=pages.index(st.session_state["page"]),
            label_visibility="collapsed",
        )
        st.session_state["page"] = selected

        st.divider()
        st.caption("**진행 상태**")

        def _status_icon(ok: bool) -> str:
            return "✅" if ok else "⬜"

        st.write(f"{_status_icon(st.session_state['profile_indexed'])}  프로필 등록")
        st.write(f"{_status_icon(bool(st.session_state['job_posting_text']))}  채용공고 입력")
        st.write(f"{_status_icon(st.session_state['pipeline_ran'])}  분석 완료")
        st.write(f"{_status_icon(bool(st.session_state['resume_final']))}  이력서 생성")
        st.write(f"{_status_icon(bool(st.session_state['ranked_matches']))}  공고 추천")

        if st.session_state["fit_score"] is not None:
            st.divider()
            score = st.session_state["fit_score"]
            colour = "#22c55e" if score >= 0.7 else "#f59e0b" if score >= 0.4 else "#ef4444"
            st.markdown(
                f"<div style='text-align:center;font-size:2rem;color:{colour};font-weight:700;'>"
                f"{score:.0%}</div><div style='text-align:center;font-size:0.8rem;color:#9ca3af;'>적합도</div>",
                unsafe_allow_html=True,
            )

    return selected


# ------------------------------------------------------------------ #
# Page 1: 프로필 등록                                                  #
# ------------------------------------------------------------------ #

def page_profile() -> None:
    st.header("👤 프로필 등록")
    st.caption("PDF 이력서를 업로드하거나 직접 입력하세요. 벡터 DB에 저장되어 RAG 검색에 활용됩니다.")

    if st.session_state["profile_indexed"]:
        st.success(
            f"✅ 프로필이 등록되어 있습니다 "
            f"({st.session_state['profile_chunk_count']}개 청크)"
        )
        if not st.button("🔄 프로필 다시 등록"):
            return

    tab_pdf, tab_manual = st.tabs(["📄 PDF 업로드", "✏️ 직접 입력"])

    # ── PDF 탭 ───────────────────────────────────────────────────────
    with tab_pdf:
        st.info("텍스트 기반 PDF만 지원됩니다 (스캔본 이미지 불가).")
        uploaded = st.file_uploader("PDF 이력서 선택", type=["pdf"])
        if uploaded and st.button("📤 업로드 & 인덱싱", key="pdf_btn"):
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
                tmp.write(uploaded.read())
                tmp_path = tmp.name

            with st.spinner("PDF 분석 및 벡터 DB 저장 중…"):
                try:
                    from rag.profile_loader import ProfileLoader
                    loader = ProfileLoader()
                    loader.clear_profile()
                    count = loader.load_from_pdf(tmp_path)
                    _save_profile_success(count)
                except Exception as exc:
                    st.error(f"오류: {exc}")
                finally:
                    os.unlink(tmp_path)

    # ── 직접 입력 탭 ────────────────────────────────────────────────
    with tab_manual:
        with st.form("profile_form"):
            c1, c2 = st.columns(2)
            with c1:
                name  = st.text_input("이름 *", placeholder="홍길동")
                edu   = st.text_input("학력", placeholder="○○대학교 컴퓨터공학과 졸업 (2020)")
                certs = st.text_input("자격증", placeholder="정보처리기사, AWS SAA")
            with c2:
                skills = st.text_area(
                    "기술 스택 * (쉼표 구분)",
                    placeholder="Python, FastAPI, LangChain, Docker, PostgreSQL",
                    height=108,
                )

            exp = st.text_area(
                "경력 사항",
                height=140,
                placeholder=(
                    "스타트업 A (2022~2024) 백엔드 개발자\n"
                    "  - FastAPI 서버 개발, 트래픽 40% 개선\n\n"
                    "스타트업 B (2020~2022) 풀스택 개발자\n"
                    "  - Django + React 개발"
                ),
            )
            proj = st.text_area(
                "주요 프로젝트",
                height=110,
                placeholder=(
                    "AI 채용 분석 시스템 — LangGraph + RAG 파이프라인, GPT-4o 연동\n"
                    "실시간 알림 서비스 — WebSocket + Redis Pub/Sub, 동접 1만명"
                ),
            )
            summary = st.text_area("자기소개 요약", height=80)

            submitted = st.form_submit_button("💾 저장 & 인덱싱", type="primary")

        if submitted:
            if not name:
                st.warning("이름은 필수 항목입니다.")
            elif not skills.strip():
                st.warning("기술 스택을 최소 하나 이상 입력하세요.")
            else:
                profile_dict = {
                    "이름": name,
                    "기술": [s.strip() for s in skills.split(",") if s.strip()],
                    "경력": [line.strip() for line in exp.strip().splitlines() if line.strip()] if exp else [],
                    "프로젝트": [line.strip() for line in proj.strip().splitlines() if line.strip()] if proj else [],
                    "학력": edu or "",
                    "자격증": [c.strip() for c in certs.split(",") if c.strip()] if certs else [],
                    "자기소개": summary or "",
                }
                with st.spinner("벡터 DB 인덱싱 중…"):
                    try:
                        from rag.profile_loader import ProfileLoader
                        loader = ProfileLoader()
                        loader.clear_profile()
                        count = loader.load_from_dict(profile_dict)
                        _save_profile_success(count)
                    except Exception as exc:
                        st.error(f"오류: {exc}")


def _save_profile_success(count: int) -> None:
    if count > 0:
        st.session_state["profile_indexed"] = True
        st.session_state["profile_chunk_count"] = count
        st.success(f"✅ {count}개 청크가 벡터 DB에 저장되었습니다!")
        st.balloons()
    else:
        st.error("저장된 청크가 없습니다. 입력 내용을 확인하세요.")


# ------------------------------------------------------------------ #
# Page 2: 공고 입력                                                    #
# ------------------------------------------------------------------ #

def page_job_input() -> None:
    st.header("🏢 채용공고 입력")
    st.caption("분석할 채용공고를 입력하세요.")

    company_name = st.text_input(
        "회사명 *",
        value=st.session_state["company_name"],
        placeholder="카카오, 네이버, 토스, Line …",
    )
    st.session_state["company_name"] = company_name

    source = st.radio(
        "입력 방법",
        ["직접 붙여넣기", "이미지 업로드", "자동 검색"],
        horizontal=True,
    )

    if source == "직접 붙여넣기":
        _job_input_text()
    elif source == "이미지 업로드":
        _job_input_image(company_name)
    else:
        _job_input_search(company_name)

    # 미리보기
    if st.session_state["job_posting_text"]:
        with st.expander("📋 입력된 채용공고 미리보기", expanded=False):
            st.text(st.session_state["job_posting_text"][:1200] + (
                "\n…(이하 생략)" if len(st.session_state["job_posting_text"]) > 1200 else ""
            ))
        st.success(
            f"✅ 채용공고 {len(st.session_state['job_posting_text'])}자 준비 완료"
        )
        if st.button("🚀 분석 & 이력서 생성 시작 →", type="primary"):
            st.session_state["page"] = "분석 & 이력서"
            st.rerun()


def _job_input_text() -> None:
    text = st.text_area(
        "채용공고 전문 붙여넣기",
        value=st.session_state["job_posting_text"],
        height=320,
        placeholder="공고 텍스트를 전부 붙여넣으세요…",
    )
    if st.button("확인", key="txt_confirm"):
        if text.strip():
            st.session_state["job_posting_text"] = text
            st.rerun()
        else:
            st.warning("공고 내용을 입력하세요.")


def _job_input_image(company_name: str) -> None:
    st.info("공고 스크린샷을 업로드하면 OpenAI Vision으로 텍스트를 추출합니다.")
    img_file = st.file_uploader(
        "이미지 업로드 (PNG / JPG / WEBP)",
        type=["png", "jpg", "jpeg", "webp"],
        key="img_upload",
    )
    if img_file and st.button("🔍 이미지 분석", key="img_btn"):
        with tempfile.NamedTemporaryFile(
            suffix=Path(img_file.name).suffix, delete=False
        ) as tmp:
            tmp.write(img_file.read())
            tmp_path = tmp.name

        with st.spinner("Vision AI로 분석 중…"):
            try:
                from search.image_parser import ImageParser
                parser = ImageParser()
                result = parser.parse_job_posting_image(tmp_path)

                detected_company = result.get("company_name", "")
                raw_text = result.get("raw_text", "")

                if detected_company and not company_name:
                    st.session_state["company_name"] = detected_company
                if raw_text:
                    st.session_state["job_posting_text"] = raw_text
                    st.success(f"✅ 분석 완료! 회사: {detected_company or '미확인'}")
                    st.rerun()
                else:
                    st.error("텍스트를 추출하지 못했습니다.")
            except Exception as exc:
                st.error(f"이미지 분석 오류: {exc}")
            finally:
                os.unlink(tmp_path)


def _job_input_search(company_name: str) -> None:
    role = st.text_input("직군 키워드", value="백엔드 엔지니어", key="search_role")
    if st.button("🔎 채용공고 검색", disabled=not company_name, key="search_btn"):
        with st.spinner(f"'{company_name}' 채용 정보 검색 중… (Tavily API)"):
            try:
                from search.pipeline import JobSearchPipeline
                pipeline = JobSearchPipeline()
                result = pipeline.run(
                    company_name=company_name,
                    role=role,
                    fetch_detail=False,
                )
                posting_text = result.get("job_posting_text", "")
                if posting_text:
                    st.session_state["job_posting_text"] = posting_text
                    with st.expander("검색된 회사 정보"):
                        st.write(result.get("summary", ""))
                    st.success("✅ 채용공고 검색 완료!")
                    st.rerun()
                else:
                    st.warning("검색 결과가 없습니다. 직접 붙여넣기를 사용해 주세요.")
            except Exception as exc:
                st.error(f"검색 오류: {exc}")


# ------------------------------------------------------------------ #
# Page 3: 분석 & 이력서 생성                                           #
# ------------------------------------------------------------------ #

def page_analysis() -> None:
    st.header("🚀 분석 & 이력서 생성")

    # ── 전제조건 확인 ────────────────────────────────────────────────
    ready = True
    if not st.session_state["profile_indexed"]:
        st.warning("👤 프로필을 먼저 등록해 주세요.")
        ready = False
    if not st.session_state["job_posting_text"]:
        st.warning("🏢 채용공고를 먼저 입력해 주세요.")
        ready = False

    if not ready:
        return

    # ── 실행 버튼 ────────────────────────────────────────────────────
    if not st.session_state["pipeline_ran"]:
        col_l, col_r = st.columns([2, 1])
        with col_l:
            st.info(
                f"**{st.session_state['company_name'] or '(회사명 미입력)'}** 채용공고를 분석합니다.\n\n"
                "LangGraph 파이프라인 (5단계) 실행 후 이력서를 생성합니다."
            )
        with col_r:
            run_btn = st.button("▶ 파이프라인 실행", type="primary", use_container_width=True)

        if run_btn:
            _run_pipeline()
            st.rerun()
        return

    # ── 결과 표시 ────────────────────────────────────────────────────
    _render_results()

    st.divider()
    if st.button("🔄 다시 분석", use_container_width=False):
        for k in ["pipeline_ran", "fit_score", "fit_feedback",
                  "resume_draft", "resume_final", "pipeline_errors", "job_requirements"]:
            st.session_state[k] = _DEFAULTS[k]
        st.rerun()


# ------------------------------------------------------------------ #
# 파이프라인 실행 (노드별 진행 표시)                                    #
# ------------------------------------------------------------------ #

def _run_pipeline() -> None:
    NODES = [
        ("job_parser",    "📋 채용공고 파싱"),
        ("rag_retriever", "🔍 경험 검색 (RAG)"),
        ("fit_analyzer",  "📊 적합도 분석"),
        ("resume_writer", "✍️  이력서 초안 작성"),
        ("reviewer",      "🔎 이력서 검토 & 완성"),
    ]

    node_status = {n: "wait" for n, _ in NODES}
    status_placeholders = {}

    st.subheader("파이프라인 진행")
    cols = st.columns(len(NODES))
    for i, (node_id, label) in enumerate(NODES):
        with cols[i]:
            status_placeholders[node_id] = st.empty()
            status_placeholders[node_id].markdown(
                f"<div class='node-wait'>⬜ {label}</div>", unsafe_allow_html=True
            )

    progress_bar = st.progress(0)
    status_text  = st.empty()

    def _update_node(node_id: str, state: str, label: str) -> None:
        icon = {"active": "🔵", "done": "✅", "skip": "⏭️"}.get(state, "⬜")
        css_class = {"active": "node-active", "done": "node-done", "skip": "node-wait"}.get(state, "node-wait")
        status_placeholders[node_id].markdown(
            f"<div class='{css_class}'>{icon} {label}</div>", unsafe_allow_html=True
        )

    try:
        from agents.job_parser    import job_parser_node
        from agents.rag_retriever import rag_retriever_node
        from agents.fit_analyzer  import fit_analyzer_node
        from agents.resume_writer import resume_writer_node
        from agents.reviewer      import reviewer_node

        state: dict = {
            "company_name":    st.session_state["company_name"] or "지원 회사",
            "job_posting_raw": st.session_state["job_posting_text"],
            "errors":          [],
        }

        # ── 노드 1: job_parser ──────────────────────────────────────
        _update_node("job_parser", "active", "채용공고 파싱")
        status_text.info("채용공고를 구조화된 요구사항으로 파싱하는 중…")
        progress_bar.progress(10)
        state = job_parser_node(state)
        _update_node("job_parser", "done", "채용공고 파싱")
        progress_bar.progress(20)

        # ── 노드 2: rag_retriever ───────────────────────────────────
        _update_node("rag_retriever", "active", "경험 검색 (RAG)")
        status_text.info("벡터 DB에서 관련 경험을 검색하는 중…")
        progress_bar.progress(30)
        state = rag_retriever_node(state)
        _update_node("rag_retriever", "done", "경험 검색 (RAG)")
        progress_bar.progress(45)

        # ── 노드 3: fit_analyzer ────────────────────────────────────
        _update_node("fit_analyzer", "active", "적합도 분석")
        status_text.info("AI가 적합도를 계산하는 중…")
        progress_bar.progress(55)
        state = fit_analyzer_node(state)
        _update_node("fit_analyzer", "done", "적합도 분석")
        progress_bar.progress(65)

        fit_score: float = state.get("fit_score") or 0.0

        if fit_score < 0.3:
            _update_node("resume_writer", "skip", "이력서 작성 (건너뜀)")
            _update_node("reviewer",      "skip", "이력서 검토 (건너뜀)")
            progress_bar.progress(100)
            status_text.warning(
                f"적합도가 {fit_score:.0%}로 낮아 이력서 생성을 건너뜁니다. "
                "프로필이나 공고를 바꿔보세요."
            )
        else:
            # ── 노드 4: resume_writer (스트리밍) ─────────────────────
            from agents.resume_writer import stream_resume_writer
            from agents.reviewer      import stream_reviewer

            _update_node("resume_writer", "active", "이력서 초안 작성")
            status_text.info("채용공고에 맞춤화된 이력서를 작성하는 중…")
            progress_bar.progress(75)

            st.markdown("**✍️ 이력서 초안 생성 중…**")
            resume_draft = st.write_stream(stream_resume_writer(state))
            state["resume_draft"] = resume_draft

            _update_node("resume_writer", "done", "이력서 초안 작성")
            progress_bar.progress(88)

            # ── 노드 5: reviewer (스트리밍) ───────────────────────────
            _update_node("reviewer", "active", "이력서 검토 & 완성")
            status_text.info("AI가 이력서를 교정하는 중…")

            st.markdown("**🔎 최종 이력서 교정 중…**")
            resume_final = st.write_stream(stream_reviewer(state))
            state["resume_final"] = resume_final
            # errors는 streaming 경로에서는 노드를 건너뛰므로 기존 유지
            state.setdefault("errors", [])

            _update_node("reviewer", "done", "이력서 검토 & 완성")
            progress_bar.progress(100)
            status_text.success("✅ 파이프라인 완료!")

        # ── 세션에 저장 ──────────────────────────────────────────────
        st.session_state.update({
            "fit_score":        fit_score,
            "fit_feedback":     state.get("fit_feedback", ""),
            "job_requirements": state.get("job_requirements", {}),
            "resume_draft":     state.get("resume_draft", ""),
            "resume_final":     state.get("resume_final", ""),
            "pipeline_errors":  state.get("errors", []),
            "pipeline_ran":     True,
        })

    except Exception as exc:
        st.error(f"파이프라인 오류: {exc}")
        st.exception(exc)


# ------------------------------------------------------------------ #
# PDF 생성 헬퍼                                                        #
# ------------------------------------------------------------------ #

@st.cache_data(show_spinner=False)
def _generate_pdf(resume_text: str, company: str) -> bytes | None:
    """이력서 Markdown을 PDF 바이트로 변환합니다. 실패 시 None 반환."""
    try:
        from frontend.pdf_exporter import markdown_to_pdf_bytes
        return markdown_to_pdf_bytes(resume_text, company_name=company)
    except Exception as exc:
        st.warning(f"PDF 생성 실패: {exc}")
        return None


# ------------------------------------------------------------------ #
# 결과 렌더링                                                          #
# ------------------------------------------------------------------ #

def _render_results() -> None:
    fit_score    = st.session_state["fit_score"]
    fit_feedback = st.session_state["fit_feedback"]
    job_req      = st.session_state["job_requirements"]
    resume_final = st.session_state["resume_final"]
    resume_draft = st.session_state["resume_draft"]
    errors       = st.session_state["pipeline_errors"]
    company      = st.session_state["company_name"] or "회사"

    # ── 적합도 섹션 ──────────────────────────────────────────────────
    st.subheader("📊 적합도 분석 결과")
    col_score, col_detail = st.columns([1, 2])

    with col_score:
        if fit_score is not None:
            colour = "#22c55e" if fit_score >= 0.7 else "#f59e0b" if fit_score >= 0.4 else "#ef4444"
            grade  = "높음" if fit_score >= 0.7 else "보통" if fit_score >= 0.4 else "낮음"
            st.markdown(
                f"<div style='background:white;border-radius:12px;padding:20px;text-align:center;"
                f"box-shadow:0 2px 8px rgba(0,0,0,0.07);'>"
                f"<div style='font-size:0.85rem;color:#6b7280;margin-bottom:4px;'>{company} 적합도</div>"
                f"<div style='font-size:4rem;font-weight:800;color:{colour};line-height:1.1;'>{fit_score:.0%}</div>"
                f"<div style='font-size:0.9rem;color:{colour};font-weight:600;'>● {grade}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.progress(float(fit_score))

    with col_detail:
        if fit_feedback:
            st.markdown("**AI 분석 피드백**")
            st.write(fit_feedback)

    # ── 기술 매칭 시각화 ──────────────────────────────────────────────
    req_skills  = job_req.get("required_skills", [])
    pref_skills = job_req.get("preferred_skills", [])

    if req_skills or pref_skills:
        st.markdown("---")
        st.subheader("🔧 기술 스택 분석")

        # 프로필에 저장된 문서에서 기술 키워드 추출 (간단히 profile_loader count 기반)
        try:
            from rag.profile_loader import ProfileLoader
            loader = ProfileLoader()
            # 각 기술에 대해 검색 결과 있으면 보유로 간주
            owned_skills: set[str] = set()
            for sk in req_skills + pref_skills:
                results = loader.search(sk, k=1)
                if results:
                    owned_skills.add(sk)
        except Exception:
            owned_skills = set()

        col_req, col_pref = st.columns(2)
        with col_req:
            st.markdown("**필수 기술**")
            if req_skills:
                badges = "".join(
                    f"<span class='skill-tag'>✓ {s}</span>"
                    if s in owned_skills
                    else f"<span class='skill-tag-missing'>✗ {s}</span>"
                    for s in req_skills
                )
                st.markdown(badges, unsafe_allow_html=True)
                matched = sum(1 for s in req_skills if s in owned_skills)
                st.caption(f"매칭: {matched}/{len(req_skills)}")
        with col_pref:
            st.markdown("**우대 기술**")
            if pref_skills:
                badges = "".join(
                    f"<span class='skill-tag'>✓ {s}</span>"
                    if s in owned_skills
                    else f"<span class='skill-tag-missing'>✗ {s}</span>"
                    for s in pref_skills
                )
                st.markdown(badges, unsafe_allow_html=True)

    # ── 이력서 섹션 ──────────────────────────────────────────────────
    if resume_final or resume_draft:
        st.markdown("---")
        st.subheader("📄 생성된 이력서")

        tab_final, tab_draft, tab_side = st.tabs(["✨ 최종 이력서", "📝 초안", "🆚 비교 보기"])

        with tab_final:
            if resume_final:
                # ── 다운로드 버튼 영역 ────────────────────────────────
                dl_md, dl_pdf, _ = st.columns([1, 1, 2])
                with dl_md:
                    st.download_button(
                        label="⬇️ Markdown",
                        data=resume_final,
                        file_name=f"resume_{company}.md",
                        mime="text/markdown",
                        use_container_width=True,
                    )
                with dl_pdf:
                    pdf_bytes = _generate_pdf(resume_final, company)
                    if pdf_bytes:
                        st.download_button(
                            label="⬇️ PDF",
                            data=pdf_bytes,
                            file_name=f"resume_{company}.pdf",
                            mime="application/pdf",
                            use_container_width=True,
                        )
                st.markdown(resume_final)
            else:
                st.info("적합도가 낮아 이력서가 생성되지 않았습니다.")

        with tab_draft:
            if resume_draft:
                st.caption("검토 전 초안입니다. 최종 이력서와 비교해 보세요.")
                st.markdown(resume_draft)
            else:
                st.info("초안이 없습니다.")

        with tab_side:
            if resume_draft and resume_final:
                c_left, c_right = st.columns(2)
                with c_left:
                    st.markdown("**📝 초안**")
                    st.markdown(resume_draft)
                with c_right:
                    st.markdown("**✨ 최종본 (AI 교정 후)**")
                    st.markdown(resume_final)
            else:
                st.info("초안과 최종본이 모두 있어야 비교가 가능합니다.")

    # ── 오류 표시 ────────────────────────────────────────────────────
    if errors:
        with st.expander("⚠️ 파이프라인 경고 / 오류"):
            for err in errors:
                st.warning(err)


# ------------------------------------------------------------------ #
# 메인 라우터                                                          #
# ------------------------------------------------------------------ #

# ------------------------------------------------------------------ #
# Page 4: 공고 추천                                                    #
# ------------------------------------------------------------------ #

def page_job_recommend() -> None:
    st.header("🔎 내 프로필에 맞는 공고 추천")
    st.caption("키워드를 입력하면 AI가 채용공고를 수집·분석하여 적합도 순으로 추천합니다.")

    if not st.session_state["profile_indexed"]:
        st.warning("👤 프로필을 먼저 등록해 주세요.")
        return

    # ── 검색 입력 ────────────────────────────────────────────────────
    col_input, col_btn = st.columns([3, 1])
    with col_input:
        query = st.text_input(
            "직무 키워드",
            value=st.session_state["rec_query"],
            placeholder="Python 백엔드, AI 엔지니어, 데이터 사이언티스트 …",
            label_visibility="collapsed",
        )
    with col_btn:
        run_btn = st.button("🔍 공고 탐색", type="primary", use_container_width=True)

    if run_btn:
        if not query.strip():
            st.warning("키워드를 입력하세요.")
        else:
            st.session_state["rec_query"] = query
            st.session_state["ranked_matches"] = []
            st.session_state["rec_errors"] = []
            _run_recommender(query)
            st.rerun()

    # ── 결과 표시 ────────────────────────────────────────────────────
    matches = st.session_state["ranked_matches"]
    if not matches:
        if st.session_state["rec_query"]:
            st.info("결과가 없거나 탐색 중입니다.")
        return

    st.success(f"✅ '{st.session_state['rec_query']}' 관련 공고 {len(matches)}개 추천")
    st.divider()

    for match in matches:
        rank        = match.get("rank", "?")
        title       = match.get("title", "제목 없음")
        company     = match.get("company", "")
        url         = match.get("url", "")
        fit_score   = match.get("fit_score", 0.0)
        feedback    = match.get("fit_feedback", "")
        req         = match.get("requirements", {})

        colour = "#22c55e" if fit_score >= 0.7 else "#f59e0b" if fit_score >= 0.4 else "#ef4444"

        with st.container():
            # 헤더 행
            h_col, s_col = st.columns([4, 1])
            with h_col:
                link = f"[{title}]({url})" if url else title
                st.markdown(f"### {rank}위 · {link}")
                if company:
                    st.caption(f"🏢 {company}")
            with s_col:
                st.markdown(
                    f"<div style='text-align:center;background:white;border-radius:10px;"
                    f"padding:10px;box-shadow:0 2px 6px rgba(0,0,0,0.07);'>"
                    f"<div style='font-size:1.8rem;font-weight:800;color:{colour};'>{fit_score:.0%}</div>"
                    f"<div style='font-size:0.75rem;color:#9ca3af;'>적합도</div>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.progress(float(fit_score))

            # 기술 배지
            req_skills = req.get("required_skills", [])
            if req_skills:
                badges = "".join(
                    f"<span class='skill-tag'>{s}</span>" for s in req_skills[:8]
                )
                st.markdown(badges, unsafe_allow_html=True)

            # 공고 상세
            with st.expander("📋 상세 분석 보기"):
                detail_cols = st.columns(2)
                with detail_cols[0]:
                    st.markdown("**공고 정보**")
                    if req.get("job_title"):
                        st.write(f"직무: {req['job_title']}")
                    if req.get("experience_years") is not None:
                        st.write(f"경력: {req['experience_years']}년 이상")
                    if req.get("location"):
                        st.write(f"위치: {req['location']}")
                    if req.get("remote_policy"):
                        st.write(f"근무: {req['remote_policy']}")
                    if req.get("job_type"):
                        st.write(f"고용형태: {req['job_type']}")
                    if req.get("salary_range"):
                        st.write(f"급여: {req['salary_range']}")
                with detail_cols[1]:
                    st.markdown("**AI 분석**")
                    st.write(feedback[:300] + ("…" if len(feedback) > 300 else ""))

            # 이 공고로 이력서 생성 버튼
            if st.button(
                f"📄 이 공고로 이력서 생성",
                key=f"resume_btn_{rank}",
                use_container_width=False,
            ):
                st.session_state["company_name"] = company or title[:20]
                st.session_state["job_posting_text"] = match.get("raw_text", "")
                st.session_state["pipeline_ran"] = False
                for k in ["fit_score", "fit_feedback", "resume_draft",
                          "resume_final", "pipeline_errors", "job_requirements"]:
                    st.session_state[k] = _DEFAULTS[k]
                st.session_state["page"] = "분석 & 이력서"
                st.rerun()

            st.divider()

    if st.session_state["rec_errors"]:
        with st.expander("⚠️ 오류"):
            for e in st.session_state["rec_errors"]:
                st.warning(e)


def _run_recommender(query: str) -> None:
    """job_recommender_node를 실행하고 결과를 세션에 저장합니다."""
    with st.spinner(f"'{query}' 공고 탐색 중… (Tavily 검색 + Structured Output 분석)"):
        try:
            from agents.job_recommender import job_recommender_node
            mock_state = {
                "recommendation_query": query,
                "errors": [],
            }
            result = job_recommender_node(mock_state)
            st.session_state["ranked_matches"] = result.get("ranked_matches", [])
            st.session_state["rec_errors"]     = result.get("errors", [])
        except Exception as exc:
            st.error(f"추천 오류: {exc}")
            st.exception(exc)


# ------------------------------------------------------------------ #
# 메인 라우터                                                          #
# ------------------------------------------------------------------ #

def main() -> None:
    page = render_sidebar()

    if page == "프로필 등록":
        page_profile()
    elif page == "공고 입력":
        page_job_input()
    elif page == "분석 & 이력서":
        page_analysis()
    elif page == "공고 추천":
        page_job_recommend()


if __name__ == "__main__":
    main()
