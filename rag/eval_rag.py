"""
rag/eval_rag.py

RAG 검색 설정별 품질을 RAGAS 로 측정·비교하는 실험 스크립트.

- 실제 프로필 DB 를 건드리지 않도록 임시 Chroma 디렉토리를 사용합니다.
- 청크 16개짜리 평가용 프로필 + 질문/정답 10쌍으로 평가합니다.
  (청크 12개 이하는 rag_retriever 가 검색 없이 전체를 쓰므로, 검색 자체를 평가하려면 더 커야 함)
- 비교 설정:
    A) vector_top5     : 벡터 검색 상위 5개
    B) rerank_top5     : 벡터 상위 20개 → BGE Reranker 상위 5개
    C) vector_top3     : 벡터 검색 상위 3개 (top-k 축소 영향 확인)

실행:
    python -m rag.eval_rag                 # 전체 설정 비교
    python -m rag.eval_rag --configs B     # 특정 설정만

결과는 rag/eval_results/ 에 JSON 으로 저장됩니다. 비용: gpt-4o-mini 기준 1회 수십 원 수준.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime
from pathlib import Path

# 평가 전용 임시 DB. 환경변수로 바꾸는 방식은 `python -m rag.eval_rag` 실행 시
# rag/__init__.py 가 먼저 settings 를 만들어 버려 무시됨 (PROJECT_DOCS #009)
# → VectorStore 를 이 경로로 명시적으로 생성해서 넘긴다.
_TMP_DB = tempfile.mkdtemp(prefix="goodjob_eval_chroma_")

RESULTS_DIR = Path(__file__).parent / "eval_results"

EVAL_PROFILE: list[str] = [
    "이름: 이영희 / 백엔드·데이터 엔지니어 5년차",
    "경력: 핀테크 B사 (2021~2025) 백엔드 개발자 - Spring Boot 기반 결제 승인 API 개발, 일 거래 200만 건 처리",
    "경력: 핀테크 B사 - 결제 승인 API p99 지연을 1.2초에서 300ms로 단축 (커넥션 풀 튜닝, 비동기 로깅 전환)",
    "경력: 핀테크 B사 - Kafka 기반 이벤트 파이프라인 구축, 정산 배치를 실시간 스트리밍으로 전환해 정산 지연 하루 → 10분",
    "경력: 핀테크 B사 - AWS EKS 클러스터 3개 운영, Helm 차트 표준화로 신규 서비스 배포 준비 시간 2일 → 2시간",
    "경력: 이커머스 C사 (2019~2021) 주니어 백엔드 개발자 - Django REST Framework로 상품 검색 API 개발",
    "경력: 이커머스 C사 - Elasticsearch 도입으로 상품 검색 응답 800ms → 120ms, 검색 전환율 15% 증가",
    "프로젝트: 사내 LLM 상담봇 - LangChain + OpenAI API로 고객 문의 자동 분류, 상담원 1차 응대 업무 40% 절감",
    "프로젝트: 이상거래 탐지 모델 - XGBoost로 이상 결제 탐지, 정밀도 0.91 / 재현율 0.84, 월 손실액 30% 감소",
    "기술: Java, Spring Boot, Python, Django, FastAPI, Kafka, Elasticsearch, PostgreSQL, Redis",
    "기술: AWS (EKS, RDS, S3), Kubernetes, Helm, Terraform, GitHub Actions, Prometheus, Grafana",
    "협업: 결제팀 4명 리드 — 주 1회 기술 공유 세션 운영, 코드 리뷰 문화 정착 (PR 리뷰 평균 24시간 → 4시간)",
    "장애 대응: 블랙프라이데이 트래픽 5배 급증 시 Redis 캐시 히트율 개선과 오토스케일링 조정으로 무장애 운영",
    "학력: 한국대학교 컴퓨터공학과 졸업 (2019)",
    "자격증: 정보처리기사, AWS Solutions Architect Associate, CKA (Certified Kubernetes Administrator)",
    "대외활동: 오픈소스 Kafka 커넥터 프로젝트 컨트리뷰터, 사내 기술 블로그 글 12편 작성",
]

EVAL_QA: list[dict[str, str]] = [
    {"q": "Kubernetes 운영 경험이 있나요?",
     "ref": "핀테크 B사에서 AWS EKS 클러스터 3개를 운영했고 Helm 차트 표준화로 배포 준비 시간을 2일에서 2시간으로 줄였다. CKA 자격증을 보유하고 있다."},
    {"q": "대용량 트래픽 처리 경험을 설명해 주세요.",
     "ref": "일 거래 200만 건을 처리하는 결제 승인 API를 개발했고, 블랙프라이데이에 트래픽이 5배 급증했을 때 Redis 캐시 히트율 개선과 오토스케일링 조정으로 무장애 운영했다."},
    {"q": "API 성능을 개선한 경험이 있나요?",
     "ref": "결제 승인 API p99 지연을 1.2초에서 300ms로 단축했고, Elasticsearch 도입으로 상품 검색 응답을 800ms에서 120ms로 줄였다."},
    {"q": "메시지 큐나 스트리밍 처리 경험이 있나요?",
     "ref": "Kafka 기반 이벤트 파이프라인을 구축해 정산 배치를 실시간 스트리밍으로 전환했고 정산 지연을 하루에서 10분으로 줄였다. 오픈소스 Kafka 커넥터 프로젝트에도 기여했다."},
    {"q": "LLM이나 생성형 AI 관련 경험이 있나요?",
     "ref": "LangChain과 OpenAI API로 고객 문의를 자동 분류하는 사내 LLM 상담봇을 만들어 상담원 1차 응대 업무를 40% 줄였다."},
    {"q": "머신러닝 모델 개발 경험이 있나요?",
     "ref": "XGBoost로 이상 결제 탐지 모델을 개발해 정밀도 0.91, 재현율 0.84를 달성했고 월 손실액을 30% 줄였다."},
    {"q": "팀 리딩이나 협업 경험을 알려주세요.",
     "ref": "결제팀 4명을 리드하며 주 1회 기술 공유 세션을 운영했고 코드 리뷰 문화를 정착시켜 PR 리뷰 시간을 평균 24시간에서 4시간으로 줄였다."},
    {"q": "검색 엔진 관련 경험이 있나요?",
     "ref": "이커머스 C사에서 Elasticsearch를 도입해 상품 검색 응답을 800ms에서 120ms로 줄이고 검색 전환율을 15% 높였다."},
    {"q": "Python 웹 프레임워크 경험은 어느 정도인가요?",
     "ref": "이커머스 C사에서 Django REST Framework로 상품 검색 API를 개발했고, 기술 스택에 Python, Django, FastAPI가 있다."},
    {"q": "보유한 클라우드 관련 자격증과 인프라 자동화 경험은?",
     "ref": "AWS Solutions Architect Associate와 CKA 자격증을 보유했고 Terraform, Helm, GitHub Actions를 사용한다."},
]

CONFIGS: dict[str, dict] = {
    "A": {"name": "vector_top5", "reranker": False, "k": 5},
    "B": {"name": "rerank_top5", "reranker": True, "k": 5},
    "C": {"name": "vector_top3", "reranker": False, "k": 3},
}

ANSWER_SYSTEM = (
    "당신은 채용 담당자의 질문에 지원자 프로필만 근거로 답하는 어시스턴트입니다. "
    "컨텍스트에 없는 내용은 말하지 말고, 없으면 '프로필에서 확인되지 않습니다'라고 답하세요. "
    "2~3문장으로 답하세요."
)


def _eval_store():
    """임시 경로의 VectorStore. 실제 프로필 DB 경로면 즉시 중단."""
    from config.settings import settings
    from rag.vectorstore import VectorStore

    if os.path.abspath(_TMP_DB) == os.path.abspath(settings.CHROMA_PERSIST_DIR):
        raise RuntimeError("평가가 실제 프로필 DB 경로를 사용하려 합니다 — 중단합니다.")
    return VectorStore(persist_dir=_TMP_DB)


def _build_index() -> None:
    vs = _eval_store()
    vs.add_documents(EVAL_PROFILE, metadatas=[{"source": "eval"} for _ in EVAL_PROFILE])
    assert vs.count() == len(EVAL_PROFILE), "임시 DB가 비어 있지 않습니다."


def _run_config(cfg: dict) -> list[dict]:
    from config.settings import settings
    from llm.openai_client import FAST_MODEL, OpenAIClient
    from rag.profile_loader import ProfileLoader

    settings.RERANKER_ENABLED = cfg["reranker"]
    loader = ProfileLoader(vectorstore=_eval_store())
    client = OpenAIClient(model=FAST_MODEL)

    samples = []
    for qa in EVAL_QA:
        contexts = loader.search(qa["q"], k=cfg["k"])
        context_text = "\n".join(f"- {c}" for c in contexts)
        answer = client.generate(
            prompt=f"### 지원자 프로필\n{context_text}\n\n### 질문\n{qa['q']}",
            system=ANSWER_SYSTEM,
            temperature=0,
            max_tokens=300,
        )
        samples.append({
            "user_input": qa["q"],
            "retrieved_contexts": contexts,
            "response": answer,
            "reference": qa["ref"],
        })
    return samples


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description="GoodJob RAG 품질 비교 (RAGAS)")
    parser.add_argument("--configs", default="ABC", help="실행할 설정 (예: AB)")
    args = parser.parse_args()

    from dotenv import load_dotenv
    load_dotenv()

    from rag.evaluator import RAGEvaluator

    try:
        _build_index()
        evaluator = RAGEvaluator()
        summary: dict[str, dict] = {}
        details: dict[str, list] = {}

        for key in args.configs:
            cfg = CONFIGS[key]
            print(f"[{key}] {cfg['name']} 평가 중…", flush=True)
            samples = _run_config(cfg)
            scores = evaluator.evaluate(samples)
            summary[cfg["name"]] = scores["mean"]
            details[cfg["name"]] = [
                {**s, **p} for s, p in zip(samples, scores["per_sample"])
            ]

        metric_names = list(next(iter(summary.values())).keys())
        print("\n" + "설정".ljust(14) + "".join(m.rjust(19) for m in metric_names))
        for name, means in summary.items():
            print(name.ljust(14) + "".join(f"{means.get(m, float('nan')):19.3f}" for m in metric_names))

        RESULTS_DIR.mkdir(exist_ok=True)
        out = RESULTS_DIR / f"ragas_{datetime.now():%Y%m%d_%H%M%S}.json"
        out.write_text(
            json.dumps({"summary": summary, "configs": CONFIGS, "details": details},
                       ensure_ascii=False, indent=2, default=str),
            encoding="utf-8",
        )
        print(f"\n결과 저장: {out}")
    finally:
        shutil.rmtree(_TMP_DB, ignore_errors=True)


if __name__ == "__main__":
    main()
