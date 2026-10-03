"""
rag/evaluator.py

RAGAS(0.4.x)를 사용한 RAG 파이프라인 품질 자동 평가.

측정 지표
---------
- context_recall     : 정답(reference)에 필요한 정보를 검색 결과가 빠짐없이 담았는지
- context_precision  : 검색된 청크 중 실제로 정답에 쓸모 있는 청크가 상위에 있는지
- faithfulness       : 생성된 답변이 검색 컨텍스트에 근거하는지 (환각 탐지)
- answer_relevancy   : 답변이 질문에 맞는 내용인지

평가 LLM 은 gpt-4o-mini, 임베딩은 text-embedding-3-small 을 사용합니다.

사용 방법
---------
    from rag.evaluator import RAGEvaluator
    evaluator = RAGEvaluator()
    scores = evaluator.evaluate([
        {
            "user_input": "Kubernetes 운영 경험이 있나요?",
            "retrieved_contexts": ["..."],
            "response": "네, ...",
            "reference": "EKS 클러스터 3개를 운영했다.",
        },
    ])
    print(scores["mean"])   # {"context_recall": 0.9, ...}

전체 비교 실험은 ``python -m rag.eval_rag`` 를 참고하세요.
"""

from __future__ import annotations

import logging
import warnings
from typing import Any

logger = logging.getLogger(__name__)

EVAL_LLM_MODEL = "gpt-4o-mini"
EVAL_EMBEDDING_MODEL = "text-embedding-3-small"
METRIC_NAMES = ["context_recall", "context_precision", "faithfulness", "answer_relevancy"]


class RAGEvaluator:
    """RAGAS 기반 RAG 품질 평가기."""

    def __init__(self) -> None:
        from config.settings import settings
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas.embeddings import LangchainEmbeddingsWrapper
            from ragas.llms import LangchainLLMWrapper

            self._llm = LangchainLLMWrapper(
                ChatOpenAI(model=EVAL_LLM_MODEL, temperature=0, api_key=settings.OPENAI_API_KEY)
            )
            self._embeddings = LangchainEmbeddingsWrapper(
                OpenAIEmbeddings(model=EVAL_EMBEDDING_MODEL, api_key=settings.OPENAI_API_KEY)
            )

    def evaluate(self, samples: list[dict[str, Any]]) -> dict[str, Any]:
        """
        샘플 목록을 평가합니다.

        Parameters
        ----------
        samples : list[dict]
            각 dict 키: user_input, retrieved_contexts, response, reference

        Returns
        -------
        dict
            ``mean``: 지표별 평균 점수, ``per_sample``: 샘플별 점수 리스트
        """
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            from ragas import EvaluationDataset, evaluate
            from ragas.metrics import (
                Faithfulness,
                LLMContextPrecisionWithReference,
                LLMContextRecall,
                ResponseRelevancy,
            )

            metrics = [
                LLMContextRecall(llm=self._llm),
                LLMContextPrecisionWithReference(llm=self._llm),
                Faithfulness(llm=self._llm),
                ResponseRelevancy(llm=self._llm, embeddings=self._embeddings),
            ]
            dataset = EvaluationDataset.from_list(samples)
            result = evaluate(dataset, metrics=metrics, show_progress=False)

        df = result.to_pandas()
        # ragas 버전에 따라 컬럼명이 다를 수 있어 표준 이름으로 매핑
        rename = {
            "llm_context_precision_with_reference": "context_precision",
            "context_precision": "context_precision",
            "context_recall": "context_recall",
            "faithfulness": "faithfulness",
            "answer_relevancy": "answer_relevancy",
        }
        df = df.rename(columns={c: rename[c] for c in df.columns if c in rename})
        cols = [c for c in METRIC_NAMES if c in df.columns]
        mean = {c: round(float(df[c].mean()), 3) for c in cols}
        logger.info("[RAGEvaluator] 평가 완료: %s", mean)
        return {"mean": mean, "per_sample": df[["user_input", *cols]].to_dict("records")}
