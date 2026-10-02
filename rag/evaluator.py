"""
rag/evaluator.py

RAGAS를 사용한 RAG 파이프라인 품질 자동 평가.

측정 지표
---------
- context_precision  : 검색된 컨텍스트가 실제로 답변에 도움이 됐는지
- context_recall     : 정답 생성에 필요한 컨텍스트를 빠뜨리지 않았는지
- faithfulness       : 생성된 답변이 컨텍스트에 충실한지 (환각 탐지)
- answer_relevancy   : 최종 답변이 질문과 얼마나 관련 있는지

사용 방법
---------
    from rag.evaluator import RAGEvaluator
    evaluator = RAGEvaluator()
    score = evaluator.evaluate_single(
        question="Python 백엔드 경험이 있나요?",
        answer="네, 3년간 FastAPI로 ...",
        contexts=["FastAPI 프로젝트 2021~2024 ..."],
    )
    print(score)  # {"faithfulness": 0.95, "answer_relevancy": 0.88, ...}
"""

from __future__ import annotations

import logging
from typing import Optional

logger = logging.getLogger(__name__)


class RAGEvaluator:
    """RAGAS 기반 RAG 품질 평가기."""

    def evaluate_single(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: Optional[str] = None,
    ) -> dict[str, float]:
        """
        단일 QA 쌍에 대한 RAG 품질 점수 반환.

        Parameters
        ----------
        question : str
            사용자 질문 또는 채용공고 요구 사항.
        answer : str
            RAG가 생성한 답변 또는 이력서 섹션.
        contexts : list[str]
            검색된 사용자 경험 청크들.
        ground_truth : str, optional
            정답 레퍼런스 (없으면 일부 지표 생략).

        Returns
        -------
        dict[str, float]
            각 지표의 점수 (0.0 ~ 1.0).
        """
        try:
            from ragas import evaluate
            from ragas.metrics import (
                answer_relevancy,
                context_precision,
                faithfulness,
            )
            from datasets import Dataset

            data = {
                "question": [question],
                "answer": [answer],
                "contexts": [contexts],
            }
            if ground_truth:
                data["ground_truth"] = [ground_truth]

            dataset = Dataset.from_dict(data)
            metrics = [faithfulness, answer_relevancy, context_precision]

            result = evaluate(dataset, metrics=metrics)
            scores = {k: float(v) for k, v in result.items()}
            logger.info("[RAGEvaluator] 평가 완료: %s", scores)
            return scores

        except Exception as exc:
            logger.error("[RAGEvaluator] 평가 실패: %s", exc)
            return {}

    def evaluate_batch(self, samples: list[dict]) -> list[dict[str, float]]:
        """
        여러 샘플 일괄 평가.

        Parameters
        ----------
        samples : list[dict]
            각 dict는 evaluate_single 의 파라미터와 동일한 키를 가짐.
        """
        return [self.evaluate_single(**s) for s in samples]
