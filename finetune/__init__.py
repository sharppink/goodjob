"""
finetune package

공고 파싱 sLLM 파인튜닝 (Qwen2.5-7B + Unsloth QLoRA, Google Colab 에서 학습).

- data_collector.py : 실제 공고 수집 + gpt-4o 라벨링 → 학습 JSONL
- metrics.py        : 파싱 품질 지표 (Colab·로컬 공통)
- train.py          : 학습 / 평가 / GGUF 내보내기 (Colab 노트북이 import 해서 사용)
- colab_finetune.ipynb : Colab 실행 노트북
- eval_local.py     : Ollama 에 올린 모델을 로컬에서 평가
- mlflow_logger.py  : 실험 기록

무거운 의존성(torch, unsloth)을 피하려고 여기서는 아무것도 import 하지 않습니다.
"""
