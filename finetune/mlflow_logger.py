"""
finetune/mlflow_logger.py

MLflow를 사용한 파인튜닝 실험 추적.

기록하는 정보
-------------
- 하이퍼파라미터 (learning_rate, batch_size, LoRA r/alpha 등)
- 학습 지표 (train/loss, eval/loss per step)
- 모델 아티팩트 경로
- 시스템 정보 (GPU 모델, VRAM)

사용 방법
---------
    from finetune.mlflow_logger import MLflowLogger
    with MLflowLogger() as mlf:
        mlf.log_params({"lr": 2e-4, "lora_r": 16})
        mlf.log_metric("train/loss", 0.42, step=10)
        mlf.log_model_path("finetune/outputs/goodjob-lora")
"""

from __future__ import annotations

import logging
import platform
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)


class MLflowLogger:
    """MLflow 실험 로거 (context manager 지원)."""

    def __init__(
        self,
        run_name: Optional[str] = None,
        tags: Optional[dict[str, str]] = None,
    ) -> None:
        from config.settings import settings
        self._tracking_uri = settings.MLFLOW_TRACKING_URI
        self._experiment_name = settings.MLFLOW_EXPERIMENT_NAME
        self._run_name = run_name
        self._tags = tags or {}
        self._run = None

    # ------------------------------------------------------------------ #
    # Context manager                                                      #
    # ------------------------------------------------------------------ #

    def __enter__(self) -> "MLflowLogger":
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None:
            self.end(status="FAILED")
        else:
            self.end(status="FINISHED")

    # ------------------------------------------------------------------ #
    # Lifecycle                                                            #
    # ------------------------------------------------------------------ #

    def start(self) -> None:
        try:
            import mlflow
            mlflow.set_tracking_uri(self._tracking_uri)
            mlflow.set_experiment(self._experiment_name)
            self._run = mlflow.start_run(
                run_name=self._run_name,
                tags={**self._tags, "platform": platform.system()},
            )
            logger.info(
                "[MLflow] 실험 시작 — URI: %s  Run ID: %s",
                self._tracking_uri,
                self._run.info.run_id,
            )
        except Exception as exc:
            logger.warning("[MLflow] 시작 실패 (추적 없이 계속): %s", exc)

    def end(self, status: str = "FINISHED") -> None:
        try:
            import mlflow
            mlflow.end_run(status=status)
            logger.info("[MLflow] 실험 종료 — 상태: %s", status)
        except Exception:
            pass

    # ------------------------------------------------------------------ #
    # Logging helpers                                                      #
    # ------------------------------------------------------------------ #

    def log_params(self, params: dict[str, Any]) -> None:
        try:
            import mlflow
            mlflow.log_params(params)
        except Exception as exc:
            logger.debug("[MLflow] log_params 실패: %s", exc)

    def log_metric(self, key: str, value: float, step: Optional[int] = None) -> None:
        try:
            import mlflow
            mlflow.log_metric(key, value, step=step)
        except Exception as exc:
            logger.debug("[MLflow] log_metric 실패: %s", exc)

    def log_model_path(self, model_dir: str) -> None:
        try:
            import mlflow
            path = Path(model_dir)
            if path.exists():
                mlflow.log_artifacts(str(path), artifact_path="model")
                logger.info("[MLflow] 모델 아티팩트 저장: %s", model_dir)
        except Exception as exc:
            logger.debug("[MLflow] log_model_path 실패: %s", exc)

    def log_system_info(self) -> None:
        """GPU/VRAM 정보를 MLflow에 기록합니다."""
        try:
            import torch, mlflow
            info = {
                "gpu_available": torch.cuda.is_available(),
                "gpu_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "N/A",
                "vram_gb": round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1)
                if torch.cuda.is_available() else 0,
                "python_version": platform.python_version(),
            }
            mlflow.log_params(info)
            logger.info("[MLflow] 시스템 정보 기록: %s", info)
        except Exception as exc:
            logger.debug("[MLflow] log_system_info 실패: %s", exc)
