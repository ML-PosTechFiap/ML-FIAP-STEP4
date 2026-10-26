"""Treino do modelo base: XGBoost com early stopping, ponto de corte por KS e auditoria."""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score, roc_curve
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBClassifier

from credit_monitor.config import (
    CATEGORICAL_FEATURES,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    XGB_PARAMS,
    Settings,
)
from credit_monitor.contracts.schema import CONTRACT_VERSION
from credit_monitor.contracts.validate import validate_batch
from credit_monitor.model.fairness import fairness_report, proxy_scores
from credit_monitor.model.features import model_matrix
from credit_monitor.model.predict import METADATA_FILE, MODEL_FILE, score
from credit_monitor.observability.logging import get_logger
from credit_monitor.observability.tracking import Tracker

EXPERIMENT = "credit-scoring-training"
REGISTERED_MODEL = "credit-scoring"
SPLITS = ("train", "valid", "test")


def build_pipeline(scale_pos_weight: float) -> Pipeline:
    """One-hot nas categoricas, numericas direto, seguidos do XGBClassifier."""
    preprocess = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )
    classifier = XGBClassifier(**XGB_PARAMS, scale_pos_weight=scale_pos_weight)
    return Pipeline([("preprocess", preprocess), ("classifier", classifier)])


def ks_statistic(y_true: np.ndarray, score: np.ndarray) -> float:
    """Estatistica KS: maximo de TPR - FPR sobre todos os limiares."""
    fpr, tpr, _ = roc_curve(y_true, score)
    return float(np.max(tpr - fpr))


def choose_cutoff(y_true: np.ndarray, score: np.ndarray) -> float:
    """Limiar que maximiza o KS. Aprova-se quem tem `score < corte`."""
    fpr, tpr, thresholds = roc_curve(y_true, score)
    cutoff = float(thresholds[int(np.argmax(tpr - fpr))])
    return cutoff if np.isfinite(cutoff) else float(np.max(score))


def _fit(pipeline: Pipeline, X_train, y_train, X_valid, y_valid) -> None:
    """Fit com early stopping: o eval_set precisa chegar ja transformado ao classificador."""
    preprocess = pipeline.named_steps["preprocess"].fit(X_train)
    pipeline.fit(
        X_train,
        y_train,
        classifier__eval_set=[(preprocess.transform(X_valid), y_valid)],
        classifier__verbose=False,
    )


def _load_splits(settings: Settings) -> dict[str, pd.DataFrame]:
    ref_dir = settings.data_dir / "reference"
    domains = json.loads((ref_dir / "domains.json").read_text(encoding="utf-8"))
    frames = {name: pd.read_parquet(ref_dir / f"{name}.parquet") for name in SPLITS}
    for name, frame in frames.items():
        result = validate_batch(frame.reset_index(drop=True), domains)
        if not result.passed:
            raise RuntimeError(
                f"A base de {name} viola o contrato de dados: {result.violations}. "
                "Execute `prepare` novamente antes de treinar."
            )
    return frames


def run_train(settings: Settings) -> dict:
    """Treina, avalia no teste, grava `models/` e o teste pontuado; devolve o metadata."""
    run_id = uuid.uuid4().hex[:12]
    log = get_logger(settings, run_id)
    frames = _load_splits(settings)
    train, valid, test = (frames[name] for name in SPLITS)
    y_train, y_valid, y_test = (f["TARGET"].to_numpy() for f in (train, valid, test))

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    pipeline = build_pipeline(scale_pos_weight)
    log.info("Treinando o modelo base (%d linhas de treino).", len(train),
             extra={"stage": "train", "rows_in": len(train)})

    tracker = Tracker(settings, logger=log)
    with tracker.run(EXPERIMENT, f"train-{run_id}", {"contract_version": CONTRACT_VERSION}):
        _fit(pipeline, model_matrix(train), y_train, model_matrix(valid), y_valid)

        cutoff = choose_cutoff(y_valid, score(pipeline, valid))
        test_score = score(pipeline, test)
        metrics = {
            "auc": float(roc_auc_score(y_test, test_score)),
            "pr_auc": float(average_precision_score(y_test, test_score)),
            "ks": ks_statistic(y_test, test_score),
            "brier": float(brier_score_loss(y_test, test_score)),
        }
        fairness = fairness_report(test, test_score < cutoff, y_test)
        proxy = proxy_scores(model_matrix(test), test["CODE_GENDER"])
        fairness["proxy_top5"] = {name: float(v) for name, v in proxy.head(5).items()}

        tracker.log_params({**XGB_PARAMS, "scale_pos_weight": scale_pos_weight,
                            "cutoff": cutoff, "contract_version": CONTRACT_VERSION,
                            "best_iteration": pipeline.named_steps["classifier"].best_iteration})
        tracker.log_metrics(metrics)
        version = tracker.log_model(pipeline, REGISTERED_MODEL)

        metadata = {
            "features": list(MODEL_FEATURES),
            "cutoff": cutoff,
            "contract_version": CONTRACT_VERSION,
            "reference_metrics": metrics,
            "mlflow_model_version": version,
            "trained_at": datetime.now(UTC).isoformat(),
            "fairness": fairness,
        }
        settings.models_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, settings.models_dir / MODEL_FILE)
        metadata_path = settings.models_dir / METADATA_FILE
        metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2),
                                 encoding="utf-8")
        tracker.log_artifact(metadata_path)

    scored = test.copy()
    scored["score"] = test_score
    scored.to_parquet(settings.data_dir / "reference" / "test_scored.parquet", index=False)
    log.info("Modelo treinado: AUC=%.4f, KS=%.4f, corte=%.4f.", metrics["auc"], metrics["ks"],
             cutoff, extra={"stage": "train", "rows_out": len(test)})
    return metadata
