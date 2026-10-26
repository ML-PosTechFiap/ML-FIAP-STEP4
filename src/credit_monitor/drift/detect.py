"""Deteccao por lote: desempenho, relatorios Evidently, resumo e recomendacao (spec 6.2-6.3)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from credit_monitor.config import (
    AUC_DROP_ALERT,
    CALIBRATION_ALERT,
    CATEGORICAL_FEATURES,
    DISPARATE_IMPACT_MIN,
    DRIFT_SHARE_ALERT,
    MODEL_FEATURES,
    NUMERIC_FEATURES,
    PRED_PSI_ALERT,
    PSI_DRIFT,
)
from credit_monitor.drift.stats import feature_drift_table, psi
from credit_monitor.model.fairness import fairness_report
from credit_monitor.model.features import model_matrix
from credit_monitor.model.train import ks_statistic


def performance(
    y_true: np.ndarray,
    score: np.ndarray,
    cutoff: float,
    reference_factor: float | None = None,
) -> dict:
    """Metricas com rotulo. `auc`/`ks` sao `None` com uma so classe; aprovado = `score < cutoff`.

    `calibration_ratio_raw` = default observado / PD media prevista. O modelo usa
    `scale_pos_weight`, entao o score nao e uma probabilidade calibrada e o valor bruto ja
    nasce longe de 1. Por isso, com `reference_factor` (o mesmo quociente na referencia),
    `calibration_ratio` e relativo a ela: `bruto / reference_factor` (1,0 = calibracao igual
    a da referencia). Sem o fator, `calibration_ratio` e o proprio valor bruto.
    """
    y = np.asarray(y_true)
    s = np.asarray(score, dtype="float64")
    single_class = len(np.unique(y)) < 2
    observed = float(y.mean())
    predicted = float(s.mean())
    raw = observed / predicted if predicted > 0 else None
    relative = raw
    if raw is not None and reference_factor:
        relative = raw / reference_factor
    return {
        "auc": None if single_class else float(roc_auc_score(y, s)),
        "ks": None if single_class else ks_statistic(y, s),
        "default_rate_observed": observed,
        "mean_predicted_pd": predicted,
        "calibration_ratio": relative,
        "calibration_ratio_raw": raw,
        "approval_rate": float((s < cutoff).mean()),
    }


def calibration_factor(scored: pd.DataFrame) -> float | None:
    """Default observado / PD media prevista de um frame com `TARGET` e `score`."""
    if "TARGET" not in scored.columns or not scored["TARGET"].notna().any():
        return None
    mean_score = float(scored["score"].mean())
    if mean_score <= 0:
        return None
    return float(scored["TARGET"].mean()) / mean_score


def _alerts(summary: dict, reference_auc: float) -> tuple[list[str], list[str]]:
    """Motivos (em portugues) de retreino e de investigacao, pela regra da spec 6.3."""
    retrain: list[str] = []
    investigate: list[str] = []
    perf = summary.get("performance")
    if perf:
        auc, ratio = perf.get("auc"), perf.get("calibration_ratio")
        if auc is not None and auc < reference_auc - AUC_DROP_ALERT:
            retrain.append(
                f"AUC {auc:.3f} caiu mais de {AUC_DROP_ALERT:.2f} abaixo da referência "
                f"({reference_auc:.3f})."
            )
        if ratio is not None and ratio > CALIBRATION_ALERT:
            retrain.append(
                f"Calibração: a razão observado/previsto é {ratio:.2f}x a da referência "
                f"(limite {CALIBRATION_ALERT})."
            )
    share = summary.get("dataset_drift_share", 0.0)
    if share >= DRIFT_SHARE_ALERT:
        investigate.append(f"{share:.0%} das features com drift (limite {DRIFT_SHARE_ALERT:.0%}).")
    pred_psi = summary.get("prediction_psi", 0.0)
    if pred_psi >= PRED_PSI_ALERT:
        investigate.append(f"PSI do score {pred_psi:.3f} (limite {PRED_PSI_ALERT}).")
    for attribute, impact in (summary.get("fairness") or {}).items():
        if impact < DISPARATE_IMPACT_MIN:
            investigate.append(
                f"Impacto disparato de {attribute} {impact:.2f} (mínimo {DISPARATE_IMPACT_MIN})."
            )
    return retrain, investigate


def recommend(summary: dict, reference_auc: float) -> str:
    """`RETREINAR`, `INVESTIGAR` ou `OK` (spec 6.3). Sem `performance`, so sinais sem rotulo."""
    retrain, investigate = _alerts(summary, reference_auc)
    if retrain:
        return "RETREINAR"
    return "INVESTIGAR" if investigate else "OK"


def evidently_reports(
    ref_scored: pd.DataFrame,
    cur_scored: pd.DataFrame,
    batch_id: str,
    out_dir: Path,
    labeled: bool,
    cutoff: float | None = None,
) -> dict:
    """Relatorios Evidently (unico ponto de uso da biblioteca).

    Os frames ja devem ter as `MODEL_FEATURES` (como em `model_matrix`) e `score`; com
    `labeled`, tambem `TARGET`. So essas colunas chegam ao Evidently: sem IDs nem linhas
    extras. `cutoff` e o limiar de probabilidade do relatorio de classificacao.
    Devolve `{"paths": {...}, "psi": {coluna: PSI do Evidently}}`.
    """
    from evidently import BinaryClassification, DataDefinition, Dataset, Report
    from evidently.presets import ClassificationPreset, DataDriftPreset

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    def as_dataset(df: pd.DataFrame, columns: list[str], definition: DataDefinition) -> Dataset:
        frame = df[columns].reset_index(drop=True).copy()
        for col in CATEGORICAL_FEATURES:
            if col in frame.columns:
                frame[col] = frame[col].astype(object)
        return Dataset.from_pandas(frame, data_definition=definition)

    def save(snapshot, kind: str, paths: dict) -> None:
        html, js = out_dir / f"{batch_id}_{kind}.html", out_dir / f"{batch_id}_{kind}.json"
        snapshot.save_html(str(html))
        snapshot.save_json(str(js))
        paths[kind] = {"html": html.name, "json": js.name}

    paths: dict = {}
    drift_columns = [*MODEL_FEATURES, "score"]
    drift_definition = DataDefinition(
        numerical_columns=[*NUMERIC_FEATURES, "score"], categorical_columns=CATEGORICAL_FEATURES
    )
    drift_snapshot = Report([DataDriftPreset(method="psi", threshold=PSI_DRIFT)]).run(
        current_data=as_dataset(cur_scored, drift_columns, drift_definition),
        reference_data=as_dataset(ref_scored, drift_columns, drift_definition),
    )
    save(drift_snapshot, "data_drift", paths)

    psi_by_column: dict[str, float] = {}
    for metric in drift_snapshot.dict()["metrics"]:
        config = metric.get("config", {})
        if config.get("type", "").endswith(":ValueDrift"):
            psi_by_column[config["column"]] = float(metric["value"])

    if labeled:
        definition = DataDefinition(
            classification=[
                BinaryClassification(target="TARGET", prediction_probas="score", pos_label=1)
            ]
        )
        columns = ["TARGET", "score"]
        performance_snapshot = Report([ClassificationPreset(probas_threshold=cutoff)]).run(
            current_data=as_dataset(cur_scored, columns, definition),
            reference_data=as_dataset(ref_scored, columns, definition),
        )
        save(performance_snapshot, "performance", paths)

    return {"paths": paths, "psi": psi_by_column}


def json_safe(value):
    """Converte numpy/NaN/inf em tipos JSON validos (nao finitos viram `None`)."""
    if isinstance(value, dict):
        return {str(k): json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def analyze_batch(
    ref_scored: pd.DataFrame,
    cur_scored: pd.DataFrame,
    batch_id: str,
    meta: dict,
    out_dir: Path,
) -> dict:
    """Resumo do lote (drift, desempenho, vies, recomendacao); grava `<batch>_summary.json`.

    Os dois frames trazem as colunas brutas do modelo, `CODE_GENDER` e `score` (e `TARGET`
    no lote rotulado). O resumo nao contem linhas nem IDs.
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    labeled = "TARGET" in cur_scored.columns and bool(cur_scored["TARGET"].notna().any())

    table = feature_drift_table(ref_scored, cur_scored)
    drifted = sorted(table.loc[table["status"] == "drift", "feature"])
    cutoff = float(meta["cutoff"])

    ref_matrix = model_matrix(ref_scored).assign(score=ref_scored["score"].to_numpy())
    cur_matrix = model_matrix(cur_scored).assign(score=cur_scored["score"].to_numpy())
    if labeled:
        ref_matrix["TARGET"] = ref_scored["TARGET"].to_numpy()
        cur_matrix["TARGET"] = cur_scored["TARGET"].to_numpy()
    evidently = evidently_reports(ref_matrix, cur_matrix, batch_id, out_dir, labeled, cutoff)

    approved = cur_scored["score"].to_numpy() < cutoff
    fairness = fairness_report(cur_scored, approved, None)
    summary = {
        "batch_id": batch_id,
        "labeled": labeled,
        "features": table.to_dict("records"),
        "drifted_features": drifted,
        "dataset_drift_share": len(drifted) / len(table),
        "prediction_psi": psi(ref_scored["score"], cur_scored["score"]),
        "performance": (
            performance(
                cur_scored["TARGET"].to_numpy(),
                cur_scored["score"].to_numpy(),
                cutoff,
                reference_factor=calibration_factor(ref_scored),
            )
            if labeled
            else None
        ),
        "fairness": {name: report["disparate_impact"] for name, report in fairness.items()},
        "evidently": evidently["paths"],
        "psi": evidently["psi"],
    }
    reference_auc = meta["reference_metrics"]["auc"]
    retrain, investigate = _alerts(summary, reference_auc)
    summary["recommendation"] = recommend(summary, reference_auc)
    summary["recommendation_reasons"] = retrain + investigate

    summary = json_safe(summary)
    path = out_dir / f"{batch_id}_summary.json"
    path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    summary["summary_path"] = str(path)
    return summary
