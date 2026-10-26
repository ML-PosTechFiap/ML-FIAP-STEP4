"""Analise causal do lote mais degradado (spec 8.4): decomposicao, ablacao e SHAP.

As intervencoes sao validas porque o processo gerador dos dados e controlado pela simulacao.
"""

from __future__ import annotations

import html
import json
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline

from credit_monitor.config import MODEL_FEATURES, SEED, Settings
from credit_monitor.drift.detect import calibration_factor, json_safe, performance
from credit_monitor.drift.stats import feature_drift_table, psi
from credit_monitor.model.features import model_matrix
from credit_monitor.model.predict import ModelNotTrained, load_model, score
from credit_monitor.observability.logging import get_logger
from credit_monitor.observability.tracking import Tracker
from credit_monitor.pipeline import EXPERIMENT, BatchNotFound
from credit_monitor.privacy.pseudonymize import pseudonymize

REPORT_NAME = "causal_report"
RECOVERY_EPS = 1e-6  # abaixo disto o denominador da recuperacao nao tem sinal util

NOTES = (
    "As intervenções (mapeamento de quantis feature a feature e a comparação entre rótulos "
    "originais e novos) só têm leitura causal porque a simulação controla o processo gerador "
    "dos dados: sabemos exatamente o que foi alterado em X e quais rótulos foram invertidos. "
    "Com dados reais seriam apenas associações. A ablação restaura a distribuição marginal de "
    "cada feature e não preserva a dependência entre features."
)


def quantile_map(current: pd.Series, reference: pd.Series) -> pd.Series:
    """Leva `current` para a distribuicao de `reference`, preservando os NaN do lote.

    Numerica: o posto percentual de cada valor nao nulo no lote vira o quantil
    correspondente da referencia (empates desfeitos por sorteio com `SEED`, para que uma
    indicadora binaria tambem convirja). Categorica: reamostragem com as frequencias da
    referencia, com `default_rng(SEED)`.
    """
    rng = np.random.default_rng(SEED)
    present = current.notna().to_numpy()
    n = int(present.sum())
    ref_valid = reference.dropna()
    if n == 0 or ref_valid.empty:
        return pd.Series(np.nan, index=current.index, dtype="float64")

    if pd.api.types.is_numeric_dtype(reference) and not pd.api.types.is_bool_dtype(reference):
        out = pd.Series(np.nan, index=current.index, dtype="float64")
        values = current.astype("float64").to_numpy()[present]
        order = np.lexsort((rng.random(n), values))  # valor, depois sorteio nos empates
        ranks = np.empty(n)
        ranks[order] = np.arange(n)
        out[present] = np.quantile(ref_valid.astype("float64").to_numpy(), (ranks + 0.5) / n)
        return out

    freq = ref_valid.astype(object).value_counts(normalize=True)
    out = pd.Series(np.nan, index=current.index, dtype="object")
    out[present] = rng.choice(freq.index.to_numpy(dtype=object), size=n, p=freq.to_numpy())
    return out


def decompose(
    pipeline: Pipeline,
    cur_df: pd.DataFrame,
    y_original: np.ndarray,
    y_new: np.ndarray,
    reference_auc: float,
) -> dict:
    """AUC de referencia, so com a mudanca em X (rotulos originais) e efeito total."""
    scores = score(pipeline, cur_df)
    covariate_only = float(roc_auc_score(y_original, scores))
    total = float(roc_auc_score(y_new, scores))
    return {
        "auc_reference": float(reference_auc),
        "auc_covariate_only": covariate_only,
        "auc_total": total,
        "drop_covariate": float(reference_auc) - covariate_only,
        "drop_concept": covariate_only - total,
    }


def _predict(pipeline: Pipeline, X: pd.DataFrame) -> np.ndarray:
    return pipeline.predict_proba(X)[:, 1]


def _recovery(base: float | None, after: float | None, target: float | None) -> float:
    """(base - after) / (base - target); NaN quando nao ha o que recuperar."""
    if base is None or after is None or target is None:
        return float("nan")
    denominator = base - target
    if abs(denominator) < RECOVERY_EPS:
        return float("nan")
    return (base - after) / denominator


def ablation(
    pipeline: Pipeline,
    ref_X: pd.DataFrame,
    cur_X: pd.DataFrame,
    y: np.ndarray,
    ref_scores: np.ndarray,
    cutoff: float,
    features: list[str],
    *,
    reference_auc: float | None = None,
    reference_factor: float | None = None,
) -> pd.DataFrame:
    """Devolve cada feature a distribuicao de referencia (quantis) e mede a recuperacao.

    Opera sobre a `model_matrix`. `psi_recovery = (psi_base - psi_apos) / psi_base`;
    `auc_recovery = (auc_apos - auc_base) / (auc_ref - auc_base)`, `NaN` se o denominador
    for ~0 ou sem `reference_auc`. A tabela vem ordenada por `psi_recovery` decrescente.
    """
    columns = ["feature", "prediction_psi_after", "auc_after", "calibration_after",
               "psi_recovery", "auc_recovery"]
    base_scores = _predict(pipeline, cur_X)
    psi_base = psi(pd.Series(ref_scores), pd.Series(base_scores))
    auc_base = performance(y, base_scores, cutoff)["auc"]

    rows = []
    for feature in features:
        mapped = cur_X.assign(**{feature: quantile_map(cur_X[feature], ref_X[feature])})
        scores = _predict(pipeline, mapped)
        perf = performance(y, scores, cutoff, reference_factor)
        psi_after = psi(pd.Series(ref_scores), pd.Series(scores))
        rows.append({
            "feature": feature,
            "prediction_psi_after": psi_after,
            "auc_after": perf["auc"],
            "calibration_after": perf["calibration_ratio"],
            "psi_recovery": _recovery(psi_base, psi_after, 0.0),
            # (base - apos) / (base - ref) == (apos - base) / (ref - base)
            "auc_recovery": _recovery(auc_base, perf["auc"], reference_auc),
        })
    table = pd.DataFrame(rows, columns=columns)
    return table.sort_values("psi_recovery", ascending=False, na_position="last",
                             ignore_index=True)


def _feature_of(output_name: str) -> str:
    """Feature original de uma coluna de saida do `ColumnTransformer` (`prefixo__nome[_cat]`)."""
    raw = output_name.split("__", 1)[1]
    for feature in sorted(MODEL_FEATURES, key=len, reverse=True):
        if raw == feature or raw.startswith(feature + "_"):
            return feature
    raise ValueError(f"coluna de saída sem feature de origem: {output_name}")


def shap_importance(pipeline: Pipeline, X: pd.DataFrame) -> pd.Series:
    """|SHAP| medio por feature (contribuicoes nativas do XGBoost, em log-odds).

    As colunas one-hot sao somadas de volta a feature original pelos nomes de saida do
    `ColumnTransformer`. A coluna de bias e descartada.
    """
    preprocess = pipeline.named_steps["preprocess"]
    classifier = pipeline.named_steps["classifier"]
    transformed = preprocess.transform(X)
    best = getattr(classifier, "best_iteration", None)
    kwargs = {"iteration_range": (0, best + 1)} if best is not None else {}
    contribs = classifier.get_booster().predict(
        xgb.DMatrix(transformed), pred_contribs=True, **kwargs
    )[:, :-1]  # a ultima coluna e o bias
    per_column = pd.Series(
        np.abs(contribs).mean(axis=0),
        index=[_feature_of(name) for name in preprocess.get_feature_names_out()],
    )
    return per_column.groupby(level=0).sum().reindex(MODEL_FEATURES)


def _number(value: float | None, digits: int = 4) -> str:
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def _table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(cell)}</td>" for cell in row) + "</tr>"
        for row in rows
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def render_html(report: dict) -> str:
    """Pagina autocontida em portugues com as tres tabelas e a ressalva."""
    d = report["decomposition"]
    decomposition = _table(
        ["Etapa", "AUC", "Queda"],
        [
            ["Referência", _number(d["auc_reference"]), "—"],
            ["Só mudança em X (rótulos originais)", _number(d["auc_covariate_only"]),
             _number(d["drop_covariate"])],
            ["Efeito total (rótulos novos)", _number(d["auc_total"]), _number(d["drop_concept"])],
        ],
    )
    if report["ablation"]:
        ablation_html = _table(
            ["Feature", "PSI do score após", "AUC após", "Calibração após",
             "Recuperação do PSI", "Recuperação do AUC"],
            [[r["feature"], _number(r["prediction_psi_after"]), _number(r["auc_after"]),
              _number(r["calibration_after"], 3), _number(r["psi_recovery"], 3),
              _number(r["auc_recovery"], 3)] for r in report["ablation"]],
        )
    else:
        ablation_html = "<p>Nenhuma feature com drift detectado: não há o que ablacionar.</p>"
    shap_html = _table(
        ["Feature", "|SHAP| na referência", f"|SHAP| no lote {report['batch_id']}", "Variação"],
        [[r["feature"], _number(r["reference"]), _number(r["batch"]), _number(r["delta"])]
         for r in report["shap"]],
    )
    batch = html.escape(report["batch_id"])
    return f"""<!doctype html>
<html lang="pt-BR"><head><meta charset="utf-8">
<title>Análise causal do lote {batch}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; margin-bottom: 1.5rem; }}
th, td {{ border: 1px solid #ccc; padding: 6px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
.nota {{ background: #fff8e1; border-left: 4px solid #b26a00; padding: 0.5rem 1rem; }}
</style></head><body>
<h1>Análise causal do lote {batch}</h1>
<h2>Decomposição do drift</h2>
<p>A queda por mudança em X é o data drift; a diferença para o efeito total é o concept drift.</p>
{decomposition}
<h2>Ablação por quantis (features com drift)</h2>
{ablation_html}
<h2>Importância SHAP: referência vs lote</h2>
{shap_html}
<p class="nota"><strong>Ressalva.</strong> {html.escape(report["notes"])}</p>
</body></html>
"""


def _read_inputs(settings: Settings, batch_id: str):
    pipeline, meta = load_model(settings)  # antes dos dados: ModelNotTrained sobe primeiro
    batch_dir = settings.data_dir / "batches" / batch_id
    reference_file = settings.data_dir / "reference" / "test_scored.parquet"
    if not reference_file.exists():
        raise ModelNotTrained(
            f"Referência pontuada não encontrada em {reference_file}. Execute o comando `train`."
        )
    for name in ("batch.parquet", "labels_original.parquet"):
        if not (batch_dir / name).exists():
            raise BatchNotFound(
                f"Lote {batch_id} não encontrado em {batch_dir / name}. "
                "Execute `prepare` para gerar os lotes."
            )
    raw = pd.read_parquet(batch_dir / "batch.parquet")
    labels = pd.read_parquet(batch_dir / "labels_original.parquet")
    if len(labels) != len(raw) or not np.array_equal(
        labels["SK_ID_CURR"].to_numpy(), raw["SK_ID_CURR"].to_numpy()
    ):
        raise ValueError(f"Os rótulos originais do lote {batch_id} não correspondem ao lote.")
    y_original = labels["TARGET"].to_numpy()
    batch = pseudonymize(raw, settings.pseudonymization_salt)  # nenhum ID bruto adiante
    return pipeline, meta, batch, y_original, pd.read_parquet(reference_file)


def run_causal(settings: Settings, batch_id: str = "06") -> Path:
    """Gera `reports/causal/causal_report.{json,html}` e registra no MLflow.

    Devolve o caminho do HTML (`.with_suffix(".json")` e o JSON). Levanta `ModelNotTrained`
    e `BatchNotFound`.
    """
    run_id = f"causal-{batch_id}-{uuid.uuid4().hex[:8]}"
    log = get_logger(settings, run_id)
    pipeline, meta, batch, y_original, reference = _read_inputs(settings, batch_id)
    y_new = batch["TARGET"].to_numpy()
    cutoff = float(meta["cutoff"])
    reference_auc = float(meta["reference_metrics"]["auc"])
    log.info("Análise causal do lote %s (%d linhas).", batch_id, len(batch),
             extra={"batch_id": batch_id, "stage": "causal", "rows_in": len(batch)})

    decomposition = decompose(pipeline, batch, y_original, y_new, reference_auc)

    table = feature_drift_table(reference, batch)
    drifted = sorted(table.loc[table["status"] == "drift", "feature"])
    ref_X, cur_X = model_matrix(reference), model_matrix(batch)
    ablation_table = ablation(
        pipeline, ref_X, cur_X, y_new, reference["score"].to_numpy(), cutoff, drifted,
        reference_auc=reference_auc, reference_factor=calibration_factor(reference),
    )

    shap_ref, shap_cur = shap_importance(pipeline, ref_X), shap_importance(pipeline, cur_X)
    shap_table = pd.DataFrame({"feature": shap_ref.index, "reference": shap_ref.to_numpy(),
                               "batch": shap_cur.to_numpy()})
    shap_table["delta"] = shap_table["batch"] - shap_table["reference"]
    shap_table = shap_table.sort_values("delta", key=np.abs, ascending=False, ignore_index=True)

    report = json_safe({
        "batch_id": batch_id,
        "decomposition": decomposition,
        "ablation": ablation_table.to_dict("records"),
        "shap": shap_table.to_dict("records"),
        "notes": NOTES,
    })
    out_dir = settings.reports_dir / "causal"
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / f"{REPORT_NAME}.json"
    html_path = out_dir / f"{REPORT_NAME}.html"
    json_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(render_html(report), encoding="utf-8")

    tracker = Tracker(settings, logger=log)
    with tracker.run(EXPERIMENT, f"causal-{batch_id}", {"batch_id": batch_id, "kind": "causal"}):
        tracker.log_metrics(decomposition)
        tracker.log_artifact(json_path)
        tracker.log_artifact(html_path)

    log.info("Relatório causal gravado em %s.", html_path,
             extra={"batch_id": batch_id, "stage": "causal", "event": "causal_end",
                    "status": "ok", "rows_out": len(ablation_table)})
    return html_path
