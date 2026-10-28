import json

import numpy as np
import pandas as pd
import pytest
from scipy.stats import spearmanr

from credit_monitor.cli import main
from credit_monitor.config import CATEGORICAL_FEATURES, MODEL_FEATURES
from credit_monitor.drift.causal import quantile_map
from credit_monitor.drift.stats import psi


def test_quantile_map_restores_distribution_and_rank():
    rng = np.random.default_rng(0)
    ref = pd.Series(rng.normal(0, 1, 20_000))
    cur = pd.Series(rng.normal(2, 3, 20_000))
    cur[::10] = np.nan
    mapped = quantile_map(cur, ref)
    assert psi(ref, mapped.dropna()) < 0.02
    assert spearmanr(cur.dropna(), mapped.dropna()).correlation > 0.999
    assert mapped.isna().sum() == cur.isna().sum()


def test_quantile_map_handles_tied_values():
    """Indicadora binaria: 30% de uns no lote, 10% na referencia."""
    ref = pd.Series([0] * 9_000 + [1] * 1_000, dtype="float64")
    cur = pd.Series([0] * 7_000 + [1] * 3_000, dtype="float64")
    mapped = quantile_map(cur, ref)
    assert mapped.mean() == pytest.approx(0.10, abs=0.01)


def test_quantile_map_categorical_resamples_reference_frequencies():
    ref = pd.Series(["a"] * 800 + ["b"] * 200)
    cur = pd.Series(["b"] * 900 + ["a"] * 100 + [None] * 50, dtype="object")
    mapped = quantile_map(cur, ref)
    assert mapped.isna().sum() == 50
    assert mapped.isna().equals(cur.isna())
    share_a = (mapped.dropna() == "a").mean()
    assert share_a == pytest.approx(0.8, abs=0.04)
    pd.testing.assert_series_equal(mapped, quantile_map(cur, ref))  # rng fixo: deterministico


def test_run_causal_synthetic(trained):  # fixture de sessao da Tarefa 9
    from credit_monitor.drift.causal import run_causal

    path = run_causal(trained, "06")
    rep = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    d = rep["decomposition"]
    assert d["drop_concept"] > 0
    assert d["auc_total"] < d["auc_reference"]
    assert d["drop_covariate"] == pytest.approx(d["auc_reference"] - d["auc_covariate_only"])
    assert d["drop_concept"] == pytest.approx(d["auc_covariate_only"] - d["auc_total"])
    assert rep["batch_id"] == "06"
    assert rep["ablation"][0]["psi_recovery"] >= rep["ablation"][-1]["psi_recovery"]
    assert {r["feature"] for r in rep["shap"]} == set(MODEL_FEATURES)
    assert all(r["delta"] == pytest.approx(r["batch"] - r["reference"]) for r in rep["shap"])
    assert "simulação" in rep["notes"]
    html = path.with_suffix(".html").read_text(encoding="utf-8")
    assert "<table" in html and "simulação" in html
    # nenhum ID bruto nos relatorios
    assert "SK_ID_CURR" not in path.with_suffix(".json").read_text(encoding="utf-8")


def test_ablation_only_detected_drifted_features(trained):
    from credit_monitor.drift.causal import run_causal
    from credit_monitor.drift.stats import feature_drift_table

    path = run_causal(trained, "06")
    rep = json.loads(path.with_suffix(".json").read_text(encoding="utf-8"))
    reference = pd.read_parquet(trained.data_dir / "reference" / "test_scored.parquet")
    from credit_monitor.privacy.pseudonymize import pseudonymize

    batch = pseudonymize(
        pd.read_parquet(trained.data_dir / "batches" / "06" / "batch.parquet"),
        trained.pseudonymization_salt,
    )
    table = feature_drift_table(reference, batch)
    expected = set(table.loc[table["status"] == "drift", "feature"])
    assert {r["feature"] for r in rep["ablation"]} == expected
    for record in rep["ablation"]:
        assert {"prediction_psi_after", "auc_after", "calibration_after", "psi_recovery",
                "auc_recovery"} <= set(record)


def test_ablation_auc_recovery_guarded_when_denominator_is_zero(trained):
    from credit_monitor.drift.causal import ablation
    from credit_monitor.model.features import model_matrix
    from credit_monitor.model.predict import load_model, score

    pipeline, meta = load_model(trained)
    ref = pd.read_parquet(trained.data_dir / "reference" / "test_scored.parquet")
    X = model_matrix(ref)
    y = ref["TARGET"].to_numpy()
    s = score(pipeline, ref)
    # lote == referencia: nada a recuperar (psi_base ~ 0, auc_ref == auc_base)
    from sklearn.metrics import roc_auc_score

    table = ablation(pipeline, X, X, y, s, meta["cutoff"], ["EXT_SOURCE_2"],
                     reference_auc=float(roc_auc_score(y, s)))
    row = table.iloc[0]
    assert pd.isna(row["auc_recovery"])


def test_shap_importance_one_value_per_feature(trained):
    from credit_monitor.drift.causal import shap_importance
    from credit_monitor.model.features import model_matrix
    from credit_monitor.model.predict import load_model

    pipeline, _ = load_model(trained)
    ref = pd.read_parquet(trained.data_dir / "reference" / "test_scored.parquet")
    importance = shap_importance(pipeline, model_matrix(ref))
    assert list(importance.index) == MODEL_FEATURES
    assert (importance >= 0).all()
    assert importance.sum() > 0
    # as categoricas one-hot foram agregadas: nenhuma coluna `FEATURE_categoria` sobrou
    assert not any(name not in MODEL_FEATURES for name in importance.index)
    assert set(CATEGORICAL_FEATURES) <= set(importance.index)


def test_cli_causal_exit_codes(trained, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(trained.data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(trained.reports_dir))
    monkeypatch.setenv("MODELS_DIR", str(trained.models_dir))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", trained.pseudonymization_salt)
    for var in ("MLFLOW_TRACKING_URI", "PUSHGATEWAY_URL", "LOKI_URL"):
        monkeypatch.delenv(var, raising=False)
    assert main(["causal", "--batch", "99"]) == 2
    assert "99" in capsys.readouterr().out
    assert main(["causal"]) == 0
    out = capsys.readouterr().out
    assert "concept drift" in out and "recupera" in out
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "vazio"))
    assert main(["causal"]) == 2
    assert "train" in capsys.readouterr().out


