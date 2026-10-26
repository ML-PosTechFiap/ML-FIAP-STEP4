import json

import numpy as np
import pandas as pd
import pytest

from credit_monitor.drift.detect import analyze_batch, performance, recommend
from credit_monitor.drift.stats import feature_drift_table, ks, psi

rng = np.random.default_rng(0)


def test_psi_identical_near_zero_and_shift_large():
    x = pd.Series(rng.normal(size=20_000))
    assert psi(x, x.sample(frac=1, random_state=1)) < 0.01
    assert psi(x, x * 1.3 + 0.5) > 0.2


@pytest.mark.parametrize(
    "ref,cur",
    [
        (pd.Series([1.0] * 1000), pd.Series(rng.normal(size=1000))),
        (pd.Series(rng.normal(size=1000)), pd.Series([np.nan] * 1000)),
        (pd.Series(["a", "b"] * 500), pd.Series(["c"] * 1000)),
    ],
)
def test_psi_degenerate_is_finite(ref, cur):
    assert np.isfinite(psi(ref, cur))


def test_psi_detects_null_rate_change():
    ref = pd.Series(np.where(rng.random(20_000) < 0.5, np.nan, rng.random(20_000)))
    cur = pd.Series(np.where(rng.random(20_000) < 0.8, np.nan, rng.random(20_000)))
    assert psi(ref, cur) > 0.2


def test_psi_categorical_counts_missing_and_new_categories():
    ref = pd.Series(["a"] * 500 + ["b"] * 500, dtype="category")
    same = pd.Series(["a", "b"] * 500, dtype="category")
    assert psi(ref, same) < 0.01
    assert psi(ref, pd.Series(["a"] * 500 + [None] * 500)) > 0.2


def test_ks_empty_side_is_zero():
    assert ks(pd.Series([np.nan] * 10), pd.Series([1.0, 2.0])) == 0.0


def test_performance_and_calibration():
    y = np.array([0] * 90 + [1] * 10)
    s = np.full(100, 0.05)
    p = performance(y, s, cutoff=0.5)
    assert p["calibration_ratio"] == pytest.approx(2.0) and p["approval_rate"] == 1.0


def test_performance_single_class_has_no_auc_or_ks():
    p = performance(np.zeros(50, dtype=int), np.linspace(0.01, 0.5, 50), cutoff=0.2)
    assert p["auc"] is None and p["ks"] is None and p["default_rate_observed"] == 0.0


def test_recommend_rules():
    base = {
        "dataset_drift_share": 0.0,
        "prediction_psi": 0.0,
        "fairness": {"gender": 1.0},
        "performance": {"auc": 0.75, "calibration_ratio": 1.0},
    }
    assert recommend(base, 0.75) == "OK"
    assert recommend({**base, "dataset_drift_share": 0.4}, 0.75) == "INVESTIGAR"
    assert recommend({**base, "fairness": {"gender": 0.7}}, 0.75) == "INVESTIGAR"
    assert (
        recommend({**base, "performance": {"auc": 0.69, "calibration_ratio": 1.0}}, 0.75)
        == "RETREINAR"
    )
    assert (
        recommend({**base, "performance": {"auc": 0.75, "calibration_ratio": 1.4}}, 0.75)
        == "RETREINAR"
    )
    assert recommend({**base, "performance": None, "prediction_psi": 0.3}, 0.75) == "INVESTIGAR"


def _scored_batch(settings, batch_id):
    """Lote pseudonimizado e pontuado, como o `monitor` o entregara a `analyze_batch`."""
    from credit_monitor.model.predict import load_model, score
    from credit_monitor.privacy.pseudonymize import pseudonymize

    pipeline, meta = load_model(settings)
    batch_dir = settings.data_dir / "batches" / batch_id
    cur = pseudonymize(pd.read_parquet(batch_dir / "batch.parquet"), settings.pseudonymization_salt)
    cur["score"] = score(pipeline, cur)
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    return cur, meta, manifest


def test_analyze_batch_synthetic(trained, tmp_path):
    ref = pd.read_parquet(trained.data_dir / "reference" / "test_scored.parquet")

    cur01, meta, manifest01 = _scored_batch(trained, "01")
    s01 = analyze_batch(ref, cur01, "01", meta, tmp_path)
    assert s01["drifted_features"] == [] and manifest01["drifted_features"] == []
    assert s01["performance"] is not None and s01["recommendation"] == "OK"

    cur06, meta, manifest06 = _scored_batch(trained, "06")
    s06 = analyze_batch(ref, cur06, "06", meta, tmp_path)
    assert set(manifest06["drifted_features"]) <= set(s06["drifted_features"])
    assert s06["recommendation"] in {"INVESTIGAR", "RETREINAR"}
    assert (tmp_path / "06_data_drift.html").exists()
    assert (tmp_path / "06_performance.html").exists()
    assert (tmp_path / "06_data_drift.json").exists()

    table = feature_drift_table(ref, cur06).set_index("feature")
    own, evidently = table.loc["AMT_INCOME_TOTAL", "psi"], s06["psi"]["AMT_INCOME_TOTAL"]
    assert abs(own - evidently) < 0.05
    assert "score" in s06["psi"]

    saved = json.loads((tmp_path / "06_summary.json").read_text(encoding="utf-8"))
    assert saved["batch_id"] == "06" and saved["recommendation"] == s06["recommendation"]
    text = json.dumps(saved)
    assert "SK_ID_CURR" not in text and "client_key" not in text


def test_analyze_batch_without_target_skips_labeled_metrics(trained, tmp_path):
    ref = pd.read_parquet(trained.data_dir / "reference" / "test_scored.parquet")
    cur, meta, _ = _scored_batch(trained, "06")
    summary = analyze_batch(ref, cur.drop(columns=["TARGET"]), "06", meta, tmp_path)
    assert summary["performance"] is None and summary["labeled"] is False
    assert not (tmp_path / "06_performance.html").exists()
    assert summary["recommendation"] in {"OK", "INVESTIGAR"}


@pytest.mark.parametrize("batch_id", ["01", "02", "03", "04", "05", "06"])
def test_manifest_features_are_detected(trained, batch_id):
    """Revocacao 1 do mapeamento `DRIFT_EFFECTS` contra a regra de drift, em todos os lotes."""
    ref = pd.read_parquet(trained.data_dir / "reference" / "test_scored.parquet")
    batch_dir = trained.data_dir / "batches" / batch_id
    manifest = json.loads((batch_dir / "manifest.json").read_text(encoding="utf-8"))
    table = feature_drift_table(ref, pd.read_parquet(batch_dir / "batch.parquet"))
    detected = set(table.loc[table["status"] == "drift", "feature"])
    assert set(manifest["drifted_features"]) <= detected
