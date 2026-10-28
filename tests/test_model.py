import json
import shutil

import numpy as np
import pandas as pd
import pytest

from credit_monitor import config
from credit_monitor.model.fairness import fairness_report, proxy_scores
from credit_monitor.model.features import build_features
from credit_monitor.model.predict import ModelNotTrained, load_model
from credit_monitor.model.train import choose_cutoff, ks_statistic


def test_features_handle_sentinel():
    df = pd.DataFrame({"DAYS_BIRTH": [-365.25 * 30], "DAYS_EMPLOYED": [365243],
                       "AMT_INCOME_TOTAL": [100.0], "AMT_CREDIT": [500.0], "AMT_ANNUITY": [50.0]})
    f = build_features(df)
    assert f.AGE_YEARS.iloc[0] == pytest.approx(30)
    assert np.isnan(f.YEARS_EMPLOYED.iloc[0]) and f.IS_UNEMPLOYED_OR_PENSIONER.iloc[0] == 1
    assert f.CREDIT_TERM.iloc[0] == 10 and f.ANNUITY_INCOME_RATIO.iloc[0] == 0.5


def test_features_do_not_mutate_and_survive_zero_and_nan():
    df = pd.DataFrame({"DAYS_BIRTH": [-365.25 * 40, -365.25 * 50],
                       "DAYS_EMPLOYED": [-730.5, -365.25],
                       "AMT_INCOME_TOTAL": [0.0, 100.0], "AMT_CREDIT": [500.0, 200.0],
                       "AMT_ANNUITY": [50.0, np.nan]})
    before = df.copy()
    f = build_features(df)
    pd.testing.assert_frame_equal(df, before)
    assert f.YEARS_EMPLOYED.tolist() == pytest.approx([2.0, 1.0])
    assert f.IS_UNEMPLOYED_OR_PENSIONER.tolist() == [0, 0]
    assert np.isnan(f.CREDIT_INCOME_RATIO.iloc[0]) and not np.isinf(f.CREDIT_INCOME_RATIO).any()
    assert np.isnan(f.ANNUITY_INCOME_RATIO.iloc[0]) and np.isnan(f.CREDIT_TERM.iloc[1])


def test_ks_perfect_and_random():
    assert ks_statistic(np.array([0, 0, 1, 1]), np.array([.1, .2, .8, .9])) == 1.0
    rng = np.random.default_rng(0)
    assert ks_statistic(rng.integers(0, 2, 5000), rng.random(5000)) < 0.05


def test_choose_cutoff_separates_classes():
    y, s = np.array([0, 0, 0, 1, 1]), np.array([.1, .2, .3, .6, .9])
    cutoff = choose_cutoff(y, s)
    assert 0.3 < cutoff <= 0.6  # aprova (score < corte) todos os bons e nenhum mau
    assert ((s < cutoff) == (y == 0)).all()


def test_disparate_impact():
    df = pd.DataFrame({"CODE_GENDER": ["F"] * 4 + ["M"] * 4, "DAYS_BIRTH": [-365.25 * 40] * 8})
    r = fairness_report(df, np.array([1, 1, 0, 0, 1, 1, 1, 1]), None)
    assert r["gender"]["disparate_impact"] == pytest.approx(0.5)
    assert r["gender"]["tpr"] is None and r["gender"]["eo_difference"] is None


def test_fairness_report_tpr_bands_and_json():
    df = pd.DataFrame({"CODE_GENDER": ["F"] * 4 + ["M"] * 4,
                       "DAYS_BIRTH": [-365.25 * a for a in (25, 25, 50, 50, 25, 25, 50, 50)]})
    approved = np.array([1, 0, 1, 1, 1, 1, 0, 0])
    y = np.array([0, 1, 0, 0, 0, 1, 0, 0])
    r = fairness_report(df, approved, y)
    assert r["gender"]["approval_rate"] == {"F": 0.75, "M": 0.5}
    assert r["gender"]["tpr"] == {"F": pytest.approx(1.0), "M": pytest.approx(1 / 3)}
    assert r["gender"]["eo_difference"] == pytest.approx(2 / 3)
    assert set(r["age_band"]["approval_rate"]) == {"<30", "45-60"}  # faixas vazias ficam de fora
    assert r["age_band"]["disparate_impact"] == pytest.approx(0.5 / 0.75)
    assert json.loads(json.dumps(r)) == r


def test_disparate_impact_when_nobody_is_approved():
    df = pd.DataFrame({"CODE_GENDER": ["F", "M"], "DAYS_BIRTH": [-365.25 * 40] * 2})
    assert fairness_report(df, np.array([0, 0]), None)["gender"]["disparate_impact"] == 1.0


def test_proxy_scores_ranks_leaky_feature_first():
    rng = np.random.default_rng(1)
    gender = pd.Series(rng.choice(["F", "M"], 2000))
    X = pd.DataFrame({"AMT_CREDIT": rng.normal(size=2000),
                      "CNT_CHILDREN": (gender == "M").astype(float) + rng.normal(0, .1, 2000),
                      "FLAG_OWN_CAR": pd.Categorical(rng.choice(["Y", "N"], 2000))})
    scores = proxy_scores(X, gender)
    assert scores.index[0] == "CNT_CHILDREN" and scores.is_monotonic_decreasing


def test_run_train_synthetic(tmp_path):
    from credit_monitor.data.prepare import run_prepare
    from credit_monitor.model.predict import score
    from credit_monitor.model.train import run_train
    from tests.test_observability import _settings
    s = _settings(tmp_path)
    run_prepare(s, synthetic=True, n=30_000)
    meta = run_train(s)
    assert meta["reference_metrics"]["auc"] > 0.65
    assert "CODE_GENDER" not in meta["features"] and 0 < meta["cutoff"] < 1
    assert (tmp_path / "reference/test_scored.parquet").exists()
    assert set(meta["fairness"]) == {"gender", "age_band", "proxy_top5"}
    assert meta["features"] == config.MODEL_FEATURES and meta["mlflow_model_version"] is None
    assert meta["contract_version"] == "1.0" and set(meta["reference_metrics"]) == {
        "auc", "pr_auc", "ks", "brier"}
    assert len(meta["fairness"]["proxy_top5"]) == 5
    assert json.loads((tmp_path / "metadata.json").read_text(encoding="utf-8")) == meta
    pipeline, loaded = load_model(s)
    test = pd.read_parquet(tmp_path / "reference/test_scored.parquet")
    assert "score" in test.columns and loaded == meta
    np.testing.assert_allclose(score(pipeline, test), test["score"], rtol=1e-6)
    # a chave pseudonimizada (nunca o ID bruto) e a unica identificacao na saida
    assert "SK_ID_CURR" not in test.columns and "client_key" in test.columns


def test_load_model_without_training(tmp_path):
    from tests.test_observability import _settings
    with pytest.raises(ModelNotTrained, match="train"):
        load_model(_settings(tmp_path))


def _reference_copy(prepared, tmp_path):
    from tests.test_observability import _settings
    shutil.copytree(prepared.data_dir / "reference", tmp_path / "data/reference")
    return _settings(tmp_path / "data", reports_dir=tmp_path / "reports",
                     models_dir=tmp_path / "models")


def _break_contract(settings):
    path = settings.data_dir / "reference/train.parquet"
    df = pd.read_parquet(path)
    df.loc[0, "AMT_INCOME_TOTAL"] = 0.0  # viola income_range
    df.to_parquet(path, index=False)


def test_run_train_rejects_contract_violation(prepared, tmp_path):
    from credit_monitor.model.train import run_train
    s = _reference_copy(prepared, tmp_path)
    _break_contract(s)
    with pytest.raises(RuntimeError, match="contrato"):
        run_train(s)
    assert not (s.models_dir / "model.joblib").exists()


def _cli_env(monkeypatch, s):
    for name in ("MLFLOW_TRACKING_URI", "LOKI_URL", "PUSHGATEWAY_URL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("DATA_DIR", str(s.data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(s.reports_dir))
    monkeypatch.setenv("MODELS_DIR", str(s.models_dir))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", "t")


def test_cli_train_exit_codes(prepared, tmp_path, monkeypatch, capsys):
    from credit_monitor.cli import main
    s = _reference_copy(prepared, tmp_path)
    _cli_env(monkeypatch, s)
    _break_contract(s)
    assert main(["train"]) == 2
    assert "contrato" in capsys.readouterr().out
    shutil.copy(prepared.data_dir / "reference/train.parquet",
                s.data_dir / "reference/train.parquet")
    assert main(["train"]) == 0
    out = capsys.readouterr().out
    assert "AUC" in out and "KS" in out and "corte" in out
    assert (s.models_dir / "model.joblib").exists()
