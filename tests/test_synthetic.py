import numpy as np
import pandas as pd
import pytest

from credit_monitor import config
from credit_monitor.data.synthetic import make_synthetic_application


def test_schema_and_anomalies():
    df = make_synthetic_application(5_000, seed=1)
    assert list(df.columns) == config.RAW_COLUMNS
    assert (df["AMT_INCOME_TOTAL"] == 117_000_000).sum() == 1
    assert (df["CODE_GENDER"] == "XNA").sum() == 4
    assert df["AMT_ANNUITY"].isna().sum() == 12
    assert df["SK_ID_CURR"].min() >= 900_000_001


def test_target_rate_and_determinism():
    a = make_synthetic_application(20_000, seed=7)
    b = make_synthetic_application(20_000, seed=7)
    assert 0.06 <= a["TARGET"].mean() <= 0.10
    assert a.equals(b)


def test_model_features_count():
    assert len(config.MODEL_FEATURES) == 22
    assert "CODE_GENDER" not in config.MODEL_FEATURES


def test_config_constants():
    assert len(config.RAW_COLUMNS) == 21
    assert len(config.NUMERIC_FEATURES) == 15
    assert len(config.CATEGORICAL_FEATURES) == 7
    assert config.MODEL_FEATURES == config.NUMERIC_FEATURES + config.CATEGORICAL_FEATURES
    assert set(config.RAW_DTYPES) == set(config.RAW_COLUMNS)
    assert config.SEED == 42
    assert config.DAYS_EMPLOYED_SENTINEL == 365243
    assert config.SIMULATION_ORDER == ["01", "02", "03-dirty", "03", "04", "05", "06"]
    assert (config.PSI_WARN, config.PSI_DRIFT, config.KS_DRIFT) == (0.10, 0.20, 0.10)
    assert (config.DRIFT_SHARE_ALERT, config.PRED_PSI_ALERT) == (0.30, 0.20)
    assert (config.AUC_DROP_ALERT, config.CALIBRATION_ALERT) == (0.05, 1.3)
    assert config.DISPARATE_IMPACT_MIN == 0.8


def test_dtypes_match_config(synthetic_raw):
    for col, dtype in config.RAW_DTYPES.items():
        assert str(synthetic_raw[col].dtype) == dtype, col
    assert synthetic_raw["SK_ID_CURR"].dtype.kind == "i"
    assert synthetic_raw["TARGET"].dtype.kind == "i"
    for col in config.CATEGORICAL_FEATURES:
        assert synthetic_raw[col].dtype == "category"


def test_anomalies_in_17_distinct_rows(synthetic_raw):
    df = synthetic_raw
    rows = (
        set(df.index[df["AMT_INCOME_TOTAL"] == 117_000_000])
        | set(df.index[df["CODE_GENDER"] == "XNA"])
        | set(df.index[df["AMT_ANNUITY"].isna()])
    )
    assert len(rows) == 17


def test_without_anomalies_respects_contract_limits():
    df = make_synthetic_application(30_000, seed=3, with_anomalies=False)
    assert df["SK_ID_CURR"].is_unique
    assert (df["AMT_INCOME_TOTAL"] > 0).all()
    assert (df["AMT_INCOME_TOTAL"] <= config.INCOME_MAX).all()
    assert (df["AMT_CREDIT"] > 0).all()
    assert (df["AMT_ANNUITY"] > 0).all()
    assert set(df["CODE_GENDER"].unique()) == {"F", "M"}
    age = -df["DAYS_BIRTH"] / 365.25
    assert (df["DAYS_BIRTH"] < 0).all()
    assert age.between(config.AGE_MIN, config.AGE_MAX).all()
    emp = df["DAYS_EMPLOYED"]
    assert ((emp <= 0) | (emp == config.DAYS_EMPLOYED_SENTINEL)).all()
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        assert df[col].dropna().between(0, 1).all()


def test_with_anomalies_clean_rows_respect_limits(synthetic_raw):
    df = synthetic_raw
    clean_income = df["AMT_INCOME_TOTAL"][df["AMT_INCOME_TOTAL"] != 117_000_000]
    assert (clean_income <= config.INCOME_MAX).all()
    assert df["SK_ID_CURR"].is_unique
    assert df["SK_ID_CURR"].max() == 900_000_000 + len(df)


def test_realistic_marginals(synthetic_raw):
    df = synthetic_raw
    sentinel = df["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
    assert 0.16 <= sentinel.mean() <= 0.20
    assert 0.50 <= df["EXT_SOURCE_1"].isna().mean() <= 0.60
    assert df["EXT_SOURCE_2"].isna().mean() < 0.01
    # sentinela <=> pensionista ou desempregado
    income_type = df["NAME_INCOME_TYPE"].astype(str)
    assert set(income_type[sentinel].unique()) <= {"Pensioner", "Unemployed"}
    assert not income_type[~sentinel].isin(["Pensioner", "Unemployed"]).any()


def test_categorical_domains(synthetic_raw):
    df = synthetic_raw
    assert set(df["NAME_CONTRACT_TYPE"].unique()) == {"Cash loans", "Revolving loans"}
    assert {"Working", "Commercial associate", "Pensioner", "State servant", "Unemployed"} <= set(
        df["NAME_INCOME_TYPE"].unique()
    )
    assert "Higher education" in set(df["NAME_EDUCATION_TYPE"].unique())
    assert "Married" in set(df["NAME_FAMILY_STATUS"].unique())
    assert "House / apartment" in set(df["NAME_HOUSING_TYPE"].unique())
    assert set(df["FLAG_OWN_CAR"].unique()) == {"Y", "N"}
    assert set(df["FLAG_OWN_REALTY"].unique()) == {"Y", "N"}
    assert set(df["CODE_GENDER"].unique()) == {"F", "M", "XNA"}


def test_target_depends_on_risk_drivers(synthetic_raw):
    df = synthetic_raw
    bad = df["TARGET"] == 1
    for col in ("EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"):
        assert df.loc[bad, col].mean() < df.loc[~bad, col].mean()
    age = -df["DAYS_BIRTH"] / 365.25
    assert age[bad].mean() < age[~bad].mean()
    ratio = df["AMT_ANNUITY"] / df["AMT_INCOME_TOTAL"]
    assert ratio[bad].mean() > ratio[~bad].mean()


def test_target_is_learnable(synthetic_raw):
    """A base precisa ter sinal: um ranking simples por EXT_SOURCE separa os grupos."""
    from sklearn.metrics import roc_auc_score

    df = synthetic_raw
    score = -df[["EXT_SOURCE_2", "EXT_SOURCE_3"]].fillna(0.5).mean(axis=1)
    assert roc_auc_score(df["TARGET"], score) > 0.60


def test_small_n_rejected():
    with pytest.raises(ValueError):
        make_synthetic_application(10)
    assert isinstance(make_synthetic_application(17), pd.DataFrame)
    assert np.isfinite(make_synthetic_application(200)["TARGET"].mean())
