"""Engenharia de features (spec 4.2). Funcoes puras: nunca alteram o frame de entrada."""

from __future__ import annotations

import pandas as pd

from credit_monitor.config import DAYS_EMPLOYED_SENTINEL, MODEL_FEATURES

DAYS_PER_YEAR = 365.25


def _ratio(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    """Divisao em float64; denominador zero ou nulo vira NaN (nunca inf)."""
    return numerator / denominator.where(denominator != 0)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Copia de `df` com as features derivadas; as demais colunas seguem inalteradas."""
    out = df.copy()
    birth = df["DAYS_BIRTH"].astype("float64")
    employed = df["DAYS_EMPLOYED"].astype("float64")
    income = df["AMT_INCOME_TOTAL"].astype("float64")
    credit = df["AMT_CREDIT"].astype("float64")
    annuity = df["AMT_ANNUITY"].astype("float64")

    unemployed = employed == DAYS_EMPLOYED_SENTINEL
    out["AGE_YEARS"] = -birth / DAYS_PER_YEAR
    out["YEARS_EMPLOYED"] = (-employed / DAYS_PER_YEAR).where(~unemployed)
    out["IS_UNEMPLOYED_OR_PENSIONER"] = unemployed.astype("int64")
    out["CREDIT_INCOME_RATIO"] = _ratio(credit, income)
    out["ANNUITY_INCOME_RATIO"] = _ratio(annuity, income)
    out["CREDIT_TERM"] = _ratio(credit, annuity)
    return out


def model_matrix(df: pd.DataFrame) -> pd.DataFrame:
    """Colunas que o modelo consome, na ordem de `MODEL_FEATURES`."""
    return build_features(df)[MODEL_FEATURES]
