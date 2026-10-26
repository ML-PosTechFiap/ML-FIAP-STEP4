"""Auditoria de vies (spec 8.3): taxas de aprovacao por grupo e teste de proxy."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.feature_selection import mutual_info_classif

from credit_monitor.config import AGE_BANDS, CATEGORICAL_FEATURES, SEED

DAYS_PER_YEAR = 365.25


def _age_band(df: pd.DataFrame) -> pd.Series:
    age = -df["DAYS_BIRTH"].astype("float64").to_numpy() / DAYS_PER_YEAR
    labels = np.full(len(df), None, dtype=object)
    for low, high, label in AGE_BANDS:
        labels[(age >= low) & (age < high)] = label
    return pd.Series(labels, dtype=object)


def _group_report(
    groups: pd.Series, approved: np.ndarray, y_true: np.ndarray | None, order: list[str]
) -> dict:
    """`groups` (posicional, None = sem grupo) contra `approved` (1 = aprovado)."""
    approval_rate: dict[str, float] = {}
    tpr: dict[str, float] = {}
    for name in order:
        mask = (groups == name).to_numpy()
        if not mask.any():
            continue
        approval_rate[name] = float(approved[mask].mean())
        if y_true is not None:
            good = mask & (y_true == 0)
            if good.any():
                tpr[name] = float(approved[good].mean())

    top = max(approval_rate.values(), default=0.0)
    impact = min(approval_rate.values()) / top if top > 0 else 1.0
    eo_difference = None
    if y_true is not None:
        eo_difference = float(max(tpr.values()) - min(tpr.values())) if tpr else 0.0
    return {
        "approval_rate": approval_rate,
        "disparate_impact": float(impact),
        "tpr": tpr if y_true is not None else None,
        "eo_difference": eo_difference,
    }


def fairness_report(df: pd.DataFrame, approved: np.ndarray, y_true: np.ndarray | None) -> dict:
    """Taxa de aprovacao, impacto disparato, TPR e diferenca de equal opportunity.

    Grupos: `CODE_GENDER` e faixa etaria. TPR = aprovacao entre `TARGET == 0`.
    `df` precisa de `CODE_GENDER` e `DAYS_BIRTH`; `approved` e `y_true` alinham por posicao.
    """
    approved = np.asarray(approved).astype(bool)
    y = None if y_true is None else np.asarray(y_true)
    gender = pd.Series(
        np.where(df["CODE_GENDER"].notna(), df["CODE_GENDER"].astype(str), None), dtype=object
    )
    genders = sorted(g for g in gender.dropna().unique())
    return {
        "gender": _group_report(gender, approved, y, genders),
        "age_band": _group_report(
            _age_band(df), approved, y, [label for _, _, label in AGE_BANDS]
        ),
    }


def proxy_scores(X: pd.DataFrame, gender: pd.Series) -> pd.Series:
    """Informacao mutua de cada feature com o genero, em ordem decrescente."""
    encoded = X.copy()
    for col in encoded.columns:
        if col in CATEGORICAL_FEATURES:
            encoded[col] = encoded[col].astype("category").cat.codes
        else:
            encoded[col] = encoded[col].astype("float64").fillna(encoded[col].median())
    encoded = encoded.fillna(0.0).astype("float64")
    discrete = [col in CATEGORICAL_FEATURES for col in encoded.columns]
    mi = mutual_info_classif(
        encoded.to_numpy(),
        gender.astype(str).to_numpy(),
        discrete_features=discrete,
        random_state=SEED,
    )
    return pd.Series(mi, index=encoded.columns).sort_values(ascending=False)
