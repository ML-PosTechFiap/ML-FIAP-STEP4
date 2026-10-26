"""Estatisticas de drift: PSI, KS e a tabela por feature (spec 6.2 e 6.3)."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from credit_monitor.config import KS_DRIFT, PSI_DRIFT, PSI_WARN
from credit_monitor.model.features import model_matrix

MISSING = "__missing__"


def _is_categorical(series: pd.Series) -> bool:
    return (
        isinstance(series.dtype, pd.CategoricalDtype)
        or pd.api.types.is_object_dtype(series.dtype)
        or pd.api.types.is_string_dtype(series.dtype)
    )


def _numeric_counts(reference: pd.Series, current: pd.Series, bins: int) -> tuple:
    """Contagens por bin de quantil da referencia; o ultimo bin e o de NaN."""
    ref = reference.astype("float64").to_numpy()
    cur = current.astype("float64").to_numpy()
    ref_valid = ref[~np.isnan(ref)]
    if len(ref_valid):
        # Cortes internos (quantis), sem repeticao; bordas externas implicitas em -inf/inf.
        cuts = np.unique(np.quantile(ref_valid, np.linspace(0, 1, bins + 1)[1:-1]))
    else:
        cuts = np.array([])

    def count(values: np.ndarray) -> np.ndarray:
        null = np.isnan(values)
        idx = np.searchsorted(cuts, values[~null], side="left")
        counts = np.bincount(idx, minlength=len(cuts) + 1).astype("float64")
        return np.append(counts, null.sum())

    return count(ref), count(cur)


def _categorical_counts(reference: pd.Series, current: pd.Series) -> tuple:
    """Uma categoria por bin (inclui `__missing__` e categorias so do lote atual)."""

    def labels(series: pd.Series) -> pd.Series:
        return series.astype(object).where(series.notna(), MISSING).astype(str)

    ref, cur = labels(reference).value_counts(), labels(current).value_counts()
    categories = sorted(set(ref.index) | set(cur.index))
    return (
        ref.reindex(categories, fill_value=0).to_numpy(dtype="float64"),
        cur.reindex(categories, fill_value=0).to_numpy(dtype="float64"),
    )


def psi(reference: pd.Series, current: pd.Series, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index. Sempre finito (eps nas proporcoes evita log(0))."""
    if _is_categorical(reference) or _is_categorical(current):
        ref_counts, cur_counts = _categorical_counts(reference, current)
    else:
        ref_counts, cur_counts = _numeric_counts(reference, current, bins)
    ref_p = ref_counts / max(ref_counts.sum(), 1.0) + eps
    cur_p = cur_counts / max(cur_counts.sum(), 1.0) + eps
    return float(np.sum((cur_p - ref_p) * np.log(cur_p / ref_p)))


def ks(reference: pd.Series, current: pd.Series) -> float:
    """Estatistica D de `ks_2samp` sobre os valores nao nulos; 0.0 se um lado ficar vazio."""
    ref = reference.astype("float64").dropna().to_numpy()
    cur = current.astype("float64").dropna().to_numpy()
    if len(ref) == 0 or len(cur) == 0:
        return 0.0
    return float(ks_2samp(ref, cur).statistic)


def _status(psi_value: float, ks_value: float | None) -> str:
    if psi_value >= PSI_DRIFT or (ks_value is not None and ks_value > KS_DRIFT):
        return "drift"
    if psi_value >= PSI_WARN:
        return "atencao"
    return "ok"


def feature_drift_table(ref: pd.DataFrame, cur: pd.DataFrame) -> pd.DataFrame:
    """PSI, KS (so numericas) e status por feature, sobre `model_matrix` dos dois lados."""
    ref_m, cur_m = model_matrix(ref), model_matrix(cur)
    rows = []
    for feature in ref_m.columns:
        categorical = _is_categorical(ref_m[feature])
        psi_value = psi(ref_m[feature], cur_m[feature])
        ks_value = None if categorical else ks(ref_m[feature], cur_m[feature])
        rows.append(
            {
                "feature": feature,
                "kind": "categorical" if categorical else "numeric",
                "psi": psi_value,
                "ks": ks_value,
                "status": _status(psi_value, ks_value),
            }
        )
    return pd.DataFrame(rows, columns=["feature", "kind", "psi", "ks", "status"])
