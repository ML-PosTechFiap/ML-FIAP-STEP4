"""Schema Pandera do contrato de dados (spec 5.1), aplicado ao frame pseudonimizado.

O `name` de cada `Check` e o label `rule` das metricas e dos relatorios: e estavel.
Not-null e unicidade sao checks nomeados (e nao `nullable`/`unique` do Pandera), para
que cada regra apareca com o nome combinado. Faixas e positividade ignoram nulos: um
nulo e contado so pela regra de not-null da coluna.
"""

from __future__ import annotations

from collections.abc import Callable

import pandas as pd
import pandera as pa

from credit_monitor.config import (
    AGE_MAX,
    AGE_MIN,
    CATEGORICAL_FEATURES,
    DAYS_EMPLOYED_SENTINEL,
    INCOME_MAX,
    MIN_BATCH_ROWS,
    RAW_COLUMNS,
)
from credit_monitor.privacy.pseudonymize import ID_COLUMN, KEY_COLUMN

CONTRACT_VERSION = "1.0"

GENDER_DOMAIN = ["F", "M"]
_NUMERIC_COLUMNS = [
    c for c in RAW_COLUMNS if c not in {ID_COLUMN, "TARGET", "CODE_GENDER", *CATEGORICAL_FEATURES}
]
_EXT_SOURCES = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]


def _check(fn: Callable[[pd.Series], pd.Series], name: str, ignore_na: bool = True) -> pa.Check:
    return pa.Check(fn, name=name, ignore_na=ignore_na)


def _not_null(name: str) -> pa.Check:
    return _check(lambda s: s.notna(), name, ignore_na=False)


def _domain(column: str, domain: list[str]) -> pa.Check:
    allowed = set(domain)
    # str(): funciona igual para dtype category e object, e pega valores fora do dominio.
    return _check(lambda s: s.astype(str).isin(allowed), f"domain_{column}")


def domains_from_reference(df: pd.DataFrame) -> dict[str, list[str]]:
    """Categorias observadas na referencia limpa; `CODE_GENDER` e fixado em F/M."""
    domains = {
        col: sorted(df[col].dropna().astype(str).unique().tolist()) for col in CATEGORICAL_FEATURES
    }
    domains["CODE_GENDER"] = list(GENDER_DOMAIN)
    return domains


def build_schema(categorical_domains: dict[str, list[str]]) -> pa.DataFrameSchema:
    """Schema do contrato. `TARGET` e opcional; colunas extras sao toleradas."""
    domains = {**categorical_domains, "CODE_GENDER": list(GENDER_DOMAIN)}

    def num(*checks: pa.Check) -> pa.Column:
        return pa.Column("float64", list(checks), nullable=True, coerce=True)

    columns: dict[str, pa.Column] = {
        KEY_COLUMN: pa.Column(
            None,
            [
                _check(lambda s: ~s.duplicated(keep=False), "client_key_unique"),
                _not_null("client_key_not_null"),
            ],
            nullable=True,
        )
    }
    for col in _NUMERIC_COLUMNS:
        columns[col] = num()
    columns["AMT_INCOME_TOTAL"] = num(
        _not_null("income_not_null"),
        _check(lambda s: (s > 0) & (s <= INCOME_MAX), "income_range"),
    )
    columns["DAYS_BIRTH"] = num(
        _check(lambda s: s.between(-AGE_MAX * 365.25, -AGE_MIN * 365.25), "age_range")
    )
    columns["DAYS_EMPLOYED"] = num(
        _check(lambda s: (s <= 0) | (s == DAYS_EMPLOYED_SENTINEL), "days_employed_valid")
    )
    columns["AMT_CREDIT"] = num(_check(lambda s: s > 0, "credit_positive"))
    columns["AMT_ANNUITY"] = num(
        _not_null("annuity_not_null"), _check(lambda s: s > 0, "annuity_positive")
    )
    for col in _EXT_SOURCES:
        columns[col] = num(_check(lambda s: s.between(0, 1), "ext_source_range"))
    for col in [*CATEGORICAL_FEATURES, "CODE_GENDER"]:
        columns[col] = pa.Column(None, [_domain(col, domains[col])], nullable=True)
    columns["TARGET"] = pa.Column(
        "float64",
        [_check(lambda s: s.isin([0, 1]), "target_binary", ignore_na=False)],
        nullable=True,
        coerce=True,
        required=False,
    )

    return pa.DataFrameSchema(
        columns,
        checks=[pa.Check(lambda df: len(df) >= MIN_BATCH_ROWS, name="min_rows")],
        coerce=True,
    )
