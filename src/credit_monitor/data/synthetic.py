"""Gerador de base sintetica no schema do application_train do Home Credit.

Serve de substituto realista quando o dataset real nao esta disponivel: o sistema
inteiro roda (e o CI verifica) com a mesma estrutura, dominios e problemas de
qualidade conhecidos do dado real.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from credit_monitor import config

ID_OFFSET = 900_000_000  # SK_ID_CURR sintetico = ID_OFFSET + 1 + i
TARGET_RATE = 0.08
N_ANOMALY_ROWS = 17  # 1 renda absurda + 4 genero XNA + 12 parcelas nulas
ABSURD_INCOME = 117_000_000

# Dominios (rotulo -> probabilidade) espelhando o Home Credit.
_INCOME_TYPES = {
    "Working": 0.500,
    "Commercial associate": 0.235,
    "Pensioner": 0.175,
    "State servant": 0.080,
    "Unemployed": 0.005,
    "Student": 0.002,
    "Businessman": 0.002,
    "Maternity leave": 0.001,
}
_EDUCATION_TYPES = {
    "Secondary / secondary special": 0.710,
    "Higher education": 0.244,
    "Incomplete higher": 0.033,
    "Lower secondary": 0.0125,
    "Academic degree": 0.0005,
}
_FAMILY_STATUS = {
    "Married": 0.640,
    "Single / not married": 0.148,
    "Civil marriage": 0.097,
    "Separated": 0.065,
    "Widow": 0.050,
}
_HOUSING_TYPES = {
    "House / apartment": 0.886,
    "With parents": 0.048,
    "Municipal apartment": 0.036,
    "Rented apartment": 0.016,
    "Office apartment": 0.0085,
    "Co-op apartment": 0.0055,
}
_LOAN_TERMS = np.array([12, 18, 24, 36, 48, 60])
_LOAN_TERM_PROBS = np.array([0.10, 0.10, 0.20, 0.30, 0.15, 0.15])


def _draw(rng: np.random.Generator, domain: dict[str, float], n: int) -> np.ndarray:
    probs = np.array(list(domain.values()))
    return rng.choice(list(domain), size=n, p=probs / probs.sum())


def _zscore(x: np.ndarray) -> np.ndarray:
    return (x - x.mean()) / x.std()


def _calibrated_probability(logit: np.ndarray, rate: float) -> np.ndarray:
    """Desloca o intercepto (bisseccao) para a media das probabilidades ser `rate`."""
    lo, hi = -20.0, 20.0
    for _ in range(60):
        mid = (lo + hi) / 2
        if (1 / (1 + np.exp(-(logit + mid)))).mean() > rate:
            hi = mid
        else:
            lo = mid
    return 1 / (1 + np.exp(-(logit + (lo + hi) / 2)))


def make_synthetic_application(
    n: int, seed: int = config.SEED, with_anomalies: bool = True
) -> pd.DataFrame:
    """Gera `n` solicitacoes de credito sinteticas com as colunas de `RAW_COLUMNS`.

    Com `with_anomalies=True`, injeta em 17 linhas distintas: 1 renda de 117 milhoes,
    4 `CODE_GENDER = "XNA"` e 12 `AMT_ANNUITY` nulos (como no dado real).
    """
    if n < N_ANOMALY_ROWS:
        raise ValueError(f"n deve ser >= {N_ANOMALY_ROWS} (recebido: {n}).")
    rng = np.random.default_rng(seed)

    income_type = _draw(rng, _INCOME_TYPES, n)
    no_job_income = np.isin(income_type, ["Pensioner", "Unemployed"])
    pensioner = income_type == "Pensioner"

    # Idade: pensionistas mais velhos; demais em idade ativa.
    age = np.where(
        pensioner,
        np.clip(rng.normal(63, 6, n), 50, 95),
        np.clip(rng.normal(40, 11, n), 20, 68),
    )
    days_birth = -np.floor(age * 365.25 + rng.uniform(0, 1, n))
    age_years = -days_birth / 365.25

    # Tempo de emprego (anos) limitado pela idade; sentinela para pensionista/desempregado.
    years_employed = np.minimum(rng.exponential(6.0, n) + 0.05, np.maximum(age_years - 18, 0.05))
    days_employed = np.where(
        no_job_income, config.DAYS_EMPLOYED_SENTINEL, -np.round(years_employed * 365.25)
    )

    # Renda: log-normal, menor para quem nao tem emprego formal.
    median_income = np.where(no_job_income, 110_000, 150_000)
    income = np.round(median_income * rng.lognormal(0.0, 0.55, n) / 500) * 500
    income = np.clip(income, 25_000, 9_000_000)

    # Credito, anuidade e preco do bem.
    contract = np.where(rng.random(n) < 0.905, "Cash loans", "Revolving loans")
    revolving = contract == "Revolving loans"
    credit = np.where(
        revolving,
        income * rng.lognormal(np.log(1.2), 0.5, n),
        income * rng.lognormal(np.log(3.5), 0.6, n),
    )
    credit = np.round(np.clip(credit, 45_000, 4_050_000) / 500) * 500
    term = rng.choice(_LOAN_TERMS, size=n, p=_LOAN_TERM_PROBS)
    annuity = np.where(revolving, credit * 0.05, credit / term * rng.uniform(1.05, 1.35, n))
    annuity = np.round(annuity, 1)
    goods = np.round(credit * rng.uniform(0.80, 1.0, n) / 500) * 500
    goods[rng.random(n) < 0.001] = np.nan

    # Scores externos: correlacionados entre si (fator latente comum).
    latent = rng.normal(size=n)
    ext = np.column_stack(
        [
            np.clip(0.5 + 0.21 * (0.6 * latent + 0.8 * rng.normal(size=n)), 0.0, 1.0)
            for _ in range(3)
        ]
    )

    # Perfil familiar.
    family_status = _draw(rng, _FAMILY_STATUS, n)
    children = np.minimum(rng.poisson(0.45, n), 6)
    partnered = np.isin(family_status, ["Married", "Civil marriage"])
    fam_members = children + 1 + partnered

    # TARGET: logistica sobre scores externos (mais forte), peso da parcela na renda,
    # idade (mais jovem, mais risco) e desemprego. Intercepto calibrado para ~8%.
    annuity_income = annuity / income
    logit = (
        -0.60 * _zscore(ext[:, 0])
        - 0.80 * _zscore(ext[:, 1])
        - 0.70 * _zscore(ext[:, 2])
        + 0.30 * _zscore(annuity_income)
        - 0.25 * _zscore(age_years)
        + 0.30 * no_job_income
        + 1.20 * (income_type == "Unemployed")
    )
    target = (rng.random(n) < _calibrated_probability(logit, TARGET_RATE)).astype(np.int64)

    # Nulos dos scores entram depois do TARGET: o risco depende do valor latente.
    ext[rng.random(n) < 0.55, 0] = np.nan
    ext[rng.random(n) < 0.002, 1] = np.nan
    ext[rng.random(n) < 0.20, 2] = np.nan

    df = pd.DataFrame(
        {
            "SK_ID_CURR": ID_OFFSET + 1 + np.arange(n),
            "AMT_INCOME_TOTAL": income,
            "AMT_CREDIT": credit,
            "AMT_ANNUITY": annuity,
            "AMT_GOODS_PRICE": goods,
            "DAYS_BIRTH": days_birth,
            "DAYS_EMPLOYED": days_employed,
            "CNT_CHILDREN": children,
            "CNT_FAM_MEMBERS": fam_members,
            "NAME_CONTRACT_TYPE": contract,
            "NAME_INCOME_TYPE": income_type,
            "NAME_EDUCATION_TYPE": _draw(rng, _EDUCATION_TYPES, n),
            "NAME_FAMILY_STATUS": family_status,
            "NAME_HOUSING_TYPE": _draw(rng, _HOUSING_TYPES, n),
            "FLAG_OWN_CAR": np.where(rng.random(n) < 0.34, "Y", "N"),
            "FLAG_OWN_REALTY": np.where(rng.random(n) < 0.69, "Y", "N"),
            "EXT_SOURCE_1": ext[:, 0],
            "EXT_SOURCE_2": ext[:, 1],
            "EXT_SOURCE_3": ext[:, 2],
            "CODE_GENDER": np.where(rng.random(n) < 0.658, "F", "M"),
            "TARGET": target,
        }
    )

    if with_anomalies:
        rows = rng.choice(n, size=N_ANOMALY_ROWS, replace=False)
        df.loc[rows[0], "AMT_INCOME_TOTAL"] = ABSURD_INCOME
        df.loc[rows[1:5], "CODE_GENDER"] = "XNA"
        df.loc[rows[5:], "AMT_ANNUITY"] = np.nan

    return df[config.RAW_COLUMNS].astype(config.RAW_DTYPES)
