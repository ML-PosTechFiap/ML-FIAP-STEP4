"""Configuracao do pipeline, lida das variaveis de ambiente."""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    data_dir: Path
    reports_dir: Path
    models_dir: Path
    mlflow_uri: str | None
    pushgateway_url: str | None
    loki_url: str | None
    pseudonymization_salt: str
    http_timeout: float = 2.0

    @classmethod
    def from_env(cls) -> Settings:
        env = os.environ
        return cls(
            data_dir=Path(env.get("DATA_DIR", "/app/data")),
            reports_dir=Path(env.get("REPORTS_DIR", "/app/reports")),
            models_dir=Path(env.get("MODELS_DIR", "/app/models")),
            mlflow_uri=env.get("MLFLOW_TRACKING_URI") or None,
            pushgateway_url=env.get("PUSHGATEWAY_URL") or None,
            loki_url=env.get("LOKI_URL") or None,
            pseudonymization_salt=env.get("PSEUDONYMIZATION_SALT", ""),
        )


# --- Dados -------------------------------------------------------------------

SEED = 42
DAYS_EMPLOYED_SENTINEL = 365243
INCOME_MAX = 10_000_000
AGE_MIN = 18
AGE_MAX = 100
MIN_BATCH_ROWS = 1_000
BATCH_SIZE = 20_000
SIMULATION_ORDER = ["01", "02", "03-dirty", "03", "04", "05", "06"]

# Colunas lidas do application_train.csv (minimizacao: usecols), na ordem da spec 4.1.
RAW_COLUMNS: list[str] = [
    "SK_ID_CURR",
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "DAYS_BIRTH",
    "DAYS_EMPLOYED",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CODE_GENDER",
    "TARGET",
]

CATEGORICAL_FEATURES: list[str] = [
    "NAME_CONTRACT_TYPE",
    "NAME_INCOME_TYPE",
    "NAME_EDUCATION_TYPE",
    "NAME_FAMILY_STATUS",
    "NAME_HOUSING_TYPE",
    "FLAG_OWN_CAR",
    "FLAG_OWN_REALTY",
]

# CODE_GENDER fica de fora: so e usada na auditoria de vies, nunca como feature.
NUMERIC_FEATURES: list[str] = [
    "AMT_INCOME_TOTAL",
    "AMT_CREDIT",
    "AMT_ANNUITY",
    "AMT_GOODS_PRICE",
    "AGE_YEARS",
    "YEARS_EMPLOYED",
    "IS_UNEMPLOYED_OR_PENSIONER",
    "CNT_CHILDREN",
    "CNT_FAM_MEMBERS",
    "EXT_SOURCE_1",
    "EXT_SOURCE_2",
    "EXT_SOURCE_3",
    "CREDIT_INCOME_RATIO",
    "ANNUITY_INCOME_RATIO",
    "CREDIT_TERM",
]

MODEL_FEATURES: list[str] = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# --- Modelo -------------------------------------------------------------------------

XGB_PARAMS: dict = dict(
    n_estimators=600,
    learning_rate=0.05,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    eval_metric="auc",
    early_stopping_rounds=50,
    random_state=SEED,
)
# (inicio, fim exclusivo, rotulo) em anos completos, para a auditoria de vies.
AGE_BANDS: list[tuple[int, int, str]] = [
    (0, 30, "<30"),
    (30, 45, "30-45"),
    (45, 60, "45-60"),
    (60, 200, "60+"),
]

_INTEGER_COLUMNS = {"SK_ID_CURR", "TARGET"}
_CATEGORICAL_COLUMNS = set(CATEGORICAL_FEATURES) | {"CODE_GENDER"}
RAW_DTYPES: dict[str, str] = {
    col: (
        "int64"
        if col in _INTEGER_COLUMNS
        else "category"
        if col in _CATEGORICAL_COLUMNS
        else "float32"
    )
    for col in RAW_COLUMNS
}

# --- Retencao (dias, medidos pelo mtime do arquivo) --------------------------------

RETENTION_DAYS = {"quarantine": 30, "logs": 90}

# --- Limiares de monitoramento (spec 6.3) --------------------------------------

PSI_WARN = 0.10
PSI_DRIFT = 0.20
KS_DRIFT = 0.10
DRIFT_SHARE_ALERT = 0.30
PRED_PSI_ALERT = 0.20
AUC_DROP_ALERT = 0.05
CALIBRATION_ALERT = 1.3
DISPARATE_IMPACT_MIN = 0.8


# --- Cenario de drift (spec 6.1) ---------------------------------------------------


@dataclass(frozen=True)
class DriftParams:
    """Parametros de drift de um lote mensal. Zero (ou `None`) = sem alteracao.

    `credit_infl` vale tambem para `AMT_GOODS_PRICE`. `employed_shrink` encolhe o
    `DAYS_EMPLOYED` dos empregados restantes. `young_weight` pondera a amostragem por
    `1 + young_weight * (idade < 30)`. `concept_flip` e a fracao dos `TARGET == 0` do
    segmento "espremido" virada para 1.
    """

    income_infl: float = 0.0
    credit_infl: float = 0.0
    annuity_infl: float = 0.0
    unemployed_target: float | None = None
    employed_shrink: float = 0.0
    young_weight: float = 0.0
    ext1_extra_null: float = 0.0
    ext23_shrink: float = 0.0
    concept_flip: float = 0.0


DRIFT_SCHEDULE: dict[str, DriftParams] = {
    "01": DriftParams(),
    "02": DriftParams(income_infl=0.02, credit_infl=0.04, annuity_infl=0.06),
    "03": DriftParams(
        income_infl=0.05,
        credit_infl=0.10,
        annuity_infl=0.14,
        unemployed_target=0.22,
        employed_shrink=0.05,
    ),
    "04": DriftParams(0.08, 0.16, 0.22, 0.25, 0.10, 1.0, 0.10, 0.05, 0.10),
    "05": DriftParams(0.10, 0.20, 0.28, 0.28, 0.15, 2.0, 0.20, 0.10, 0.18),
    "06": DriftParams(0.12, 0.25, 0.35, 0.30, 0.20, 3.0, 0.30, 0.15, 0.25),
}

# Features do modelo em que cada parametro injeta drift (alimenta o `manifest.json`).
# Cada regra: (parametro esta ativo?, features afetadas). Facil de editar: a Tarefa 9
# confere a revocacao da deteccao contra esta lista.
_RATIOS = ["CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO"]
DRIFT_EFFECTS: list[tuple[Callable[[DriftParams], bool], list[str]]] = [
    (lambda p: p.income_infl != 0, ["AMT_INCOME_TOTAL", *_RATIOS]),
    (lambda p: p.credit_infl != 0, ["AMT_CREDIT", "AMT_GOODS_PRICE", *_RATIOS]),
    (lambda p: p.annuity_infl != 0, ["AMT_ANNUITY", *_RATIOS]),
    # credito / anuidade so se desloca quando as inflacoes diferem
    (lambda p: p.credit_infl != p.annuity_infl, ["CREDIT_TERM"]),
    (
        lambda p: p.unemployed_target is not None or p.employed_shrink > 0,
        ["YEARS_EMPLOYED", "IS_UNEMPLOYED_OR_PENSIONER", "NAME_INCOME_TYPE"],
    ),
    (lambda p: p.young_weight > 0, ["AGE_YEARS"]),
    (lambda p: p.ext1_extra_null > 0, ["EXT_SOURCE_1"]),
    (lambda p: p.ext23_shrink > 0, ["EXT_SOURCE_2", "EXT_SOURCE_3"]),
]


def drifted_features(params: DriftParams) -> list[str]:
    """`MODEL_FEATURES` esperadas em drift para os parametros dados (ordenadas)."""
    found = {f for active, features in DRIFT_EFFECTS if active(params) for f in features}
    return sorted(found)
