"""Configuracao do pipeline, lida das variaveis de ambiente."""

from __future__ import annotations

import os
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

# --- Limiares de monitoramento (spec 6.3) --------------------------------------

PSI_WARN = 0.10
PSI_DRIFT = 0.20
KS_DRIFT = 0.10
DRIFT_SHARE_ALERT = 0.30
PRED_PSI_ALERT = 0.20
AUC_DROP_ALERT = 0.05
CALIBRATION_ALERT = 1.3
DISPARATE_IMPACT_MIN = 0.8
