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
