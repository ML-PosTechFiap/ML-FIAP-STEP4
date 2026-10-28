"""Carga do modelo treinado e pontuacao de lotes."""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from credit_monitor.config import Settings
from credit_monitor.model.features import model_matrix

MODEL_FILE = "model.joblib"
METADATA_FILE = "metadata.json"


class ModelNotTrained(Exception):
    """Nao ha modelo treinado em `models/`."""


def load_model(settings: Settings) -> tuple[Pipeline, dict]:
    """Le `models/model.joblib` e `models/metadata.json`."""
    path = settings.models_dir / MODEL_FILE
    if not path.exists():
        raise ModelNotTrained(
            f"Modelo não encontrado em {path}. Execute primeiro o comando `train`."
        )
    metadata = json.loads((settings.models_dir / METADATA_FILE).read_text(encoding="utf-8"))
    return joblib.load(path), metadata


def score(pipeline: Pipeline, df: pd.DataFrame) -> np.ndarray:
    """Probabilidade de inadimplencia (classe 1)."""
    return pipeline.predict_proba(model_matrix(df))[:, 1]
