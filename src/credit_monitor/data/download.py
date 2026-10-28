"""Obtenção do application_train.csv do Home Credit (Kaggle)."""

from __future__ import annotations

import os
import subprocess
import zipfile
from pathlib import Path

from credit_monitor.config import Settings

COMPETITION = "home-credit-default-risk"
DATASET_FILE = "application_train.csv"
KAGGLE_TIMEOUT_SECONDS = 600


class DatasetUnavailable(RuntimeError):
    """O dataset não está em `data/raw` e não pode ser baixado automaticamente."""


def _unavailable_message(reason: str = "") -> str:
    prefix = f"{reason}\n" if reason else ""
    return (
        f"{prefix}O arquivo {DATASET_FILE} não foi encontrado em data/raw. Há dois caminhos:\n"
        f"  1. Manual: baixe {DATASET_FILE} em "
        f"https://www.kaggle.com/competitions/{COMPETITION}/data e salve em data/raw/.\n"
        "  2. Automático: defina KAGGLE_USERNAME e KAGGLE_KEY no .env e rode 'download' "
        f"(antes, aceite as regras da competição '{COMPETITION}' na página dela no Kaggle).\n"
        "Para validar o sistema sem o dataset real, use 'prepare --synthetic'."
    )


def ensure_dataset(settings: Settings) -> Path:
    """Devolve o caminho do CSV, baixando do Kaggle quando faltar e houver credenciais."""
    raw_dir = settings.data_dir / "raw"
    csv_path = raw_dir / DATASET_FILE
    if csv_path.exists():
        return csv_path
    if not (os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY")):
        raise DatasetUnavailable(_unavailable_message())

    raw_dir.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-f", DATASET_FILE,
           "-p", str(raw_dir)]
    try:
        subprocess.run(cmd, check=True, capture_output=True, text=True,
                       timeout=KAGGLE_TIMEOUT_SECONDS)
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired) as exc:
        detail = getattr(exc, "stderr", "") or str(exc)
        raise DatasetUnavailable(
            _unavailable_message(f"Falha ao baixar do Kaggle: {detail.strip()}")
        ) from exc

    zip_path = raw_dir / f"{DATASET_FILE}.zip"
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as archive:
                archive.extract(DATASET_FILE, raw_dir)
        except (zipfile.BadZipFile, KeyError) as exc:
            raise DatasetUnavailable(
                _unavailable_message(f"O arquivo baixado do Kaggle está corrompido ({exc}).")
            ) from exc
        zip_path.unlink()
    if not csv_path.exists():
        raise DatasetUnavailable(_unavailable_message("O download terminou sem gerar o arquivo."))
    return csv_path
