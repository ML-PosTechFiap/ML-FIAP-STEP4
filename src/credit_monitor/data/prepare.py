"""Preparação da referência: limpeza guiada pelo contrato, split e gravação."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from credit_monitor.config import RAW_COLUMNS, RAW_DTYPES, SEED, Settings
from credit_monitor.contracts.schema import domains_from_reference
from credit_monitor.contracts.validate import (
    ContractResult,
    validate_batch,
    write_failure_report,
)
from credit_monitor.data.download import ensure_dataset
from credit_monitor.data.simulate import build_all_batches
from credit_monitor.data.synthetic import make_synthetic_application
from credit_monitor.privacy.pseudonymize import pseudonymize

REFERENCE_FRACTION = 0.60  # o restante (40%) é o pool de produção
TRAIN_FRACTION = 0.70  # da referência; validação e teste dividem o resto igualmente


def load_raw(path: Path) -> pd.DataFrame:
    """Lê só as colunas do contrato (minimização), com os dtypes esperados."""
    return pd.read_csv(path, usecols=RAW_COLUMNS, dtype=RAW_DTYPES)[RAW_COLUMNS]


def clean_reference(raw: pd.DataFrame, salt: str) -> tuple[pd.DataFrame, ContractResult]:
    """Remove do bruto as linhas que violam o contrato.

    Devolve o bruto limpo (ainda com `SK_ID_CURR`) e o resultado da validação original.
    """
    raw = raw.reset_index(drop=True)  # índice único: `validate_batch` mapeia client_key por rótulo
    pseudo = pseudonymize(raw, salt)
    domains = domains_from_reference(raw[raw["CODE_GENDER"].astype(str) != "XNA"])
    result = validate_batch(pseudo, domains)

    recheck = validate_batch(pseudo.drop(index=result.failing_index), domains)
    if not recheck.passed:
        raise RuntimeError(
            "A referência limpa ainda viola o contrato de dados "
            f"({recheck.violations}); revise a limpeza antes de continuar."
        )
    clean = raw.drop(index=result.failing_index).reset_index(drop=True)
    for col in clean.select_dtypes("category").columns:
        clean[col] = clean[col].cat.remove_unused_categories()
    return clean, result


def split_reference(clean: pd.DataFrame, seed: int) -> dict[str, pd.DataFrame]:
    """Divide em train/valid/test (60% x 70/15/15) e pool (40%), estratificado por TARGET."""
    reference, pool = train_test_split(
        clean, train_size=REFERENCE_FRACTION, stratify=clean["TARGET"], random_state=seed
    )
    train, rest = train_test_split(
        reference, train_size=TRAIN_FRACTION, stratify=reference["TARGET"], random_state=seed
    )
    valid, test = train_test_split(
        rest, train_size=0.5, stratify=rest["TARGET"], random_state=seed
    )
    return {"train": train, "valid": valid, "test": test, "pool": pool}


def run_prepare(settings: Settings, synthetic: bool, n: int) -> dict[str, Path]:
    """Grava a referência pseudonimizada, o pool bruto, os domínios, o relatório do contrato
    e os lotes mensais simulados (`data/batches/<id>/`)."""
    if synthetic:
        raw = make_synthetic_application(n)
    else:
        raw = load_raw(ensure_dataset(settings))

    clean, result = clean_reference(raw, settings.pseudonymization_salt)
    parts = split_reference(clean, SEED)

    ref_dir = settings.data_dir / "reference"
    ref_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for name in ("train", "valid", "test"):
        paths[name] = ref_dir / f"{name}.parquet"
        pseudonymized = pseudonymize(parts[name], settings.pseudonymization_salt)
        pseudonymized.reset_index(drop=True).to_parquet(paths[name], index=False)
    paths["pool"] = ref_dir / "production_pool.parquet"
    parts["pool"].reset_index(drop=True).to_parquet(paths["pool"], index=False)

    paths["domains"] = ref_dir / "domains.json"
    paths["domains"].write_text(
        json.dumps(domains_from_reference(clean), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    paths["failures_json"], paths["failures_html"] = write_failure_report(
        result, "raw_reference", settings.reports_dir / "contracts"
    )
    build_all_batches(settings, parts["pool"], parts["train"], SEED)
    return paths
