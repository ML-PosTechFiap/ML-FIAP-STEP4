import json

import pandas as pd
import pytest

from credit_monitor.data.prepare import clean_reference, split_reference
from tests.test_observability import _settings


def test_clean_reference_catches_real_anomalies(synthetic_raw):
    clean, result = clean_reference(synthetic_raw, "t")
    assert not result.passed
    assert result.violations["income_range"] == 1
    assert result.violations["domain_CODE_GENDER"] == 4
    assert result.violations["annuity_not_null"] == 12
    assert len(clean) == len(synthetic_raw) - 17
    assert "SK_ID_CURR" in clean.columns
    assert "XNA" not in set(clean["CODE_GENDER"].astype(str))


def test_clean_reference_raises_when_clean_frame_still_fails(synthetic_raw):
    bad = synthetic_raw.copy()
    bad["EXT_SOURCE_1"] = 2.0  # falha de coluna inteira: nenhuma linha removida resolve
    with pytest.raises(RuntimeError, match="contrato"):
        clean_reference(bad, "t")


def test_split_proportions_and_stratification(synthetic_raw):
    clean, _ = clean_reference(synthetic_raw, "t")
    parts = split_reference(clean, 42)
    n = len(clean)
    assert set(parts) == {"train", "valid", "test", "pool"}
    assert sum(len(p) for p in parts.values()) == n
    assert abs(len(parts["pool"]) / n - 0.40) < 0.01
    assert abs(len(parts["train"]) / n - 0.42) < 0.01
    assert abs(len(parts["valid"]) / n - 0.09) < 0.01
    assert abs(len(parts["test"]) / n - 0.09) < 0.01
    rates = [p["TARGET"].mean() for p in parts.values()]
    assert max(rates) - min(rates) < 0.01


def test_run_prepare_synthetic_writes_pseudonymized_reference(tmp_path):
    from credit_monitor.data.prepare import run_prepare

    s = _settings(tmp_path)
    paths = run_prepare(s, synthetic=True, n=20_000)
    train = pd.read_parquet(tmp_path / "reference/train.parquet")
    pool = pd.read_parquet(tmp_path / "reference/production_pool.parquet")
    assert "SK_ID_CURR" not in train.columns and "client_key" in train.columns
    assert "SK_ID_CURR" in pool.columns and "client_key" not in pool.columns
    assert json.loads((tmp_path / "reports/contracts/raw_reference_failures.json").read_text())["violations"]
    assert (tmp_path / "reports/contracts/raw_reference_failures.html").exists()
    domains = json.loads((tmp_path / "reference/domains.json").read_text())
    assert domains["CODE_GENDER"] == ["F", "M"]
    assert all(p.exists() for p in paths.values())
    for name in ("valid", "test"):
        part = pd.read_parquet(tmp_path / f"reference/{name}.parquet")
        assert "SK_ID_CURR" not in part.columns


def test_missing_dataset_message(tmp_path, monkeypatch):
    from credit_monitor.data.download import DatasetUnavailable, ensure_dataset

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    with pytest.raises(DatasetUnavailable, match="application_train.csv"):
        ensure_dataset(_settings(tmp_path))


def test_existing_dataset_short_circuits(tmp_path, monkeypatch):
    from credit_monitor.data.download import ensure_dataset

    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    csv = tmp_path / "raw" / "application_train.csv"
    csv.parent.mkdir(parents=True)
    csv.write_text("SK_ID_CURR\n1\n")
    assert ensure_dataset(_settings(tmp_path)) == csv


def test_prepare_command_without_dataset_returns_2(tmp_path, monkeypatch, capsys):
    from credit_monitor.cli import main

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", "t")
    monkeypatch.delenv("KAGGLE_USERNAME", raising=False)
    assert main(["prepare"]) == 2
    assert "application_train.csv" in capsys.readouterr().out


def test_prepare_command_synthetic_prints_counts(tmp_path, monkeypatch, capsys):
    from credit_monitor.cli import main

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", "t")
    assert main(["prepare", "--synthetic", "--n", "5000"]) == 0
    out = capsys.readouterr().out
    assert "5000" in out and "17" in out
    assert (tmp_path / "reference/train.parquet").exists()
