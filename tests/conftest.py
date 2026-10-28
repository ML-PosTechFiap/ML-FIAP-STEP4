"""Fixtures compartilhadas."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_monitor.data.synthetic import make_synthetic_application


@pytest.fixture(autouse=True)
def _isolated_dirs(tmp_path_factory, monkeypatch):
    """Aponta DATA/REPORTS/MODELS_DIR para um diretorio temporario em todo teste.

    Sem isso, testes que chamam `main()` via `Settings.from_env()` e nao definem
    todas as variaveis caem nos defaults `/app/...`: no container gravam no
    `reports/` do repositorio e no CI falham (`/app` nao existe). Testes que
    definem as variaveis sobrescrevem estes valores.
    """
    root = tmp_path_factory.mktemp("env")
    for name in ("DATA_DIR", "REPORTS_DIR", "MODELS_DIR"):
        monkeypatch.setenv(name, str(root / name.removesuffix("_DIR").lower()))


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    """Base sintetica bruta (com as 17 anomalias) no schema do Home Credit."""
    return make_synthetic_application(30_000)


@pytest.fixture(scope="session")
def prepared(tmp_path_factory):
    """Settings apos `run_prepare` sintetico (referencia + lotes); models_dir separado."""
    from credit_monitor.data.prepare import run_prepare
    from tests.test_observability import _settings

    root = tmp_path_factory.mktemp("sim")
    s = _settings(root, models_dir=root / "models")
    run_prepare(s, synthetic=True, n=60_000)
    return s


@pytest.fixture(scope="session")
def trained(prepared):
    """Settings de `prepared` apos `run_train`: models/ e test_scored.parquet gravados."""
    from credit_monitor.model.train import run_train

    run_train(prepared)
    return prepared
