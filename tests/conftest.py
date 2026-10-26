"""Fixtures compartilhadas."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_monitor.data.synthetic import make_synthetic_application


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
