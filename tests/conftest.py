"""Fixtures compartilhadas."""

from __future__ import annotations

import pandas as pd
import pytest

from credit_monitor.data.synthetic import make_synthetic_application


@pytest.fixture(scope="session")
def synthetic_raw() -> pd.DataFrame:
    """Base sintetica bruta (com as 17 anomalias) no schema do Home Credit."""
    return make_synthetic_application(30_000)
