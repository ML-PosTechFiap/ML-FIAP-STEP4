import os
import time

import numpy as np
import pandas as pd
import pytest

from credit_monitor.privacy.pseudonymize import client_key, pseudonymize


def test_deterministic_and_salted():
    assert client_key(100002, "a") == client_key(100002.0, "a")
    assert client_key(100002, "a") != client_key(100002, "b")
    assert len(client_key(1, "a")) == 16 and client_key(None, "a") is None


def test_client_key_accepts_numpy_ints_and_treats_all_nulls_as_null():
    assert client_key(np.int64(7), "a") == client_key(7, "a")
    assert client_key(np.float32(7.0), "a") == client_key(7, "a")
    for null in (None, float("nan"), np.nan, pd.NA):
        assert client_key(null, "a") is None


def test_client_key_is_not_a_plain_hash_of_the_id():
    import hashlib

    assert client_key(5, "a") != hashlib.sha256(b"5").hexdigest()[:16]


def test_empty_salt_is_rejected():
    with pytest.raises(ValueError, match="PSEUDONYMIZATION_SALT"):
        client_key(1, "")
    with pytest.raises(ValueError, match="PSEUDONYMIZATION_SALT"):
        pseudonymize(pd.DataFrame({"SK_ID_CURR": [1]}), "")


def test_pseudonymize_drops_raw_id():
    out = pseudonymize(pd.DataFrame({"SK_ID_CURR": [1, 1, None], "X": [1, 2, 3]}), "s")
    assert list(out.columns) == ["client_key", "X"]
    assert out["client_key"].iloc[0] == out["client_key"].iloc[1]
    assert out["client_key"].isna().iloc[2]


def test_pseudonymize_puts_client_key_first_and_keeps_input_intact():
    df = pd.DataFrame({"X": [1, 2], "SK_ID_CURR": [10, 20], "Y": ["a", "b"]})
    snapshot = df.copy(deep=True)
    out = pseudonymize(df, "s")
    assert list(out.columns) == ["client_key", "X", "Y"]
    assert out["client_key"].tolist() == [client_key(10, "s"), client_key(20, "s")]
    pd.testing.assert_frame_equal(df, snapshot)


def test_pseudonymize_requires_id_column():
    with pytest.raises(ValueError, match="SK_ID_CURR"):
        pseudonymize(pd.DataFrame({"X": [1]}), "s")


def test_purge_respects_windows(tmp_path):
    from credit_monitor.privacy.retention import purge
    from tests.test_observability import _settings

    q = tmp_path / "quarantine"
    q.mkdir(parents=True)
    logs = tmp_path / "reports/logs"
    logs.mkdir(parents=True)
    old_q, new_q, old_l = q / "a.parquet", q / "b.parquet", logs / "x.jsonl"
    for p in (old_q, new_q, old_l):
        p.write_text("x")
    old = time.time() - 100 * 86400
    os.utime(old_q, (old, old))
    os.utime(old_l, (old, old))
    assert purge(_settings(tmp_path)) == {"quarantine": 1, "logs": 1}
    assert new_q.exists() and not old_q.exists()


def test_purge_uses_per_area_windows_and_skips_directories(tmp_path):
    from credit_monitor.privacy.retention import purge
    from tests.test_observability import _settings

    q = tmp_path / "quarantine"
    logs = tmp_path / "reports/logs"
    (q / "sub").mkdir(parents=True)
    logs.mkdir(parents=True)
    q_45d, l_45d = q / "q.parquet", logs / "l.jsonl"
    nested = q / "sub" / "deep.parquet"
    for p in (q_45d, l_45d, nested):
        p.write_text("x")
    forty_five = time.time() - 45 * 86400
    for p in (q_45d, l_45d, nested, q / "sub"):
        os.utime(p, (forty_five, forty_five))
    # 45 dias: fora da janela da quarentena (30), dentro da dos logs (90).
    assert purge(_settings(tmp_path)) == {"quarantine": 1, "logs": 0}
    assert not q_45d.exists() and l_45d.exists()
    assert (q / "sub").is_dir() and nested.exists()


def test_purge_missing_directories_count_zero(tmp_path):
    from credit_monitor.privacy.retention import purge
    from tests.test_observability import _settings

    assert purge(_settings(tmp_path)) == {"quarantine": 0, "logs": 0}


def test_purge_accepts_explicit_now(tmp_path):
    from datetime import datetime, timedelta

    from credit_monitor.privacy.retention import purge
    from tests.test_observability import _settings

    q = tmp_path / "quarantine"
    q.mkdir()
    f = q / "a.parquet"
    f.write_text("x")
    future = datetime.now() + timedelta(days=31)
    assert purge(_settings(tmp_path), now=future) == {"quarantine": 1, "logs": 0}
    assert not f.exists()


def test_purge_cli_command_prints_counts_and_returns_0(monkeypatch, tmp_path, capsys):
    from credit_monitor.cli import COMMANDS, main

    assert "purge" in COMMANDS
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("REPORTS_DIR", str(tmp_path / "reports"))
    q = tmp_path / "quarantine"
    q.mkdir()
    old_file = q / "a.parquet"
    old_file.write_text("x")
    old = time.time() - 40 * 86400
    os.utime(old_file, (old, old))
    assert main(["purge"]) == 0
    out = capsys.readouterr().out
    assert "quarentena" in out and "1" in out
    assert not old_file.exists()
