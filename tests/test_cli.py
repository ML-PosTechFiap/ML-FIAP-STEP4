from credit_monitor.cli import build_parser, main


def test_help_lists_program():
    assert "credit_monitor" in build_parser().format_help()


def test_unknown_command_returns_2(capsys):
    assert main(["nao-existe"]) == 2


def test_settings_from_env(monkeypatch, tmp_path):
    from credit_monitor.config import Settings

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", "s")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    s = Settings.from_env()
    assert s.data_dir == tmp_path and s.mlflow_uri is None
