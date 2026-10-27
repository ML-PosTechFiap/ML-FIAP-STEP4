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


def test_train_before_prepare_is_friendly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", "t")
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert main(["train"]) == 2
    out = capsys.readouterr().out
    assert "prepare" in out and "Traceback" not in out


def test_run_train_raises_runtime_error_when_reference_is_missing(tmp_path):
    import pytest

    from credit_monitor.model.train import run_train
    from tests.test_observability import _settings

    with pytest.raises(RuntimeError, match="prepare"):
        run_train(_settings(tmp_path))


def test_prepare_reports_reference_contract_failure(tmp_path, monkeypatch, capsys):
    from credit_monitor.data import prepare

    def boom(raw, salt):
        raise RuntimeError("A referência limpa ainda viola o contrato de dados (x).")

    monkeypatch.setattr(prepare, "clean_reference", boom)
    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", "t")
    assert main(["prepare", "--synthetic", "--n", "2000"]) == 2
    assert "ainda viola" in capsys.readouterr().out


def test_placeholder_salt_is_a_friendly_usage_error(tmp_path, monkeypatch, capsys):
    from credit_monitor.privacy.pseudonymize import PLACEHOLDER_SALT

    monkeypatch.setenv("DATA_DIR", str(tmp_path))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", PLACEHOLDER_SALT)
    monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    assert main(["prepare", "--synthetic", "--n", "2000"]) == 2
    out = capsys.readouterr().out
    assert "secrets.token_hex" in out and "Traceback" not in out
