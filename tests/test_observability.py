import json
import os
import time
import types

import pytest

from credit_monitor.observability.logging import get_logger
from credit_monitor.observability.metrics import METRIC_CATALOG, MetricsPublisher
from credit_monitor.observability.stage import stage
from credit_monitor.observability.tracking import Tracker

DEAD = "http://10.255.255.1:9"  # não roteável: força timeout


def _settings(tmp_path, **kw):
    from credit_monitor.config import Settings

    base = dict(data_dir=tmp_path, reports_dir=tmp_path / "reports", models_dir=tmp_path,
                mlflow_uri=None, pushgateway_url=None, loki_url=None,
                pseudonymization_salt="t", http_timeout=2.0)
    return Settings(**{**base, **kw})


def test_stage_success_records_status_and_log(tmp_path):
    s = _settings(tmp_path)
    pub = MetricsPublisher(s, "01")
    log = get_logger(s, "r1")
    with stage("contract", "01", pub, log) as ctx:
        ctx.rows_in = ctx.rows_out = 100
    v = pub.values()
    assert v[("credit_pipeline_stage_status", (("stage", "contract"),))] == 1
    line = json.loads((tmp_path / "reports/logs/pipeline.jsonl").read_text().splitlines()[-1])
    assert line["stage"] == "contract" and line["rows_out"] == 100 and line["batch_id"] == "01"


def test_stage_block_and_error(tmp_path):
    s = _settings(tmp_path)
    pub = MetricsPublisher(s, "01")
    log = get_logger(s, "r1")
    with stage("contract", "01", pub, log) as ctx:
        ctx.block("contrato violado")
    with pytest.raises(ValueError):
        with stage("predict", "01", pub, log):
            raise ValueError("x")
    v = pub.values()
    assert v[("credit_pipeline_stage_status", (("stage", "contract"),))] == -1
    assert v[("credit_pipeline_stage_status", (("stage", "predict"),))] == 0


def test_unknown_metric_rejected(tmp_path):
    with pytest.raises(KeyError):
        MetricsPublisher(_settings(tmp_path), "01").set("credit_nao_existe", 1)


def test_dead_infra_does_not_hang(tmp_path):
    s = _settings(tmp_path, mlflow_uri=DEAD, pushgateway_url=DEAD, loki_url=DEAD)
    t0 = time.monotonic()
    log = get_logger(s, "r1")
    pub = MetricsPublisher(s, "01")
    for _ in range(20):
        log.info("evento", extra={"batch_id": "01"})
    assert pub.push() is False and Tracker(s).enabled is False
    assert time.monotonic() - t0 < 10


def test_catalog_has_no_batch_label():
    assert all("batch" not in labels for _, labels in METRIC_CATALOG.values())


class _Ok:
    def raise_for_status(self):
        pass


def _fake_mlflow(monkeypatch, *, fail_on):
    """Troca as chamadas do mlflow por fakes; `fail_on` e o nome da chamada que levanta."""
    import mlflow

    calls = []

    def make(name, result=None):
        def fn(*args, **kwargs):
            calls.append(name)
            if name == fail_on:
                raise RuntimeError("servidor caiu")
            return result

        return fn

    run = types.SimpleNamespace(info=types.SimpleNamespace(run_id="abc"))
    for name in ("set_tracking_uri", "set_experiment", "log_metrics", "end_run"):
        monkeypatch.setattr(mlflow, name, make(name))
    monkeypatch.setattr(mlflow, "start_run", make("start_run", run))
    return calls


def test_tracker_bounds_mlflow_client_timeout(tmp_path, monkeypatch):
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_TIMEOUT", "999")
    monkeypatch.setenv("MLFLOW_HTTP_REQUEST_MAX_RETRIES", "999")
    monkeypatch.setattr("requests.get", lambda *a, **k: _Ok())
    Tracker(_settings(tmp_path, mlflow_uri="http://mlflow:5000", http_timeout=2.0))
    assert os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] == "2"
    assert os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] == "0"


def test_tracker_disabled_mid_run_skips_end_run(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Ok())
    calls = _fake_mlflow(monkeypatch, fail_on="log_metrics")
    t = Tracker(_settings(tmp_path, mlflow_uri="http://mlflow:5000"))
    assert t.enabled is True
    with t.run("exp", "run", {}) as run_id:
        assert run_id == "abc" and t.active_run_id == "abc"
        t.log_metrics({"auc": 0.7})
        assert t.enabled is False
        t.log_metrics({"auc": 0.8})  # no-op: nao tenta de novo
    assert calls.count("log_metrics") == 1
    assert "end_run" not in calls and t.active_run_id is None


def test_tracker_healthy_run_calls_end_run(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Ok())
    calls = _fake_mlflow(monkeypatch, fail_on=None)
    t = Tracker(_settings(tmp_path, mlflow_uri="http://mlflow:5000"))
    with t.run("exp", "run", {}):
        t.log_metrics({"auc": 0.7})
    assert calls[-1] == "end_run" and t.enabled is True


def _fake_model_registry(monkeypatch, *, fail_on):
    """Fakes de `mlflow.sklearn.log_model` e `MlflowClient`; `fail_on` e a chamada que levanta."""
    import mlflow
    import mlflow.sklearn

    calls = []

    def log_model(model, artifact_path, registered_model_name=None, **kw):
        calls.append(("log_model", artifact_path, registered_model_name))
        if fail_on == "log_model":
            raise RuntimeError("servidor caiu")
        return types.SimpleNamespace(registered_model_version="7")

    class FakeClient:
        def set_registered_model_alias(self, name, alias, version):
            calls.append(("alias", name, alias, version))
            if fail_on == "alias":
                raise RuntimeError("servidor caiu")

    monkeypatch.setattr(mlflow.sklearn, "log_model", log_model)
    monkeypatch.setattr(mlflow, "MlflowClient", FakeClient)
    return calls


def test_tracker_log_model_registers_and_sets_alias(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Ok())
    _fake_mlflow(monkeypatch, fail_on=None)
    calls = _fake_model_registry(monkeypatch, fail_on=None)
    t = Tracker(_settings(tmp_path, mlflow_uri="http://mlflow:5000"))
    with t.run("exp", "run", {}):
        assert t.log_model(object(), "credit-scoring") == "7"
    assert calls == [("log_model", "model", "credit-scoring"),
                     ("alias", "credit-scoring", "production", "7")]
    assert t.enabled is True


@pytest.mark.parametrize("fail_on", ["log_model", "alias"])
def test_tracker_log_model_failure_disables(tmp_path, monkeypatch, fail_on):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Ok())
    _fake_mlflow(monkeypatch, fail_on=None)
    _fake_model_registry(monkeypatch, fail_on=fail_on)
    t = Tracker(_settings(tmp_path, mlflow_uri="http://mlflow:5000"))
    with t.run("exp", "run", {}):
        assert t.log_model(object(), "credit-scoring") is None
        assert t.enabled is False


def test_tracker_log_model_noop_when_disabled(tmp_path):
    assert Tracker(_settings(tmp_path)).log_model(object(), "credit-scoring") is None


def test_tracker_setup_failure_disables_without_end_run(tmp_path, monkeypatch):
    monkeypatch.setattr("requests.get", lambda *a, **k: _Ok())
    calls = _fake_mlflow(monkeypatch, fail_on="set_experiment")
    t = Tracker(_settings(tmp_path, mlflow_uri="http://mlflow:5000"))
    with t.run("exp", "run", {}) as run_id:
        assert run_id is None
    assert t.enabled is False and "end_run" not in calls


def test_warnings_reach_json_log(tmp_path, monkeypatch):
    def boom(*a, **k):
        raise OSError("fora")

    monkeypatch.setattr("requests.get", boom)
    s = _settings(tmp_path, mlflow_uri="http://mlflow:5000")  # sem pushgateway
    log = get_logger(s, "r9")
    assert MetricsPublisher(s, "07", logger=log).push() is False
    assert Tracker(s, logger=log).enabled is False
    lines = [
        json.loads(x)
        for x in (tmp_path / "reports/logs/pipeline.jsonl").read_text().splitlines()
    ]
    warns = [x for x in lines if x["level"] == "WARNING"]
    assert len(warns) == 2 and all(x["run_id"] == "r9" for x in warns)
    assert any("Pushgateway" in x["message"] and x["batch_id"] == "07" for x in warns)
    assert any("MLflow" in x["message"] for x in warns)
