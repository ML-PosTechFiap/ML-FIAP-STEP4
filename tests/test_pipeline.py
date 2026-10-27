import json
import re
import urllib.error

import pandas as pd
import pytest

from credit_monitor import pipeline
from credit_monitor.cli import main, parse_interval
from credit_monitor.observability.report_index import build_index
from credit_monitor.pipeline import BatchNotFound, run_monitor, run_simulate_all


@pytest.fixture(scope="module")
def simulated(trained):
    """Uma unica simulacao completa, compartilhada pelos testes do modulo."""
    return {r.batch_id: r for r in run_simulate_all(trained)}


def test_simulate_all_end_to_end(trained, simulated):
    results = simulated
    assert list(results) == ["01", "02", "03-dirty", "03", "04", "05", "06"]
    assert results["03-dirty"].status == "blocked"
    assert results["03-dirty"].summary_path is None and results["03-dirty"].recommendation is None
    assert all(results[b].status == "ok" for b in ["01", "02", "03", "04", "05", "06"])
    assert results["01"].recommendation == "OK"
    assert results["06"].recommendation in {"INVESTIGAR", "RETREINAR"}
    assert results["06"].summary_path.exists()
    assert (trained.reports_dir / "index.html").exists()
    assert (trained.data_dir / "quarantine/03-dirty.parquet").exists()
    assert (trained.data_dir / "batches/03-dirty/batch.parquet").exists()  # copiado, nao movido
    assert (trained.reports_dir / "contracts/03-dirty_failures.json").exists()


def test_index_lists_every_simulated_batch(trained, simulated):
    text = (trained.reports_dir / "index.html").read_text(encoding="utf-8")
    positions = [text.index(f"<td>{b}</td>") for b in simulated]
    assert positions == sorted(positions)
    assert "BLOQUEADO" in text and "OK" in text


def test_blocked_batch_can_rerun(trained):
    assert run_monitor(trained, "03-dirty").status == "blocked"
    assert run_monitor(trained, "03-dirty").status == "blocked"


def test_no_raw_ids_leak(trained, simulated):
    ids = set(
        pd.read_parquet(trained.data_dir / "reference/production_pool.parquet")["SK_ID_CURR"]
        .astype("int64")
        .astype(str)
    )
    scanned = 0
    for path in trained.reports_dir.rglob("*"):
        if path.is_file() and path.suffix in {".json", ".jsonl", ".html"}:
            scanned += 1
            found = set(re.findall(r"\b9\d{8}\b", path.read_text(errors="ignore"))) & ids
            assert not found, f"ID bruto em {path}"
    assert scanned > 10


def test_cli_errors_are_friendly(trained, tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(trained.data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(trained.reports_dir))
    monkeypatch.setenv("MODELS_DIR", str(trained.models_dir))
    assert main(["monitor", "--batch", "99"]) == 2
    captured = capsys.readouterr()
    assert "Traceback" not in captured.err
    assert "99" in captured.out and "prepare" in captured.out
    monkeypatch.setenv("MODELS_DIR", str(tmp_path / "vazio"))
    assert main(["monitor", "--batch", "01"]) == 2
    assert "train" in capsys.readouterr().out


def test_cli_monitor_exit_codes(trained, monkeypatch, capsys):
    monkeypatch.setenv("DATA_DIR", str(trained.data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(trained.reports_dir))
    monkeypatch.setenv("MODELS_DIR", str(trained.models_dir))
    monkeypatch.setenv("PSEUDONYMIZATION_SALT", trained.pseudonymization_salt)
    for var in ("MLFLOW_TRACKING_URI", "PUSHGATEWAY_URL", "LOKI_URL"):
        monkeypatch.delenv(var, raising=False)
    assert main(["monitor", "--batch", "03-dirty"]) == 1
    assert "bloqueado" in capsys.readouterr().out
    assert main(["monitor", "--batch", "01"]) == 0
    assert "OK" in capsys.readouterr().out


def test_batch_not_found_is_an_exception(trained):
    with pytest.raises(BatchNotFound) as exc:
        run_monitor(trained, "99")
    assert "99" in str(exc.value)


# --- publicacao de metricas e MLflow (com dublês) -----------------------------------


@pytest.fixture
def recorded(monkeypatch):
    """Substitui MetricsPublisher e Tracker por subclasses que registram o que recebem."""
    publishers, trackers = [], []

    class Publisher(pipeline.MetricsPublisher):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.pushes = 0
            publishers.append(self)

        def push(self):
            self.pushes += 1
            return True

    class Tracker:
        enabled = True

        def __init__(self, settings, logger=None):
            self.experiment = self.run_name = self.tags = None
            self.metrics, self.artifacts = {}, []
            trackers.append(self)

        def run(self, experiment, run_name, tags=None):
            from contextlib import nullcontext

            self.experiment, self.run_name, self.tags = experiment, run_name, tags
            return nullcontext("run-id")

        def log_metrics(self, d):
            self.metrics.update(d)

        def log_params(self, d):
            pass

        def log_artifact(self, path):
            self.artifacts.append(path)

    monkeypatch.setattr(pipeline, "MetricsPublisher", Publisher)
    monkeypatch.setattr(pipeline, "Tracker", Tracker)
    return publishers, trackers


def _metric(publisher, name, **labels):
    return publisher.values()[(name, tuple(sorted((k, str(v)) for k, v in labels.items())))]


def test_ok_batch_publishes_all_metrics_once(trained, recorded):
    publishers, trackers = recorded
    result = run_monitor(trained, "06")
    assert result.status == "ok"
    pub = publishers[-1]
    assert pub.pushes == 1
    summary = json.loads(result.summary_path.read_text(encoding="utf-8"))

    assert _metric(pub, "credit_contract_passed") == 1
    for stage_name in ["ingest", "contract", "transform", "predict", "drift"]:
        assert _metric(pub, "credit_pipeline_stage_status", stage=stage_name) == 1
    assert _metric(pub, "credit_dataset_drift_share") == summary["dataset_drift_share"]
    assert _metric(pub, "credit_prediction_psi") == summary["prediction_psi"]
    codes = {"OK": 0, "INVESTIGAR": 1, "RETREINAR": 2}
    assert _metric(pub, "credit_monitor_recommendation") == codes[result.recommendation]
    assert _metric(pub, "credit_model_auc") == summary["performance"]["auc"]
    assert _metric(pub, "credit_calibration_ratio") == summary["performance"]["calibration_ratio"]
    for attribute, impact in summary["fairness"].items():
        assert _metric(pub, "credit_fairness_disparate_impact", attribute=attribute) == impact
    assert summary["fairness_reference"].keys() == summary["fairness"].keys()
    for attribute, impact in summary["fairness_reference"].items():
        assert (
            _metric(pub, "credit_fairness_disparate_impact_reference", attribute=attribute)
            == impact
        )
    assert _metric(pub, "credit_model_auc_reference") > 0.5
    assert _metric(pub, "credit_pipeline_last_success_timestamp") > 1e9
    assert _metric(pub, "credit_pipeline_peak_memory_bytes") >= 0

    values = pub.values()
    psi_features = {dict(lbl)["feature"] for (name, lbl) in values if name == "credit_feature_psi"}
    ks_features = {dict(lbl)["feature"] for (name, lbl) in values if name == "credit_feature_ks"}
    table = {row["feature"]: row for row in summary["features"]}
    assert psi_features == set(table)
    assert ks_features == {f for f, row in table.items() if row["kind"] == "numeric"}
    assert "NAME_INCOME_TYPE" in psi_features - ks_features

    tracker = trackers[-1]
    assert tracker.experiment == "credit-monitoring" and tracker.run_name == "06"
    assert tracker.tags["batch_id"] == "06" and tracker.tags["status"] == "ok"
    assert tracker.tags["recommendation"] == result.recommendation
    assert {"model_version", "contract_version"} <= set(tracker.tags)
    names = {p.name for p in tracker.artifacts}
    assert {"06_summary.json", "06_data_drift.html", "06_data_drift.json", "manifest.json"} <= names
    assert "dataset_drift_share" in tracker.metrics


def test_blocked_batch_publishes_contract_metrics(trained, recorded):
    publishers, trackers = recorded
    result = run_monitor(trained, "03-dirty")
    assert result.status == "blocked"
    pub = publishers[-1]
    assert pub.pushes == 1
    assert _metric(pub, "credit_contract_passed") == 0
    rules = {dict(lbl)["rule"] for (name, lbl) in pub.values()
             if name == "credit_contract_violations"}
    expected = json.loads(
        (trained.reports_dir / "contracts/03-dirty_failures.json").read_text(encoding="utf-8")
    )["violations"]
    assert rules == set(expected) and rules
    assert _metric(pub, "credit_pipeline_stage_status", stage="contract") == -1
    stages = {dict(lbl)["stage"] for (name, lbl) in pub.values()
              if name == "credit_pipeline_stage_status"}
    assert stages == {"ingest", "contract"}  # predicao e drift nao rodam
    assert ("credit_model_auc", ()) not in pub.values()

    tracker = trackers[-1]
    assert tracker.tags["status"] == "blocked"
    assert "03-dirty_failures.json" in {p.name for p in tracker.artifacts}


# --- reset-metrics --------------------------------------------------------------------


def test_reset_metrics_deletes_every_group(trained, monkeypatch):
    from dataclasses import replace

    calls = []
    monkeypatch.setattr(
        pipeline, "delete_from_gateway", lambda url, job, grouping_key, timeout: calls.append(
            (url, job, grouping_key["batch"])
        )
    )
    settings = replace(trained, pushgateway_url="http://gw:9091")
    assert pipeline.reset_metrics(settings) == 8
    assert [c[2] for c in calls] == [*pipeline.SIMULATION_ORDER, "latest"]
    assert {c[:2] for c in calls} == {("http://gw:9091", "credit_pipeline")}


def test_reset_metrics_http_errors_are_counted_and_loop_continues(trained, monkeypatch):
    from dataclasses import replace

    attempts = []

    def flaky(url, job, grouping_key, timeout):
        attempts.append(grouping_key["batch"])
        if grouping_key["batch"] == "02":
            raise urllib.error.HTTPError(url, 500, "erro", None, None)

    monkeypatch.setattr(pipeline, "delete_from_gateway", flaky)
    report = pipeline.reset_groups(replace(trained, pushgateway_url="http://gw:9091"))
    assert attempts == [*pipeline.SIMULATION_ORDER, "latest"]  # seguiu ate o fim
    assert (report.deleted, report.failed, report.unreachable) == (7, 1, False)
    assert pipeline.reset_metrics(replace(trained, pushgateway_url="http://gw:9091")) == 7


@pytest.mark.parametrize("error", [
    lambda: urllib.error.URLError("sem rota"),
    lambda: TimeoutError("lento"),
    lambda: ConnectionRefusedError("recusada"),
])
def test_reset_metrics_stops_after_first_connection_failure(trained, monkeypatch, error):
    from dataclasses import replace

    attempts = []

    def boom(url, job, grouping_key, timeout):
        attempts.append(grouping_key["batch"])
        raise error()

    monkeypatch.setattr(pipeline, "delete_from_gateway", boom)
    report = pipeline.reset_groups(replace(trained, pushgateway_url="http://gw:9091"))
    assert len(attempts) == 1  # nao trava esperando timeout em cada grupo
    assert (report.deleted, report.failed, report.unreachable) == (0, 1, True)


def test_cli_reset_metrics_reports_abort_and_failures(trained, monkeypatch, capsys):
    monkeypatch.setenv("PUSHGATEWAY_URL", "http://gw:9091")

    def boom(*args, **kwargs):
        raise urllib.error.URLError("sem rota")

    monkeypatch.setattr(pipeline, "delete_from_gateway", boom)
    assert main(["reset-metrics"]) == 0
    out = capsys.readouterr().out
    assert "0 de 8" in out and "1 falha" in out and "gateway inacessível" in out


def test_reset_metrics_without_gateway_returns_zero(trained, capsys):
    assert pipeline.reset_metrics(trained) == 0  # trained.pushgateway_url is None


def test_simulate_all_waits_between_batches(trained, monkeypatch):
    monkeypatch.setattr(
        pipeline, "run_monitor", lambda s, b: pipeline.BatchResult(b, "ok", None, "OK")
    )
    sleeps = []
    monkeypatch.setattr(pipeline.time, "sleep", sleeps.append)
    results = run_simulate_all(trained, interval_seconds=1.5)
    assert len(results) == 7 and sleeps == [1.5] * 6


# --- --interval --------------------------------------------------------------------------


@pytest.mark.parametrize("text,expected", [("30", 30.0), ("30s", 30.0), ("0.5s", 0.5), ("0", 0.0)])
def test_parse_interval_accepts_seconds(text, expected):
    assert parse_interval(text) == expected


@pytest.mark.parametrize("text", ["abc", "-1", "5m", ""])
def test_parse_interval_rejects_garbage(text):
    import argparse

    with pytest.raises(argparse.ArgumentTypeError):
        parse_interval(text)


def test_cli_simulate_all_and_reset_return_zero(trained, monkeypatch, capsys):
    monkeypatch.setattr(pipeline, "run_simulate_all", lambda s, interval_seconds=0.0: [
        pipeline.BatchResult("01", "ok", None, "OK"),
        pipeline.BatchResult("03-dirty", "blocked", None, None),
    ])
    monkeypatch.setenv("DATA_DIR", str(trained.data_dir))
    monkeypatch.setenv("REPORTS_DIR", str(trained.reports_dir))
    monkeypatch.setenv("MODELS_DIR", str(trained.models_dir))
    monkeypatch.delenv("PUSHGATEWAY_URL", raising=False)
    assert main(["simulate-all", "--interval", "30s"]) == 0
    assert "bloqueado" in capsys.readouterr().out
    assert main(["reset-metrics"]) == 0


# --- indice HTML ---------------------------------------------------------------------------


def _write_summary(reports, batch_id, **over):
    drift = reports / "drift"
    drift.mkdir(parents=True, exist_ok=True)
    summary = {
        "batch_id": batch_id,
        "drifted_features": ["AMT_CREDIT", "AGE_YEARS"],
        "dataset_drift_share": 0.4,
        "performance": {"auc": 0.71, "calibration_ratio": 1.234},
        "recommendation": "INVESTIGAR",
        "evidently": {"data_drift": {"html": f"{batch_id}_data_drift.html",
                                     "json": f"{batch_id}_data_drift.json"}},
        **over,
    }
    (drift / f"{batch_id}_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    (drift / f"{batch_id}_data_drift.html").write_text("<html></html>", encoding="utf-8")


def _write_failures(reports, batch_id):
    contracts = reports / "contracts"
    contracts.mkdir(parents=True, exist_ok=True)
    (contracts / f"{batch_id}_failures.json").write_text(
        json.dumps({"batch_id": batch_id, "passed": False, "violations": {"age_range": 3}}),
        encoding="utf-8",
    )
    (contracts / f"{batch_id}_failures.html").write_text("<html></html>", encoding="utf-8")


def test_build_index_rows_in_simulation_order(tmp_path):
    _write_summary(tmp_path, "06")
    _write_summary(tmp_path, "01", drifted_features=[], dataset_drift_share=0.0,
                   recommendation="OK")
    _write_failures(tmp_path, "03-dirty")
    (tmp_path / "contracts" / "raw_reference_failures.json").write_text("{}", encoding="utf-8")

    path = build_index(tmp_path)
    assert path == tmp_path / "index.html"
    text = path.read_text(encoding="utf-8")
    assert text.index("<td>01</td>") < text.index("<td>03-dirty</td>") < text.index("<td>06</td>")
    assert text.count("<tr class=") == 3  # sem linha para a referencia bruta
    assert "BLOQUEADO" in text and "age_range" in text
    assert "drift/06_data_drift.html" in text and "contracts/03-dirty_failures.html" in text
    assert "40%" in text and "0,71" in text and "1,23" in text
    assert "INVESTIGAR" in text
    assert "http" not in text  # autocontido: sem recursos externos


def test_build_index_latest_run_wins_and_empty_dir_works(tmp_path):
    import os

    assert "Nenhum lote" in build_index(tmp_path).read_text(encoding="utf-8")
    _write_failures(tmp_path, "03")
    _write_summary(tmp_path, "03", recommendation="OK", drifted_features=[])
    failures = tmp_path / "contracts" / "03_failures.json"
    os.utime(failures, (1, 1))  # a falha e antiga: o lote foi reprocessado com sucesso
    text = build_index(tmp_path).read_text(encoding="utf-8")
    assert "BLOQUEADO" not in text.split("<tbody>")[1]


def test_build_index_handles_unlabeled_batch_and_escapes(tmp_path):
    _write_summary(tmp_path, "02", performance=None, recommendation="<b>OK</b>")
    text = build_index(tmp_path).read_text(encoding="utf-8")
    assert "<b>OK</b>" not in text and "&lt;b&gt;OK&lt;/b&gt;" in text


# --- caminho de erro no meio da esteira ----------------------------------------------------


@pytest.mark.parametrize(
    "target,stage_name", [("build_features", "transform"), ("score", "predict")]
)
def test_unexpected_error_publishes_stage_status_zero_and_propagates(
    trained, recorded, monkeypatch, target, stage_name
):
    publishers, trackers = recorded

    def boom(*args, **kwargs):
        raise RuntimeError("defeito inesperado")

    monkeypatch.setattr(pipeline, target, boom)
    with pytest.raises(RuntimeError, match="defeito inesperado"):
        run_monitor(trained, "01")
    pub = publishers[-1]
    assert pub.pushes == 1
    assert _metric(pub, "credit_pipeline_stage_status", stage=stage_name) == 0
    assert _metric(pub, "credit_pipeline_stage_status", stage="contract") == 1
    assert ("credit_pipeline_last_success_timestamp", ()) not in pub.values()
    assert ("credit_monitor_recommendation", ()) not in pub.values()


def test_usage_errors_do_not_push(trained, recorded, tmp_path):
    from dataclasses import replace

    from credit_monitor.model.predict import ModelNotTrained

    publishers, _ = recorded
    with pytest.raises(BatchNotFound):
        run_monitor(trained, "99")
    assert publishers[-1].pushes == 0
    with pytest.raises(ModelNotTrained):
        run_monitor(replace(trained, models_dir=tmp_path / "vazio"), "01")
    assert publishers[-1].pushes == 0


# --- trecho de log como artefato do MLflow --------------------------------------------------


@pytest.mark.parametrize("batch_id", ["06", "03-dirty"])
def test_run_log_excerpt_artifact_has_only_this_run(trained, recorded, batch_id):
    _, trackers = recorded
    run_monitor(trained, "01")  # outra execucao no mesmo pipeline.jsonl
    run_monitor(trained, batch_id)
    artifacts = {p.name: p for p in trackers[-1].artifacts}
    excerpt = artifacts[f"{batch_id}_run.jsonl"]
    assert excerpt == trained.reports_dir / "logs" / f"{batch_id}_run.jsonl"
    entries = [json.loads(line) for line in excerpt.read_text(encoding="utf-8").splitlines()]
    assert entries and len({e["run_id"] for e in entries}) == 1
    assert entries[0]["run_id"].startswith(f"{batch_id}-")
    assert {e["batch_id"] for e in entries} == {batch_id}
    full = (trained.reports_dir / "logs" / "pipeline.jsonl").read_text(encoding="utf-8")
    assert len(full.splitlines()) > len(entries)
