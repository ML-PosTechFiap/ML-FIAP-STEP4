import json
import time

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
