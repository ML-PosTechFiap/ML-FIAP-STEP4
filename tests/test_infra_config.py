"""Consistencia entre o codigo (METRIC_CATALOG) e a configuracao de infraestrutura."""

import json
import re
from pathlib import Path

import yaml

from credit_monitor.observability.metrics import JOB, METRIC_CATALOG

INFRA = Path(__file__).resolve().parents[1] / "infra"

ALERT_NAMES = {
    "DataContractViolation",
    "PipelineStageFailed",
    "FeatureDriftDetected",
    "DatasetDriftDetected",
    "PredictionDrift",
    "ModelPerformanceDegraded",
    "ModelMiscalibrated",
    "FairnessDegraded",
    "InfraTargetDown",
}


def _metric_names(text: str) -> set[str]:
    # `credit_pipeline` tambem e o valor do label job (seletores LogQL/PromQL), nao uma metrica:
    # o regex o capturaria por engano, entao o token exato e ignorado.
    return set(re.findall(r"\bcredit_[a-z_]+", text)) - {JOB}


def _alert_rules() -> list[dict]:
    rules = yaml.safe_load((INFRA / "prometheus/alerts.yml").read_text(encoding="utf-8"))
    return [r for g in rules["groups"] for r in g["rules"]]


def test_alerts_reference_only_cataloged_metrics():
    rules = _alert_rules()
    assert {r["alert"] for r in rules} == ALERT_NAMES
    exprs = " ".join(r["expr"] for r in rules)
    assert _metric_names(exprs) <= set(METRIC_CATALOG)


def test_alerts_have_severity_summary_and_latest_selector():
    for rule in _alert_rules():
        assert rule["labels"]["severity"] in {"warning", "critical"}
        assert rule["annotations"]["summary"]
        if rule["alert"] == "InfraTargetDown":
            assert rule["for"] == "1m"
        else:
            assert rule["for"] == "0m"
            assert 'batch="latest"' in rule["expr"]


def test_dashboard_references_only_cataloged_metrics():
    text = (INFRA / "grafana/dashboards/credit-monitoring.json").read_text(encoding="utf-8")
    dashboard = json.loads(text)
    assert dashboard["uid"] == "credit-monitoring"
    assert _metric_names(text) <= set(METRIC_CATALOG)


def test_dashboard_datasources_are_provisioned_uids():
    dashboard = json.loads(
        (INFRA / "grafana/dashboards/credit-monitoring.json").read_text(encoding="utf-8")
    )
    provisioned = yaml.safe_load(
        (INFRA / "grafana/provisioning/datasources/datasources.yml").read_text(encoding="utf-8")
    )
    uids = {ds["uid"] for ds in provisioned["datasources"]}
    assert uids == {"prometheus", "loki"}

    def walk(node):
        if isinstance(node, dict):
            if "uid" in node and "type" in node and node["type"] in {"prometheus", "loki"}:
                yield node["uid"]
            for value in node.values():
                yield from walk(value)
        elif isinstance(node, list):
            for value in node:
                yield from walk(value)

    used = set(walk(dashboard))
    assert used and used <= uids


def test_prometheus_scrapes_all_services_and_loads_alerts():
    config = yaml.safe_load((INFRA / "prometheus/prometheus.yml").read_text(encoding="utf-8"))
    assert config["rule_files"] == ["alerts.yml"]
    jobs = {job["job_name"]: job for job in config["scrape_configs"]}
    assert set(jobs) == {"pushgateway", "prometheus", "mlflow", "grafana", "loki"}
    assert jobs["pushgateway"]["honor_labels"] is True
