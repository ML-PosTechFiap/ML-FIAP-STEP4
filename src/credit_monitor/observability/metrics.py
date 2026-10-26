"""Metricas Prometheus publicadas no Pushgateway (job credit_pipeline)."""

from __future__ import annotations

import logging
from typing import Any

from prometheus_client import CollectorRegistry, Gauge, push_to_gateway

from credit_monitor.config import Settings

JOB = "credit_pipeline"
LATEST = "latest"

# nome completo -> (descricao, labels). O label `batch` NAO entra aqui: vem da grouping key.
METRIC_CATALOG: dict[str, tuple[str, list[str]]] = {
    # Pipeline
    "credit_pipeline_stage_status": (
        "Status da etapa: 1 sucesso, 0 erro, -1 bloqueado pelo contrato",
        ["stage"],
    ),
    "credit_pipeline_stage_duration_seconds": ("Duracao da etapa em segundos", ["stage"]),
    "credit_pipeline_rows_processed": ("Linhas processadas (saida) pela etapa", ["stage"]),
    "credit_pipeline_last_success_timestamp": (
        "Timestamp Unix da ultima execucao concluida com sucesso",
        [],
    ),
    "credit_pipeline_peak_memory_bytes": ("Memoria de pico do pipeline em bytes", []),
    "credit_batch_info": ("Lote publicado no grupo latest (valor sempre 1)", ["batch_id"]),
    # Qualidade
    "credit_contract_passed": ("Contrato de dados: 1 passou, 0 violado", []),
    "credit_contract_violations": ("Violacoes do contrato de dados por regra", ["rule"]),
    # Drift
    "credit_feature_psi": ("PSI da feature vs referencia", ["feature"]),
    "credit_feature_ks": ("Estatistica KS da feature vs referencia", ["feature"]),
    "credit_dataset_drift_share": ("Fracao de features com drift", []),
    "credit_prediction_psi": ("PSI do score vs referencia", []),
    # Modelo
    "credit_model_auc": ("AUC do modelo no lote", []),
    "credit_model_ks": ("KS do modelo no lote", []),
    "credit_default_rate_observed": ("Taxa de inadimplencia observada no lote", []),
    "credit_mean_predicted_pd": ("PD media prevista no lote", []),
    "credit_calibration_ratio": ("Razao entre PD media prevista e inadimplencia observada", []),
    "credit_approval_rate": ("Taxa de aprovacao no lote", []),
    "credit_monitor_recommendation": (
        "Recomendacao do monitor: 0 OK, 1 atencao, 2 retreinar",
        [],
    ),
    "credit_model_auc_reference": ("AUC do modelo na base de referencia", []),
    "credit_model_ks_reference": ("KS do modelo na base de referencia", []),
    # Vies
    "credit_fairness_disparate_impact": (
        "Impacto disparato da aprovacao por atributo",
        ["attribute"],
    ),
}

ValuesKey = tuple[str, tuple[tuple[str, str], ...]]

_log = logging.getLogger("credit_monitor.metrics")


class MetricsPublisher:
    """Acumula gauges num registry proprio e os publica nos grupos `batch_id` e `latest`."""

    def __init__(
        self,
        settings: Settings,
        batch_id: str,
        logger: logging.Logger | logging.LoggerAdapter | None = None,
    ) -> None:
        self.settings = settings
        self._log = logger or _log
        self.batch_id = batch_id
        self.registry = CollectorRegistry()
        self._gauges: dict[str, Gauge] = {}
        self._values: dict[ValuesKey, float] = {}

    def set(self, name: str, value: float, **labels: Any) -> None:
        if name not in METRIC_CATALOG:
            raise KeyError(f"Metrica fora do catalogo: {name}")
        description, label_names = METRIC_CATALOG[name]
        if set(labels) != set(label_names):
            raise ValueError(
                f"Labels de {name} devem ser {sorted(label_names)}, recebidos {sorted(labels)}"
            )
        if name not in self._gauges:
            self._gauges[name] = Gauge(name, description, label_names, registry=self.registry)
        str_labels = {key: str(val) for key, val in labels.items()}
        gauge = self._gauges[name]
        (gauge.labels(**str_labels) if label_names else gauge).set(value)
        self._values[(name, tuple(sorted(str_labels.items())))] = float(value)

    def values(self) -> dict[ValuesKey, float]:
        """Snapshot `{(nome, ((label, valor), ...)): valor}` para o MLflow e os testes."""
        return dict(self._values)

    def push(self) -> bool:
        """PUT nos grupos `batch_id` e `latest`. Falha de infraestrutura vira False + WARN."""
        url = self.settings.pushgateway_url
        if not url:
            self._log.warning(
                "Pushgateway nao configurado; metricas nao foram publicadas.",
                extra={"batch_id": self.batch_id},
            )
            return False
        self.set("credit_batch_info", 1, batch_id=self.batch_id)
        try:
            for group in (self.batch_id, LATEST):
                push_to_gateway(
                    url,
                    job=JOB,
                    registry=self.registry,
                    grouping_key={"batch": group},
                    timeout=self.settings.http_timeout,
                )
        except Exception as exc:  # metricas nunca derrubam o pipeline
            self._log.warning(
                "Falha ao publicar no Pushgateway (%s): %s",
                type(exc).__name__,
                exc,
                extra={"batch_id": self.batch_id},
            )
            return False
        return True
