"""Esteira mensal: ingestao, contrato, features, predicao, drift e publicacao (spec 3.2)."""

from __future__ import annotations

import json
import logging
import re
import time
import urllib.error
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import pandas as pd
from prometheus_client import delete_from_gateway

from credit_monitor.config import SIMULATION_ORDER, Settings
from credit_monitor.contracts.schema import CONTRACT_VERSION
from credit_monitor.contracts.validate import quarantine, validate_batch, write_failure_report
from credit_monitor.drift.detect import analyze_batch
from credit_monitor.model.features import build_features
from credit_monitor.model.predict import ModelNotTrained, load_model, score
from credit_monitor.observability.logging import get_logger
from credit_monitor.observability.metrics import JOB, LATEST, MetricsPublisher
from credit_monitor.observability.report_index import build_index
from credit_monitor.observability.stage import peak_memory_bytes, stage
from credit_monitor.observability.tracking import Tracker
from credit_monitor.privacy.pseudonymize import KEY_COLUMN, pseudonymize

EXPERIMENT = "credit-monitoring"
RECOMMENDATION_CODES = {"OK": 0, "INVESTIGAR": 1, "RETREINAR": 2}

_log = logging.getLogger("credit_monitor.pipeline")


class BatchNotFound(Exception):
    """O arquivo do lote nao existe em `data/batches/`."""


@dataclass
class BatchResult:
    batch_id: str
    status: Literal["ok", "blocked"]
    summary_path: Path | None
    recommendation: str | None


def _batch_dir(settings: Settings, batch_id: str) -> Path:
    return settings.data_dir / "batches" / batch_id


def _set_number(publisher: MetricsPublisher, name: str, value, **labels) -> None:
    """Publica `value` se for um numero (None, vindo de NaN/rotulo ausente, e ignorado)."""
    if value is not None:
        publisher.set(name, value, **labels)


def _publish_summary(publisher: MetricsPublisher, summary: dict, meta: dict) -> None:
    """Metricas de drift, modelo e vies (spec 7.3) a partir do resumo do lote."""
    for row in summary["features"]:
        _set_number(publisher, "credit_feature_psi", row["psi"], feature=row["feature"])
        if row["kind"] == "numeric":
            _set_number(publisher, "credit_feature_ks", row["ks"], feature=row["feature"])
    _set_number(publisher, "credit_dataset_drift_share", summary["dataset_drift_share"])
    _set_number(publisher, "credit_prediction_psi", summary["prediction_psi"])

    performance = summary.get("performance")
    if performance:
        for key, metric in [
            ("auc", "credit_model_auc"),
            ("ks", "credit_model_ks"),
            ("default_rate_observed", "credit_default_rate_observed"),
            ("mean_predicted_pd", "credit_mean_predicted_pd"),
            ("calibration_ratio", "credit_calibration_ratio"),
            ("approval_rate", "credit_approval_rate"),
        ]:
            _set_number(publisher, metric, performance.get(key))
    for attribute, impact in (summary.get("fairness") or {}).items():
        _set_number(publisher, "credit_fairness_disparate_impact", impact, attribute=attribute)
    for attribute, impact in (summary.get("fairness_reference") or {}).items():
        _set_number(
            publisher, "credit_fairness_disparate_impact_reference", impact, attribute=attribute
        )

    publisher.set("credit_monitor_recommendation", RECOMMENDATION_CODES[summary["recommendation"]])
    _publish_references(publisher, meta)


def _publish_references(publisher: MetricsPublisher, meta: dict) -> None:
    """Series de referencia do modelo (AUC, KS e impacto disparato) a partir do metadata.

    No caminho bloqueado nao ha resumo do lote, mas os paineis "vs referencia" do dashboard
    leem estas series em `batch="latest"` e nao devem ficar vazios.
    """
    publisher.set("credit_model_auc_reference", meta["reference_metrics"]["auc"])
    publisher.set("credit_model_ks_reference", meta["reference_metrics"]["ks"])
    for attribute, report in (meta.get("fairness") or {}).items():
        if isinstance(report, dict) and "disparate_impact" in report:
            _set_number(publisher, "credit_fairness_disparate_impact_reference",
                        report["disparate_impact"], attribute=attribute)


def _mlflow_metrics(publisher: MetricsPublisher, summary: dict | None) -> dict[str, float]:
    """Mesmas metricas do Pushgateway (sem o prefixo `credit_`), com labels no sufixo."""
    metrics = {}
    for (name, labels), value in publisher.values().items():
        key = name.removeprefix("credit_")
        if labels:
            key += "." + ".".join(val for _, val in labels)
        metrics[re.sub(r"[^\w\-. /]", "_", key)] = value  # o MLflow rejeita outros caracteres
    raw = ((summary or {}).get("performance") or {}).get("calibration_ratio_raw")
    if raw is not None:
        metrics["calibration_ratio_raw"] = raw
    return metrics


def _track(
    tracker: Tracker,
    batch_id: str,
    tags: dict[str, str],
    publisher: MetricsPublisher,
    summary: dict | None,
    artifacts: list[Path],
) -> None:
    with tracker.run(EXPERIMENT, batch_id, tags):
        tracker.log_metrics(_mlflow_metrics(publisher, summary))
        for path in artifacts:
            if path.exists():
                tracker.log_artifact(path)


def _run_log_excerpt(settings: Settings, batch_id: str, run_id: str) -> Path:
    """`reports/logs/<lote>_run.jsonl`: so as linhas de `pipeline.jsonl` desta execucao."""
    source = settings.reports_dir / "logs" / "pipeline.jsonl"
    lines = []
    if source.exists():
        for line in source.read_text(encoding="utf-8").splitlines():
            try:
                if json.loads(line).get("run_id") == run_id:
                    lines.append(line)
            except ValueError:
                continue
    out = settings.reports_dir / "logs" / f"{batch_id}_run.jsonl"
    out.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out


def _evidently_files(summary: dict, drift_dir: Path) -> list[Path]:
    files = []
    for entry in (summary.get("evidently") or {}).values():
        files += [drift_dir / entry["html"], drift_dir / entry["json"]]
    return files


def _block(
    settings: Settings,
    batch_id: str,
    result,
    ctx,
    publisher: MetricsPublisher,
    log: logging.LoggerAdapter,
    batch_file: Path,
) -> list[Path]:
    """Efeitos do bloqueio: relatorio de falhas, quarentena (copia) e metricas do contrato."""
    rules = ", ".join(sorted(result.violations, key=lambda r: -result.violations[r]))
    ctx.block(f"contrato de dados violado nas regras: {rules}")  # so nomes de regra, nunca valores
    contracts_dir = settings.reports_dir / "contracts"
    json_path, html_path = write_failure_report(result, batch_id, contracts_dir)
    quarantine(batch_file, batch_id, settings)
    publisher.set("credit_contract_passed", 0)
    for rule, count in result.violations.items():
        publisher.set("credit_contract_violations", count, rule=rule)
    log.error(
        f"Lote bloqueado pelo contrato de dados (regras: {rules}); copiado para a quarentena.",
        extra={"batch_id": batch_id, "stage": "contract", "event": "batch_blocked",
               "status": "blocked", "rows_in": result.n_rows},
    )
    return [json_path, html_path]


def _run(
    settings: Settings, batch_id: str, run_id: str, log, publisher: MetricsPublisher
) -> BatchResult:
    model, meta = load_model(settings)  # antes da ingestao: ModelNotTrained sobe sem tocar no lote
    batch_file = _batch_dir(settings, batch_id) / "batch.parquet"

    with stage("ingest", batch_id, publisher, log) as ctx:
        if not batch_file.exists():
            raise BatchNotFound(
                f"Lote {batch_id} não encontrado em {batch_file}. "
                "Execute `prepare` para gerar os lotes."
            )
        raw = pd.read_parquet(batch_file)
        df = pseudonymize(raw, settings.pseudonymization_salt)
        del raw
        ctx.rows_in = ctx.rows_out = len(df)

    domains = json.loads(
        (settings.data_dir / "reference" / "domains.json").read_text(encoding="utf-8")
    )
    failure_files: list[Path] = []
    with stage("contract", batch_id, publisher, log) as ctx:
        ctx.rows_in = len(df)
        # `validate_batch` exige indice unico, e o lote bloqueado pode ter IDs duplicados
        validation = validate_batch(df.reset_index(drop=True), domains)
        if validation.passed:
            ctx.rows_out = len(df)
            publisher.set("credit_contract_passed", 1)
        else:
            failure_files = _block(settings, batch_id, validation, ctx, publisher, log, batch_file)

    tracker = Tracker(settings, logger=log)
    tags = {
        "batch_id": batch_id,
        "model_version": str(meta.get("mlflow_model_version") or "local"),
        "contract_version": str(meta.get("contract_version") or CONTRACT_VERSION),
    }
    if not validation.passed:
        publisher.set("credit_pipeline_peak_memory_bytes", peak_memory_bytes())
        _publish_references(publisher, meta)
        _track(tracker, batch_id, {**tags, "status": "blocked", "recommendation": "n/a"},
               publisher, None, [*failure_files, _run_log_excerpt(settings, batch_id, run_id)])
        build_index(settings.reports_dir)
        publisher.push()
        return BatchResult(batch_id, "blocked", None, None)

    with stage("transform", batch_id, publisher, log) as ctx:
        ctx.rows_in = len(df)
        scored = build_features(df).drop(columns=[KEY_COLUMN])  # o scoring nao precisa da chave
        ctx.rows_out = len(scored)

    with stage("predict", batch_id, publisher, log) as ctx:
        ctx.rows_in = len(scored)
        scored["score"] = score(model, scored)
        ctx.rows_out = len(scored)

    drift_dir = settings.reports_dir / "drift"
    with stage("drift", batch_id, publisher, log) as ctx:
        reference = pd.read_parquet(settings.data_dir / "reference" / "test_scored.parquet")
        ctx.rows_in = len(scored)
        summary = analyze_batch(reference, scored, batch_id, meta, drift_dir)
        ctx.rows_out = len(summary["features"])

    with stage("publish", batch_id, publisher, log) as ctx:
        _publish_summary(publisher, summary, meta)
        publisher.set("credit_pipeline_peak_memory_bytes", peak_memory_bytes())
        ctx.rows_in = ctx.rows_out = len(publisher.values())
        artifacts = [Path(summary["summary_path"]), *_evidently_files(summary, drift_dir)]
        artifacts.append(_batch_dir(settings, batch_id) / "manifest.json")
        artifacts.append(_run_log_excerpt(settings, batch_id, run_id))
        _track(tracker, batch_id,
               {**tags, "status": "ok", "recommendation": summary["recommendation"]},
               publisher, summary, artifacts)
        build_index(settings.reports_dir)
    # So depois de tudo ter dado certo: um push de erro nunca carrega este valor.
    publisher.set("credit_pipeline_last_success_timestamp", time.time())
    publisher.push()  # publicacao unica, ja com as metricas da propria etapa publish

    log.info(
        f"Lote {batch_id} processado: recomendação {summary['recommendation']}.",
        extra={"batch_id": batch_id, "event": "batch_end", "status": "ok",
               "rows_in": len(scored)},
    )
    return BatchResult(batch_id, "ok", Path(summary["summary_path"]), summary["recommendation"])


def run_monitor(settings: Settings, batch_id: str) -> BatchResult:
    """Roda a esteira mensal de um lote. Levanta `ModelNotTrained` e `BatchNotFound`."""
    run_id = f"{batch_id}-{uuid.uuid4().hex[:8]}"
    log = get_logger(settings, run_id)
    publisher = MetricsPublisher(settings, batch_id, logger=log)
    try:
        return _run(settings, batch_id, run_id, log, publisher)
    except Exception as exc:
        # Erro inesperado: publica o status 0 da etapa para o alerta PipelineStageFailed.
        # Lote inexistente ou modelo ausente sao erros de uso, nao falhas da esteira.
        if not isinstance(exc, BatchNotFound | ModelNotTrained):
            publisher.push()
        raise


def run_simulate_all(settings: Settings, interval_seconds: float = 0.0) -> list[BatchResult]:
    """Roda `SIMULATION_ORDER` em sequencia, com pausa entre lotes. Lotes bloqueados nao param."""
    results = []
    for position, batch_id in enumerate(SIMULATION_ORDER):
        if position and interval_seconds > 0:
            time.sleep(interval_seconds)
        results.append(run_monitor(settings, batch_id))
    return results


@dataclass
class ResetReport:
    deleted: int
    failed: int
    unreachable: bool  # gateway inacessivel: o laco foi interrompido
    configured: bool = True


def _gateway_unreachable(exc: Exception) -> bool:
    """Falha de conexao ou timeout (nao uma resposta HTTP de erro de um grupo)."""
    if isinstance(exc, urllib.error.HTTPError):
        return False
    return isinstance(exc, urllib.error.URLError | TimeoutError | ConnectionError)


def reset_groups(settings: Settings) -> ResetReport:
    """Apaga os grupos de cada lote e `latest`, contando falhas sem levantar.

    Erro HTTP de um grupo e contado e o laco segue. A primeira falha de conexao (gateway
    inacessivel) interrompe o laco, para nao travar a demo esperando timeouts.
    """
    url = settings.pushgateway_url
    if not url:
        _log.warning("Pushgateway não configurado (PUSHGATEWAY_URL); nada a apagar.")
        return ResetReport(0, 0, False, configured=False)
    deleted = failed = 0
    unreachable = False
    for group in [*SIMULATION_ORDER, LATEST]:
        try:
            delete_from_gateway(
                url, job=JOB, grouping_key={"batch": group}, timeout=settings.http_timeout
            )
        except Exception as exc:
            failed += 1
            unreachable = _gateway_unreachable(exc)
            _log.warning(
                "Falha ao apagar o grupo %s no Pushgateway (%s).", group, type(exc).__name__
            )
            if unreachable:
                break
            continue
        deleted += 1
    return ResetReport(deleted, failed, unreachable)


def reset_metrics(settings: Settings) -> int:
    """Apaga do Pushgateway os grupos de cada lote e `latest`. Devolve quantos foram apagados."""
    return reset_groups(settings).deleted
