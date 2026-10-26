"""Logs estruturados: JSON por linha em stdout, arquivo e Loki. Nunca contem dados de linha."""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import UTC, datetime
from typing import Any

import requests

from credit_monitor.config import Settings

LOG_FIELDS = (
    "run_id",
    "batch_id",
    "stage",
    "event",
    "status",
    "duration_ms",
    "rows_in",
    "rows_out",
)
LOKI_PUSH_PATH = "/loki/api/v1/push"


class JsonFormatter(logging.Formatter):
    """Uma linha JSON por registro; campos ausentes viram null."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": datetime.fromtimestamp(record.created, tz=UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
        }
        for field in LOG_FIELDS:
            payload[field] = getattr(record, field, None)
        payload["message"] = record.getMessage()
        return json.dumps(payload, ensure_ascii=False, default=str)


class LokiHandler(logging.Handler):
    """POST sincrono no Loki. Desativa-se na primeira falha para nunca travar o pipeline."""

    def __init__(self, url: str, timeout: float) -> None:
        super().__init__()
        self.endpoint = url.rstrip("/") + LOKI_PUSH_PATH
        self.timeout = timeout
        self.disabled = False

    def emit(self, record: logging.LogRecord) -> None:
        if self.disabled:
            return
        labels = {
            "job": "credit_pipeline",
            "stage": str(getattr(record, "stage", None) or "none"),
            "level": record.levelname,
            "batch": str(getattr(record, "batch_id", None) or "none"),
        }
        body = {
            "streams": [
                {"stream": labels, "values": [[str(time.time_ns()), self.format(record)]]}
            ]
        }
        try:
            response = requests.post(self.endpoint, json=body, timeout=self.timeout)
            response.raise_for_status()
        except Exception as exc:  # qualquer falha desativa o handler
            self.disabled = True
            print(
                f"AVISO: Loki indisponivel ({type(exc).__name__}); "
                "envio de logs ao Loki desativado.",
                file=sys.stderr,
            )


class _RunLogger(logging.LoggerAdapter):
    """Adapter que mescla o run_id com o `extra=` de cada chamada."""

    def process(self, msg: Any, kwargs: Any) -> tuple[Any, Any]:
        kwargs["extra"] = {**(self.extra or {}), **(kwargs.get("extra") or {})}
        return msg, kwargs


def get_logger(settings: Settings, run_id: str) -> logging.LoggerAdapter:
    """Logger JSON com handlers de stdout, `reports/logs/pipeline.jsonl` e (opcional) Loki.

    O logger e nomeado por reports_dir e run_id, e seus handlers sao recriados a cada chamada,
    entao chamar duas vezes nunca duplica linhas.
    """
    logger = logging.getLogger(f"credit_pipeline.{settings.reports_dir}.{run_id}")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    for old in list(logger.handlers):
        logger.removeHandler(old)
        old.close()

    formatter = JsonFormatter()
    handlers: list[logging.Handler] = [logging.StreamHandler(sys.stdout)]

    log_dir = settings.reports_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    handlers.append(logging.FileHandler(log_dir / "pipeline.jsonl", encoding="utf-8"))

    if settings.loki_url:
        handlers.append(LokiHandler(settings.loki_url, settings.http_timeout))

    for handler in handlers:
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return _RunLogger(logger, {"run_id": run_id})
