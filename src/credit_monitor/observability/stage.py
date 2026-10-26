"""Instrumentacao de etapas: duracao, status, linhas, metricas e log."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass

from credit_monitor.observability.metrics import MetricsPublisher

STATUS_SUCCESS = 1
STATUS_ERROR = 0
STATUS_BLOCKED = -1


@dataclass
class StageContext:
    rows_in: int = 0
    rows_out: int = 0
    blocked_reason: str | None = None

    def block(self, reason: str) -> None:
        """Marca a etapa como bloqueada (ex.: contrato violado); nao e um erro."""
        self.blocked_reason = reason


@contextmanager
def stage(
    name: str,
    batch_id: str,
    publisher: MetricsPublisher,
    logger: logging.LoggerAdapter,
) -> Iterator[StageContext]:
    ctx = StageContext()
    start = time.monotonic()
    error: BaseException | None = None
    try:
        yield ctx
    except BaseException as exc:
        error = exc
        raise
    finally:
        duration = time.monotonic() - start
        if error is not None:
            code, label, level = STATUS_ERROR, "error", logging.ERROR
            # So o tipo da excecao: a mensagem poderia carregar dados de linha.
            message = f"Etapa {name} falhou: {type(error).__name__}"
        elif ctx.blocked_reason is not None:
            code, label, level = STATUS_BLOCKED, "blocked", logging.WARNING
            message = f"Etapa {name} bloqueada: {ctx.blocked_reason}"
        else:
            code, label, level = STATUS_SUCCESS, "success", logging.INFO
            message = f"Etapa {name} concluida"
        publisher.set("credit_pipeline_stage_status", code, stage=name)
        publisher.set("credit_pipeline_stage_duration_seconds", duration, stage=name)
        publisher.set("credit_pipeline_rows_processed", ctx.rows_out, stage=name)
        logger.log(
            level,
            message,
            extra={
                "batch_id": batch_id,
                "stage": name,
                "event": "stage_end",
                "status": label,
                "duration_ms": round(duration * 1000, 1),
                "rows_in": ctx.rows_in,
                "rows_out": ctx.rows_out,
            },
        )


def peak_memory_bytes() -> int:
    """Pico de memoria residente do processo, em bytes (0 se `resource` nao existir)."""
    try:
        import resource
    except ImportError:
        return 0
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # Linux reporta KiB; macOS reporta bytes.
    return peak * 1024 if sys.platform.startswith("linux") else peak
