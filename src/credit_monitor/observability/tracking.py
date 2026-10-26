"""Rastreamento no MLflow. Sem servidor alcancavel, todas as operacoes viram no-op."""

from __future__ import annotations

import logging
import math
import os
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import requests

from credit_monitor.config import Settings

_log = logging.getLogger("credit_monitor.tracking")


class Tracker:
    def __init__(
        self, settings: Settings, logger: logging.Logger | logging.LoggerAdapter | None = None
    ) -> None:
        self.settings = settings
        self._log = logger or _log
        self.uri = settings.mlflow_uri.rstrip("/") if settings.mlflow_uri else None
        self.active_run_id: str | None = None
        if self.uri:
            # O cliente REST do MLflow nao usa o nosso timeout: sem isto o padrao e 120 s x 7 tentativas.
            os.environ["MLFLOW_HTTP_REQUEST_TIMEOUT"] = str(max(1, math.ceil(settings.http_timeout)))
            os.environ["MLFLOW_HTTP_REQUEST_MAX_RETRIES"] = "0"
        self.enabled = self._probe()

    def _probe(self) -> bool:
        if not self.uri:
            return False
        try:
            response = requests.get(f"{self.uri}/health", timeout=self.settings.http_timeout)
            response.raise_for_status()
        except Exception as exc:
            self._log.warning("MLflow indisponivel (%s); rastreamento desativado.", type(exc).__name__)
            return False
        return True

    def _disable(self, exc: Exception) -> None:
        self.enabled = False
        self._log.warning("Falha no MLflow (%s); rastreamento desativado.", type(exc).__name__)

    @contextmanager
    def run(
        self, experiment: str, run_name: str, tags: dict[str, str] | None = None
    ) -> Iterator[str | None]:
        """Abre um run; produz o run_id (ou None se desabilitado)."""
        if not self.enabled:
            yield None
            return
        try:
            import mlflow

            mlflow.set_tracking_uri(self.uri)
            mlflow.set_experiment(experiment)
            active = mlflow.start_run(run_name=run_name, tags=tags or {})
        except Exception as exc:
            self._disable(exc)
            yield None
            return
        self.active_run_id = active.info.run_id
        status = "FINISHED"
        try:
            yield self.active_run_id
        except BaseException:
            status = "FAILED"
            raise
        finally:
            self.active_run_id = None
            if self.enabled:  # desabilitado no meio do run: nao bater no servidor morto
                try:
                    mlflow.end_run(status=status)
                except Exception as exc:
                    self._disable(exc)

    def _call(self, fn_name: str, *args: Any) -> None:
        if not self.enabled or self.active_run_id is None:
            return
        try:
            import mlflow

            getattr(mlflow, fn_name)(*args)
        except Exception as exc:
            self._disable(exc)

    def log_metrics(self, d: dict[str, float]) -> None:
        self._call("log_metrics", d)

    def log_params(self, d: dict[str, Any]) -> None:
        self._call("log_params", d)

    def log_artifact(self, path: Path) -> None:
        self._call("log_artifact", str(path))

    def log_model(self, pipeline: Any, registered_name: str) -> str | None:
        """Registra o modelo no run ativo e aponta o alias `production` para a nova versao.

        Devolve a versao (str) ou `None` se desabilitado ou se o MLflow falhar.
        """
        if not self.enabled or self.active_run_id is None:
            return None
        try:
            import mlflow
            import mlflow.sklearn

            info = mlflow.sklearn.log_model(
                pipeline, "model", registered_model_name=registered_name
            )
            version = str(info.registered_model_version)
            mlflow.MlflowClient().set_registered_model_alias(
                registered_name, "production", version
            )
        except Exception as exc:
            self._disable(exc)
            return None
        return version
