"""Ponto de entrada da linha de comando: registro e despacho de subcomandos."""

from __future__ import annotations

import argparse
import json
from collections.abc import Callable

import pyarrow.parquet as pq

from credit_monitor.config import Settings
from credit_monitor.data.download import DatasetUnavailable, ensure_dataset
from credit_monitor.data.prepare import run_prepare
from credit_monitor.model.train import run_train
from credit_monitor.privacy.retention import purge

Command = Callable[[Settings, argparse.Namespace], int]
Configurer = Callable[[argparse.ArgumentParser], None]

# As tarefas seguintes registram aqui seus subcomandos (via `register`).
COMMANDS: dict[str, Command] = {}
CONFIGURERS: dict[str, Configurer] = {}


def register(name: str, handler: Command, configure: Configurer | None = None) -> None:
    """Registra um subcomando; `configure(subparser)` adiciona os argumentos dele."""
    COMMANDS[name] = handler
    if configure is not None:
        CONFIGURERS[name] = configure


def _purge(settings: Settings, args: argparse.Namespace) -> int:
    counts = purge(settings)
    print(f"Retenção aplicada: {counts['quarantine']} arquivo(s) removido(s) da quarentena, "
          f"{counts['logs']} dos logs.")
    return 0


def _download(settings: Settings, args: argparse.Namespace) -> int:
    try:
        path = ensure_dataset(settings)
    except DatasetUnavailable as exc:
        print(exc)
        return 2
    print(f"Dataset disponível em {path}.")
    return 0


def _prepare(settings: Settings, args: argparse.Namespace) -> int:
    try:
        paths = run_prepare(settings, synthetic=args.synthetic, n=args.n)
    except DatasetUnavailable as exc:
        print(exc)
        return 2
    sizes = {name: pq.ParquetFile(paths[name]).metadata.num_rows
             for name in ("train", "valid", "test", "pool")}
    n_raw = json.loads(paths["failures_json"].read_text(encoding="utf-8"))["n_rows"]
    print(f"Linhas brutas: {n_raw}. Removidas pelo contrato: {n_raw - sum(sizes.values())}.")
    print("Referência preparada: " + ", ".join(f"{name}={rows}" for name, rows in sizes.items())
          + f". Arquivos em {paths['train'].parent}.")
    return 0


def _configure_prepare(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--synthetic", action="store_true",
                        help="usa uma base sintética no mesmo schema, sem o dataset real")
    parser.add_argument("--n", type=int, default=60_000,
                        help="linhas da base sintética (padrão: 60000)")


def _train(settings: Settings, args: argparse.Namespace) -> int:
    try:
        meta = run_train(settings)
    except RuntimeError as exc:
        print(exc)
        return 2
    m = meta["reference_metrics"]
    print(f"Modelo treinado: AUC={m['auc']:.4f}, KS={m['ks']:.4f}, corte={meta['cutoff']:.4f}. "
          f"Artefatos em {settings.models_dir}.")
    return 0


def parse_interval(text: str) -> float:
    """Segundos, com ou sem sufixo `s`: `30` ou `30s`."""
    try:
        value = float(text.strip().removesuffix("s"))
    except ValueError:
        value = -1.0
    if not value >= 0:  # tambem rejeita NaN
        raise argparse.ArgumentTypeError(
            f"intervalo inválido: {text!r} (use segundos, como 30 ou 30s)"
        )
    return value


def _monitor(settings: Settings, args: argparse.Namespace) -> int:
    from credit_monitor.model.predict import ModelNotTrained
    from credit_monitor.pipeline import BatchNotFound, run_monitor

    try:
        result = run_monitor(settings, args.batch)
    except (BatchNotFound, ModelNotTrained) as exc:
        print(exc)
        return 2
    if result.status == "blocked":
        print(f"Lote {result.batch_id} bloqueado pelo contrato de dados. "
              f"Detalhes em {settings.reports_dir / 'contracts'}.")
        return 1
    print(f"Lote {result.batch_id} processado: recomendação {result.recommendation}. "
          f"Resumo em {result.summary_path}.")
    return 0


def _configure_monitor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--batch", required=True, help="identificador do lote (ex.: 01, 03-dirty)")


def _simulate_all(settings: Settings, args: argparse.Namespace) -> int:
    from credit_monitor.model.predict import ModelNotTrained
    from credit_monitor.pipeline import BatchNotFound, run_simulate_all

    try:
        results = run_simulate_all(settings, interval_seconds=args.interval)
    except (BatchNotFound, ModelNotTrained) as exc:
        print(exc)
        return 2
    for result in results:
        if result.status == "blocked":
            print(f"Lote {result.batch_id}: bloqueado pelo contrato de dados.")
        else:
            print(f"Lote {result.batch_id}: recomendação {result.recommendation}.")
    print(f"Índice dos relatórios: {settings.reports_dir / 'index.html'}.")
    return 0


def _configure_simulate_all(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--interval", type=parse_interval, default=0.0,
                        help="pausa entre lotes, em segundos (ex.: 30 ou 30s)")


def _reset_metrics(settings: Settings, args: argparse.Namespace) -> int:
    from credit_monitor.config import SIMULATION_ORDER
    from credit_monitor.pipeline import reset_groups

    report = reset_groups(settings)
    if not report.configured:
        print("Pushgateway não configurado (PUSHGATEWAY_URL): nada a apagar.")
        return 0
    total = len(SIMULATION_ORDER) + 1
    message = f"Pushgateway: {report.deleted} de {total} grupo(s) apagado(s)"
    if report.failed:
        message += f", {report.failed} falha(s)"
    if report.unreachable:
        message += ". Interrompido: gateway inacessível"
    print(message + ".")
    return 0


register("purge", _purge)
register("download", _download)
register("prepare", _prepare, _configure_prepare)
register("train", _train)
register("monitor", _monitor, _configure_monitor)
register("simulate-all", _simulate_all, _configure_simulate_all)
register("reset-metrics", _reset_metrics)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit_monitor",
        description="Pipeline de monitoramento de modelo de credit scoring.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="comando")
    for name in COMMANDS:
        subparser = sub.add_parser(name)
        if name in CONFIGURERS:
            CONFIGURERS[name](subparser)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    return COMMANDS[args.command](Settings.from_env(), args)
