"""Ponto de entrada da linha de comando: registro e despacho de subcomandos."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from credit_monitor.config import Settings
from credit_monitor.privacy.retention import purge

Command = Callable[[Settings, argparse.Namespace], int]

# As tarefas seguintes registram aqui seus subcomandos.
COMMANDS: dict[str, Command] = {}


def _purge(settings: Settings, args: argparse.Namespace) -> int:
    counts = purge(settings)
    print(f"Retenção aplicada: {counts['quarantine']} arquivo(s) removido(s) da quarentena, "
          f"{counts['logs']} dos logs.")
    return 0


COMMANDS["purge"] = _purge


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="credit_monitor",
        description="Pipeline de monitoramento de modelo de credit scoring.",
    )
    sub = parser.add_subparsers(dest="command", required=True, metavar="comando")
    for name in COMMANDS:
        sub.add_parser(name)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    try:
        args = parser.parse_args(argv)
    except SystemExit as exc:
        return exc.code if isinstance(exc.code, int) else 2
    return COMMANDS[args.command](Settings.from_env(), args)
