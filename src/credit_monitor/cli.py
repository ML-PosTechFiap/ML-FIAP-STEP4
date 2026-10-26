"""Ponto de entrada da linha de comando: registro e despacho de subcomandos."""

from __future__ import annotations

import argparse
from collections.abc import Callable

from credit_monitor.config import Settings

Command = Callable[[Settings, argparse.Namespace], int]

# As tarefas seguintes registram aqui seus subcomandos.
COMMANDS: dict[str, Command] = {}


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
