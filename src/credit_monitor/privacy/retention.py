"""Retencao: remove arquivos antigos da quarentena e dos logs (LGPD, minimizacao)."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from credit_monitor.config import RETENTION_DAYS, Settings


def _purge_dir(directory: Path, cutoff: float) -> int:
    if not directory.is_dir():
        return 0
    removed = 0
    for path in directory.iterdir():
        if path.is_symlink() or not path.is_file():
            continue
        if path.stat().st_mtime < cutoff:
            path.unlink()
            removed += 1
    return removed


def purge(settings: Settings, now: datetime | None = None) -> dict[str, int]:
    """Remove arquivos (pelo mtime) fora da janela de retencao; devolve a contagem por area."""
    now = now or datetime.now()
    areas = {
        "quarantine": settings.data_dir / "quarantine",
        "logs": settings.reports_dir / "logs",
    }
    return {
        name: _purge_dir(path, (now - timedelta(days=RETENTION_DAYS[name])).timestamp())
        for name, path in areas.items()
    }
