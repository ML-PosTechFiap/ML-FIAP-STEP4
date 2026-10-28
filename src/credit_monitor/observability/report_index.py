"""Indice HTML autocontido dos lotes processados (`reports/index.html`)."""

from __future__ import annotations

import html
import json
from pathlib import Path

from credit_monitor.config import SIMULATION_ORDER

_CSS = """
body { font-family: sans-serif; margin: 2rem; color: #222; }
table { border-collapse: collapse; width: 100%; }
th, td { border: 1px solid #ccc; padding: 6px 10px; text-align: left; vertical-align: top; }
th { background: #f0f0f0; }
small { color: #666; }
tr.ok td.status { color: #1a7f37; font-weight: bold; }
tr.blocked td.status { color: #b42318; font-weight: bold; }
tr.blocked { background: #fdf2f2; }
.rec-OK { color: #1a7f37; font-weight: bold; }
.rec-INVESTIGAR { color: #b26a00; font-weight: bold; }
.rec-RETREINAR { color: #b42318; font-weight: bold; }
"""

_COLUMNS = ["Lote", "Status", "Features com drift", "Drift share", "AUC", "Calibração",
            "Recomendação", "Relatórios"]


def _read(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _number(value: float | None, digits: int = 2) -> str:
    return "—" if value is None else f"{value:.{digits}f}".replace(".", ",")


def _link(reports_dir: Path, rel: str, label: str) -> str:
    if not (reports_dir / rel).exists():
        return ""
    return f'<a href="{html.escape(rel)}">{html.escape(label)}</a>'


def _latest_state(reports_dir: Path, batch_id: str) -> tuple[str, Path] | None:
    """`("ok", resumo)` ou `("blocked", falhas)`, o que for mais recente (reprocessamento)."""
    summary = reports_dir / "drift" / f"{batch_id}_summary.json"
    failures = reports_dir / "contracts" / f"{batch_id}_failures.json"
    candidates = [(path.stat().st_mtime, kind, path)
                  for kind, path in (("ok", summary), ("blocked", failures)) if path.exists()]
    if not candidates:
        return None
    _, kind, path = max(candidates)
    return kind, path


def _row(reports_dir: Path, batch_id: str, kind: str, path: Path) -> str | None:
    data = _read(path)
    if data is None:
        return None
    esc = html.escape
    if kind == "blocked":
        rules = ", ".join(sorted(data.get("violations", {})))
        cells = [
            f"<td class=\"status\">BLOQUEADO<br><small>{esc(rules)}</small></td>",
            "<td>—</td><td>—</td><td>—</td><td>—</td><td>—</td>",
            "<td>" + " ".join(filter(None, [
                _link(reports_dir, f"contracts/{batch_id}_failures.html", "falhas do contrato"),
                _link(reports_dir, f"contracts/{batch_id}_failures.json", "json"),
            ])) + "</td>",
        ]
    else:
        drifted = data.get("drifted_features", [])
        performance = data.get("performance") or {}
        recommendation = str(data.get("recommendation", "—"))
        links = [
            _link(reports_dir, f"drift/{batch_id}_summary.json", "resumo"),
            *(
                _link(reports_dir, f"drift/{entry['html']}", label)
                for label, key in (("drift", "data_drift"), ("desempenho", "performance"))
                if (entry := (data.get("evidently") or {}).get(key))
            ),
        ]
        share = data.get("dataset_drift_share")
        cells = [
            '<td class="status">OK</td>',
            f"<td>{len(drifted)}"
            + (f"<br><small>{esc(', '.join(drifted))}</small>" if drifted else "")
            + "</td>",
            f"<td>{'—' if share is None else f'{share:.0%}'}</td>",
            f"<td>{_number(performance.get('auc'))}</td>",
            f"<td>{_number(performance.get('calibration_ratio'))}</td>",
            f'<td class="rec-{esc(recommendation)}">{esc(recommendation)}</td>',
            "<td>" + " ".join(filter(None, links)) + "</td>",
        ]
    return f'<tr class="{kind}"><td>{esc(batch_id)}</td>' + "".join(cells) + "</tr>"


def build_index(reports_dir: Path) -> Path:
    """Gera `reports/index.html` com uma linha por lote presente, na ordem da simulacao."""
    reports_dir = Path(reports_dir)
    rows = []
    for batch_id in SIMULATION_ORDER:
        state = _latest_state(reports_dir, batch_id)
        row = _row(reports_dir, batch_id, *state) if state else None
        if row:
            rows.append(row)
    body = (
        "<table>\n<thead><tr>" + "".join(f"<th>{c}</th>" for c in _COLUMNS)
        + "</tr></thead>\n<tbody>\n" + "\n".join(rows) + "\n</tbody>\n</table>"
        if rows
        else "<p>Nenhum lote processado ainda. Execute <code>simulate-all</code>.</p>"
    )
    page = f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Monitoramento do modelo de crédito - lotes</title>
<style>{_CSS}</style>
</head>
<body>
<h1>Monitoramento do modelo de crédito</h1>
<p>Um lote por mês simulado. Lotes bloqueados não passaram no contrato de dados e não foram
pontuados. Calibração é relativa à referência (1,00 = igual à da base de referência).</p>
{body}
</body>
</html>
"""
    out = reports_dir / "index.html"
    reports_dir.mkdir(parents=True, exist_ok=True)
    out.write_text(page, encoding="utf-8")
    return out
