"""Validacao bloqueante de lotes, relatorio de falhas e quarentena."""

from __future__ import annotations

import html
import json
import shutil
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
import pandera as pa

from credit_monitor.config import Settings
from credit_monitor.contracts.schema import CONTRACT_VERSION, build_schema
from credit_monitor.privacy.pseudonymize import KEY_COLUMN

FAILURE_COLUMNS = ["rule", "column", "failure_value", "client_key"]
MAX_REPORT_ROWS = 500


@dataclass
class ContractResult:
    passed: bool
    n_rows: int
    violations: dict[str, int]  # regra -> numero de casos de falha
    failures: pd.DataFrame  # colunas: rule, column, failure_value, client_key
    failing_index: pd.Index  # rotulos das linhas que falham em alguma regra por linha


def _empty_failures() -> pd.DataFrame:
    return pd.DataFrame(columns=FAILURE_COLUMNS)


def _normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Nomes estaveis para os erros nativos do Pandera (tipo e coluna ausente)."""
    out = raw.astype({"check": str, "column": object, "failure_case": object})
    missing = out["check"] == "column_in_dataframe"
    out.loc[missing, "column"] = out.loc[missing, "failure_case"]  # o Pandera poe o nome aqui
    out.loc[missing, "failure_case"] = "coluna ausente"
    out.loc[missing, "check"] = "column_missing"
    out.loc[out["check"].str.startswith("coerce_dtype"), "check"] = "dtype_coercion"
    return out


def _stringify(value: object) -> str:
    return "nulo" if pd.isna(value) else str(value)


def _build_failures(raw: pd.DataFrame, df: pd.DataFrame) -> pd.DataFrame:
    raw = _normalize(raw)
    keys = df[KEY_COLUMN] if KEY_COLUMN in df.columns else pd.Series(dtype=object)
    keys = keys[~keys.index.duplicated()] if len(keys) else keys
    index = raw["index"]
    client_key = index.map(keys) if len(keys) else pd.Series(None, index=raw.index, dtype=object)
    out = pd.DataFrame(
        {
            "rule": raw["check"].to_numpy(),
            "column": raw["column"].where(raw["column"].notna(), None),
            "failure_value": raw["failure_case"].map(_stringify),
            "client_key": client_key.astype(object).where(client_key.notna(), None),
        }
    )
    return out.reset_index(drop=True)


def validate_batch(df: pd.DataFrame, domains: dict) -> ContractResult:
    """Valida o frame pseudonimizado contra o contrato. Nunca levanta erro por dado ruim."""
    schema = build_schema(domains)
    n_rows = len(df)
    try:
        schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as exc:
        raw = exc.failure_cases
    except Exception as exc:  # dado que o Pandera nao soube tratar: o lote ainda e bloqueado
        raw = pd.DataFrame(
            {
                "check": ["validation_error"],
                "column": [None],
                "failure_case": [f"{type(exc).__name__}: {exc}"],
                "index": [None],
            }
        )
    else:
        return ContractResult(True, n_rows, {}, _empty_failures(), df.index[:0])

    failures = _build_failures(raw, df)
    violations = {str(k): int(v) for k, v in failures["rule"].value_counts().items()}
    row_labels = raw["index"].dropna().unique()
    failing_index = df.index[df.index.isin(row_labels)]
    return ContractResult(False, n_rows, violations, failures, failing_index)


def _html_report(result: ContractResult, batch_id: str, sample: pd.DataFrame) -> str:
    esc = html.escape
    status = "APROVADO" if result.passed else "BLOQUEADO"
    rules = "".join(
        f"<tr><td>{esc(rule)}</td><td>{count}</td></tr>"
        for rule, count in sorted(result.violations.items(), key=lambda kv: -kv[1])
    )
    rows = "".join(
        "<tr>" + "".join(f"<td>{esc(str(row[c]))}</td>" for c in FAILURE_COLUMNS) + "</tr>"
        for _, row in sample.iterrows()
    )
    return f"""<!DOCTYPE html>
<html lang="pt-BR">
<head>
<meta charset="utf-8">
<title>Falhas do contrato de dados - lote {esc(batch_id)}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; }}
table {{ border-collapse: collapse; margin-bottom: 2rem; }}
th, td {{ border: 1px solid #ccc; padding: 4px 10px; text-align: left; }}
th {{ background: #f0f0f0; }}
</style>
</head>
<body>
<h1>Contrato de dados - lote {esc(batch_id)}</h1>
<p>Situacao: <strong>{status}</strong> | versao do contrato: {CONTRACT_VERSION}
| linhas no lote: {result.n_rows}</p>
<h2>Violacoes por regra</h2>
<table>
<tr><th>Regra</th><th>Ocorrencias</th></tr>
{rules}
</table>
<h2>Amostra de falhas (ate {MAX_REPORT_ROWS} linhas)</h2>
<table>
<tr><th>Regra</th><th>Coluna</th><th>Valor ofensor</th><th>client_key</th></tr>
{rows}
</table>
</body>
</html>
"""


def write_failure_report(result: ContractResult, batch_id: str, out_dir: Path) -> tuple[Path, Path]:
    """Grava `<batch>_failures.json` e `<batch>_failures.html` em `out_dir`."""
    out_dir.mkdir(parents=True, exist_ok=True)
    sample = result.failures.head(MAX_REPORT_ROWS)
    payload = {
        "contract_version": CONTRACT_VERSION,
        "batch_id": batch_id,
        "passed": result.passed,
        "n_rows": result.n_rows,
        "violations": result.violations,
        "failures": json.loads(sample.to_json(orient="records")),
    }
    json_path = out_dir / f"{batch_id}_failures.json"
    html_path = out_dir / f"{batch_id}_failures.html"
    json_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    html_path.write_text(_html_report(result, batch_id, sample), encoding="utf-8")
    return json_path, html_path


def quarantine(src: Path, batch_id: str, settings: Settings) -> Path:
    """Copia (nao move) o lote bloqueado para `data/quarantine/<batch>.parquet`."""
    dest_dir = settings.data_dir / "quarantine"
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"{batch_id}.parquet"
    shutil.copy2(src, dest)
    return dest
