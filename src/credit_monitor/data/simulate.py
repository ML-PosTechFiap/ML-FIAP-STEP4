"""Simulação de drift mensal progressivo (dados e conceito) e geração dos lotes.

Cada lote `01`..`06` é uma amostra independente do pool de produção (sem reposição dentro
do lote, mas com linhas que podem se repetir entre meses), com o drift de
`config.DRIFT_SCHEDULE` aplicado. Cada mês é um sorteio independente da população de
solicitantes; reaproveitar clientes entre meses é aceitável no monitoramento por lote e
evita esgotar o pool de jovens, o que inverteria o perfil novo dos lotes 04..06. O `03-dirty` é uma cópia do `03` com defeitos de
qualidade injetados, para exercitar o bloqueio do contrato de dados.
"""

from __future__ import annotations

import dataclasses
import json
import logging

import numpy as np
import pandas as pd

from credit_monitor import config
from credit_monitor.config import DriftParams, Settings

logger = logging.getLogger(__name__)

YOUNG_AGE = 30
DIRTY_ROWS_PER_DEFECT = 5
DIRTY_RULES = [
    "income_not_null",
    "age_range",
    "client_key_unique",
    "credit_positive",
    "domain_NAME_CONTRACT_TYPE",
    "ext_source_range",
]
UNKNOWN_CONTRACT_TYPE = "Consumer loans"
UNEMPLOYED = "Unemployed"

# Os lotes limpos que viram amostras do pool; o `03-dirty` deriva do `03`.
CLEAN_BATCH_IDS = [b for b in config.SIMULATION_ORDER if not b.endswith("-dirty")]


def _age_years(df: pd.DataFrame) -> np.ndarray:
    return (-df["DAYS_BIRTH"].to_numpy(dtype="float64")) / 365.25


def sample_batch(
    pool: pd.DataFrame,
    available: np.ndarray,
    size: int,
    young_weight: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sorteia `size` posições de `available` (sem reposição), ponderando os jovens.

    Peso de cada linha: `1 + young_weight * (idade < 30)`. Devolve posições ordenadas.
    """
    if size > len(available):
        raise ValueError(f"Pool insuficiente: {len(available)} linhas disponíveis, {size} pedidas.")
    age = _age_years(pool.iloc[available])
    weights = 1.0 + young_weight * (age < YOUNG_AGE)
    picked = rng.choice(available, size=size, replace=False, p=weights / weights.sum())
    return np.sort(picked)


def _convert_to_unemployed(df: pd.DataFrame, target: float, rng: np.random.Generator) -> None:
    sentinel = df["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
    missing = int(np.ceil(target * len(df))) - int(sentinel.sum())
    if missing <= 0:
        return
    employed = np.flatnonzero((~sentinel & df["DAYS_EMPLOYED"].notna()).to_numpy())
    rows = rng.choice(employed, size=min(missing, len(employed)), replace=False)
    if UNEMPLOYED not in df["NAME_INCOME_TYPE"].cat.categories:
        df["NAME_INCOME_TYPE"] = df["NAME_INCOME_TYPE"].cat.add_categories([UNEMPLOYED])
    df.iloc[rows, df.columns.get_loc("DAYS_EMPLOYED")] = config.DAYS_EMPLOYED_SENTINEL
    df.iloc[rows, df.columns.get_loc("NAME_INCOME_TYPE")] = UNEMPLOYED


def apply_drift(
    df: pd.DataFrame, p: DriftParams, annuity_ratio_p75: float, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.Series]:
    """Aplica o drift de `p` a uma cópia de `df` e devolve (lote alterado, TARGET original).

    Ordem de aplicação (cada passo vê o resultado do anterior):
      0. amostragem (feita antes, em `sample_batch`, ponderada por `young_weight`);
      1. inflação: `AMT_INCOME_TOTAL` (income_infl, com teto do contrato), `AMT_CREDIT` e
         `AMT_GOODS_PRICE` (credit_infl), `AMT_ANNUITY` (annuity_infl);
      2. desemprego: empregados sorteados viram `DAYS_EMPLOYED = 365243` e
         `NAME_INCOME_TYPE = "Unemployed"` até a fração de sentinelas atingir
         `unemployed_target` (nada acontece se já estiver acima);
      3. `employed_shrink` sobre o `DAYS_EMPLOYED` dos empregados restantes;
      4. nulos extras em `EXT_SOURCE_1` (fração dos não nulos);
      5. `EXT_SOURCE_2/3` multiplicados por `1 - ext23_shrink`, com clip em [0, 1];
      6. concept drift, calculado sobre os valores já alterados: no segmento
         (`AMT_ANNUITY / AMT_INCOME_TOTAL > annuity_ratio_p75` ou sentinela), uma fração
         `concept_flip` dos `TARGET == 0` vira 1.

    O TARGET original sai indexado por `SK_ID_CURR`. O índice de `df` é descartado.
    """
    out = df.reset_index(drop=True).copy()
    original = pd.Series(
        out["TARGET"].to_numpy(copy=True),
        index=pd.Index(out["SK_ID_CURR"], name="SK_ID_CURR"),
        name="TARGET",
    )

    inflation = {
        "AMT_INCOME_TOTAL": p.income_infl,
        "AMT_CREDIT": p.credit_infl,
        "AMT_GOODS_PRICE": p.credit_infl,
        "AMT_ANNUITY": p.annuity_infl,
    }
    for col, infl in inflation.items():
        if infl:
            out[col] = (out[col].astype("float64") * (1 + infl)).astype(out[col].dtype)
    out["AMT_INCOME_TOTAL"] = out["AMT_INCOME_TOTAL"].clip(upper=config.INCOME_MAX)

    if p.unemployed_target is not None:
        _convert_to_unemployed(out, p.unemployed_target, rng)

    if p.employed_shrink:
        days = out["DAYS_EMPLOYED"]
        employed = (days != config.DAYS_EMPLOYED_SENTINEL) & days.notna()
        shrunk = days.astype("float64") * (1 - p.employed_shrink)
        out["DAYS_EMPLOYED"] = days.where(~employed, shrunk).astype(days.dtype)

    if p.ext1_extra_null:
        not_null = np.flatnonzero(out["EXT_SOURCE_1"].notna().to_numpy())
        n_null = int(round(p.ext1_extra_null * len(not_null)))
        rows = rng.choice(not_null, size=n_null, replace=False)
        out.iloc[rows, out.columns.get_loc("EXT_SOURCE_1")] = np.nan

    if p.ext23_shrink:
        for col in ("EXT_SOURCE_2", "EXT_SOURCE_3"):
            shrunk = (out[col].astype("float64") * (1 - p.ext23_shrink)).clip(0, 1)
            out[col] = shrunk.astype(out[col].dtype)

    if p.concept_flip:
        ratio = out["AMT_ANNUITY"].astype("float64") / out["AMT_INCOME_TOTAL"].astype("float64")
        no_job = out["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
        squeezed = (ratio > annuity_ratio_p75) | no_job
        candidates = np.flatnonzero((squeezed & (out["TARGET"] == 0)).to_numpy())
        n_flip = int(round(p.concept_flip * len(candidates)))
        rows = rng.choice(candidates, size=n_flip, replace=False)
        out.iloc[rows, out.columns.get_loc("TARGET")] = 1

    return out, original


def make_dirty(df: pd.DataFrame, rng: np.random.Generator) -> tuple[pd.DataFrame, list[str]]:
    """Cópia de `df` com 5 linhas de cada defeito, em linhas distintas.

    Defeitos: renda nula, idade de 15 anos, `SK_ID_CURR` duplicado (5 linhas copiam o ID de
    outras 5), crédito negativo, `NAME_CONTRACT_TYPE` fora do domínio e `EXT_SOURCE_2 > 1`.
    Devolve o frame sujo e as regras do contrato que devem falhar.
    """
    out = df.reset_index(drop=True).copy()
    k = DIRTY_ROWS_PER_DEFECT
    rows = rng.choice(len(out), size=7 * k, replace=False)
    null_income, young, duplicated, negative, unknown, ext, sources = (
        rows[i * k : (i + 1) * k] for i in range(7)
    )

    out.iloc[null_income, out.columns.get_loc("AMT_INCOME_TOTAL")] = np.nan
    out.iloc[young, out.columns.get_loc("DAYS_BIRTH")] = -15 * 365.25
    out.iloc[duplicated, out.columns.get_loc("SK_ID_CURR")] = out["SK_ID_CURR"].to_numpy()[sources]
    credit = out["AMT_CREDIT"].to_numpy()[negative]
    out.iloc[negative, out.columns.get_loc("AMT_CREDIT")] = -np.abs(credit)
    contract = out["NAME_CONTRACT_TYPE"]
    out["NAME_CONTRACT_TYPE"] = contract.cat.add_categories([UNKNOWN_CONTRACT_TYPE])
    out.iloc[unknown, out.columns.get_loc("NAME_CONTRACT_TYPE")] = UNKNOWN_CONTRACT_TYPE
    out.iloc[ext, out.columns.get_loc("EXT_SOURCE_2")] = 1.5
    return out, list(DIRTY_RULES)


def _write_batch(
    settings: Settings,
    batch_id: str,
    batch: pd.DataFrame,
    original_target: pd.Series,
    params: DriftParams,
    n_flips: int,
    expected_rules: list[str] | None = None,
) -> None:
    out_dir = settings.data_dir / "batches" / batch_id
    out_dir.mkdir(parents=True, exist_ok=True)
    batch.to_parquet(out_dir / "batch.parquet", index=False)
    labels = pd.DataFrame(
        {"SK_ID_CURR": batch["SK_ID_CURR"].to_numpy(), "TARGET": original_target.to_numpy()}
    )
    labels.to_parquet(out_dir / "labels_original.parquet", index=False)
    manifest: dict = {
        "batch_id": batch_id,
        "params": dataclasses.asdict(params),
        "injected_features": config.injected_features(params),
        "drifted_features": config.drifted_features(params),
        "n_rows": len(batch),
        "n_concept_flips": n_flips,
    }
    if expected_rules is not None:
        manifest["expected_rules"] = expected_rules
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def build_all_batches(
    settings: Settings, pool: pd.DataFrame, reference_train: pd.DataFrame, seed: int
) -> list[str]:
    """Gera `data/batches/<id>/` (batch, manifest e rótulos originais) para todos os lotes.

    Cada lote tem `min(BATCH_SIZE, len(pool) // 6)` linhas, sorteadas sem reposição dentro
    do lote. Os lotes são sorteios independentes do pool inteiro (uma linha pode aparecer em
    meses diferentes), para que o viés de idade de cada mês não dependa do que os outros
    consumiram. O RNG do lote de índice `i` é `default_rng(seed + i)`. O p75 da razão
    anuidade/renda vem só do `reference_train`. Devolve os IDs em `SIMULATION_ORDER`.
    """
    pool = pool.reset_index(drop=True)
    size = min(config.BATCH_SIZE, len(pool) // len(CLEAN_BATCH_IDS))
    if size < 1:
        raise ValueError(f"Pool de produção pequeno demais para simular ({len(pool)} linhas).")
    if size < config.MIN_BATCH_ROWS:
        logger.warning(
            "Lotes de %d linhas, abaixo do mínimo do contrato (%d): serão bloqueados.",
            size,
            config.MIN_BATCH_ROWS,
        )
    ratio = reference_train["AMT_ANNUITY"] / reference_train["AMT_INCOME_TOTAL"]
    annuity_ratio_p75 = float(ratio.quantile(0.75))

    available = np.arange(len(pool))
    built: dict[str, tuple[pd.DataFrame, pd.Series]] = {}
    for i, batch_id in enumerate(CLEAN_BATCH_IDS):
        params, rng = config.DRIFT_SCHEDULE[batch_id], np.random.default_rng(seed + i)
        rows = sample_batch(pool, available, size, params.young_weight, rng)
        sampled = pool.iloc[rows].reset_index(drop=True)
        built[batch_id] = apply_drift(sampled, params, annuity_ratio_p75, rng)

    ids: list[str] = []
    for batch_id in config.SIMULATION_ORDER:
        if batch_id.endswith("-dirty"):
            base_id = batch_id.removesuffix("-dirty")
            base, _ = built[base_id]
            dirty_rng = np.random.default_rng(seed + len(CLEAN_BATCH_IDS))
            dirty, rules = make_dirty(base, dirty_rng)
            params = config.DRIFT_SCHEDULE[base_id]
            _write_batch(settings, batch_id, dirty, dirty["TARGET"], params, 0, rules)
        else:
            batch, original = built[batch_id]
            n_flips = int((batch["TARGET"].to_numpy() != original.to_numpy()).sum())
            params = config.DRIFT_SCHEDULE[batch_id]
            _write_batch(settings, batch_id, batch, original, params, n_flips)
        ids.append(batch_id)
    return ids
