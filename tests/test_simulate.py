import json

import numpy as np
import pandas as pd

from credit_monitor import config

CLEAN = ["01", "02", "03", "04", "05", "06"]


def _load(s, b):
    from credit_monitor.privacy.pseudonymize import pseudonymize

    return pseudonymize(pd.read_parquet(s.data_dir / f"batches/{b}/batch.parquet"), "t")


def _manifest(s, b):
    return json.loads((s.data_dir / f"batches/{b}/manifest.json").read_text())


def test_all_batches_created(prepared):
    for b in config.SIMULATION_ORDER:
        assert (prepared.data_dir / f"batches/{b}/manifest.json").exists()
        assert (prepared.data_dir / f"batches/{b}/batch.parquet").exists()
        assert (prepared.data_dir / f"batches/{b}/labels_original.parquet").exists()


def test_clean_batches_pass_contract_and_dirty_fails(prepared):
    from credit_monitor.contracts.validate import validate_batch

    dom = json.loads((prepared.data_dir / "reference/domains.json").read_text())
    for b in CLEAN:
        assert validate_batch(_load(prepared, b), dom).passed, b
    r = validate_batch(_load(prepared, "03-dirty"), dom)
    expected = _manifest(prepared, "03-dirty")["expected_rules"]
    assert set(expected) <= set(r.violations)
    assert all(r.violations[rule] >= 5 for rule in expected)


def test_drift_is_progressive(prepared):
    inc = [_load(prepared, b)["AMT_INCOME_TOTAL"].median() for b in ["01", "06"]]
    assert inc[1] / inc[0] > 1.08
    unemp = (_load(prepared, "06")["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL).mean()
    assert 0.28 <= unemp <= 0.32


def test_concept_drift_raises_default_and_keeps_original(prepared):
    b06 = pd.read_parquet(prepared.data_dir / "batches/06/batch.parquet")
    orig = pd.read_parquet(prepared.data_dir / "batches/06/labels_original.parquet")
    assert b06["TARGET"].mean() > orig["TARGET"].mean() + 0.02
    assert len(orig) == len(b06)
    assert (orig["SK_ID_CURR"].to_numpy() == b06["SK_ID_CURR"].to_numpy()).all()
    assert _manifest(prepared, "06")["n_concept_flips"] == int(
        (b06["TARGET"].to_numpy() != orig["TARGET"].to_numpy()).sum()
    )
    assert _manifest(prepared, "01")["n_concept_flips"] == 0


def test_batch_01_has_no_drifted_features(prepared):
    manifest = _manifest(prepared, "01")
    assert manifest["injected_features"] == [] and manifest["drifted_features"] == []


def test_manifest_content(prepared):
    m = _manifest(prepared, "06")
    assert m["batch_id"] == "06"
    assert m["params"]["unemployed_target"] == 0.30
    assert m["n_rows"] == len(pd.read_parquet(prepared.data_dir / "batches/06/batch.parquet"))
    for key in ("injected_features", "drifted_features"):
        assert m[key] == sorted(m[key])
        assert set(m[key]) <= set(config.MODEL_FEATURES)
    assert set(m["drifted_features"]) <= set(m["injected_features"])
    assert {"AGE_YEARS", "EXT_SOURCE_1", "EXT_SOURCE_2", "CREDIT_TERM", "AMT_INCOME_TOTAL"} <= set(
        m["injected_features"]
    )
    assert {"AGE_YEARS", "EXT_SOURCE_2", "CREDIT_TERM", "YEARS_EMPLOYED"} <= set(
        m["drifted_features"]
    )
    assert "expected_rules" not in m
    assert _manifest(prepared, "03-dirty")["expected_rules"] == [
        "income_not_null", "age_range", "client_key_unique",
        "credit_positive", "domain_NAME_CONTRACT_TYPE", "ext_source_range",
    ]


def _young_share(df):
    return float(((-df["DAYS_BIRTH"] / 365.25) < 30).mean())


def test_batches_have_expected_size_and_come_from_pool(prepared):
    pool = pd.read_parquet(prepared.data_dir / "reference/production_pool.parquet")
    size = min(config.BATCH_SIZE, len(pool) // 6)
    for b in CLEAN:
        ids = pd.read_parquet(prepared.data_dir / f"batches/{b}/batch.parquet")["SK_ID_CURR"]
        assert len(ids) == size and ids.is_unique  # sem reposicao dentro do lote
        assert set(ids) <= set(pool["SK_ID_CURR"])


def test_young_share_grows_progressively(prepared):
    ref = _young_share(pd.read_parquet(prepared.data_dir / "reference/train.parquet"))
    share = {b: _young_share(pd.read_parquet(prepared.data_dir / f"batches/{b}/batch.parquet"))
             for b in CLEAN}
    assert abs(share["01"] - ref) < 0.02
    assert share["04"] > ref
    assert share["05"] > share["04"]
    assert share["06"] > share["05"]


def test_dirty_has_five_rows_of_each_defect(prepared):
    clean = pd.read_parquet(prepared.data_dir / "batches/03/batch.parquet")
    dirty = pd.read_parquet(prepared.data_dir / "batches/03-dirty/batch.parquet")
    assert len(dirty) == len(clean)
    assert dirty["AMT_INCOME_TOTAL"].isna().sum() == 5
    assert (dirty["AMT_CREDIT"] < 0).sum() == 5
    assert (dirty["EXT_SOURCE_2"] > 1).sum() == 5
    assert (dirty["DAYS_BIRTH"] > -18 * 365.25).sum() == 5
    assert dirty["SK_ID_CURR"].duplicated(keep=False).sum() == 10
    assert (~dirty["NAME_CONTRACT_TYPE"].isin(clean["NAME_CONTRACT_TYPE"].unique())).sum() == 5


def test_build_all_batches_is_deterministic(prepared, tmp_path):
    from dataclasses import replace

    from credit_monitor.data.simulate import build_all_batches

    pool = pd.read_parquet(prepared.data_dir / "reference/production_pool.parquet")
    train = pd.read_parquet(prepared.data_dir / "reference/train.parquet")
    s2 = replace(prepared, data_dir=tmp_path)
    ids = build_all_batches(s2, pool, train, config.SEED)
    assert ids == config.SIMULATION_ORDER
    for b in config.SIMULATION_ORDER:
        for name in ("batch.parquet", "labels_original.parquet"):
            a = pd.read_parquet(prepared.data_dir / f"batches/{b}/{name}")
            c = pd.read_parquet(tmp_path / f"batches/{b}/{name}")
            pd.testing.assert_frame_equal(a, c)
        assert _manifest(prepared, b) == _manifest(s2, b)


def test_apply_drift_rules_on_small_frame():
    from credit_monitor.data.simulate import apply_drift
    from credit_monitor.data.synthetic import make_synthetic_application

    df = make_synthetic_application(3000, with_anomalies=False).iloc[:2000].reset_index(drop=True)
    p = config.DRIFT_SCHEDULE["06"]
    out, orig = apply_drift(df, p, 0.25, np.random.default_rng(1))
    assert list(orig.index) == list(df["SK_ID_CURR"]) and (orig.to_numpy() == df["TARGET"].to_numpy()).all()
    assert out["AMT_INCOME_TOTAL"].max() <= config.INCOME_MAX
    assert out[["EXT_SOURCE_2", "EXT_SOURCE_3"]].max().max() <= 1
    sentinel = out["DAYS_EMPLOYED"] == config.DAYS_EMPLOYED_SENTINEL
    assert abs(sentinel.mean() - 0.30) < 0.005
    assert (out.loc[sentinel, "NAME_INCOME_TYPE"].astype(str).isin(["Unemployed", "Pensioner"])).all()
    assert (out["DAYS_EMPLOYED"][~sentinel] <= 0).all()
    assert out["EXT_SOURCE_1"].isna().mean() > df["EXT_SOURCE_1"].isna().mean()
    assert (out["TARGET"] >= df["TARGET"]).all()  # so vira 0 -> 1
    assert out.dtypes.astype(str).to_dict() == df.dtypes.astype(str).to_dict()
    # sem alteracao quando a fracao ja esta acima da meta
    p_low = config.DriftParams(unemployed_target=0.01)
    low, _ = apply_drift(df, p_low, 0.25, np.random.default_rng(1))
    assert (low["DAYS_EMPLOYED"] == df["DAYS_EMPLOYED"]).all()


RATIO_FEATURES = ("CREDIT_INCOME_RATIO", "ANNUITY_INCOME_RATIO")


def test_injected_features_mapping():
    """Verdade-base: toda feature tocada pelos parametros ativos, independente da magnitude."""
    assert config.injected_features(config.DRIFT_SCHEDULE["01"]) == []
    f03 = config.injected_features(config.DRIFT_SCHEDULE["03"])
    assert {"YEARS_EMPLOYED", "CREDIT_TERM", "AMT_INCOME_TOTAL", *RATIO_FEATURES} <= set(f03)
    assert "AGE_YEARS" not in f03 and "EXT_SOURCE_1" not in f03
    f06 = config.injected_features(config.DRIFT_SCHEDULE["06"])
    assert {"EXT_SOURCE_1", "AGE_YEARS", "EXT_SOURCE_3", "AMT_GOODS_PRICE"} <= set(f06)
    same = config.DriftParams(income_infl=0.1, credit_infl=0.1, annuity_infl=0.1)
    assert "CREDIT_TERM" not in config.injected_features(same)


def test_drifted_features_mapping():
    """Subconjunto esperado como detectavel: so mudancas acima do minimo calibrado."""
    assert config.drifted_features(config.DRIFT_SCHEDULE["01"]) == []
    f03 = config.drifted_features(config.DRIFT_SCHEDULE["03"])
    assert "CREDIT_TERM" in f03 and "AGE_YEARS" not in f03 and "YEARS_EMPLOYED" not in f03
    f06 = config.drifted_features(config.DRIFT_SCHEDULE["06"])
    assert {"YEARS_EMPLOYED", "AGE_YEARS", "EXT_SOURCE_2", "NAME_INCOME_TYPE"} <= set(f06)
    assert not {"EXT_SOURCE_1", "AMT_INCOME_TOTAL", *RATIO_FEATURES} & set(f06)
    for params in config.DRIFT_SCHEDULE.values():
        assert set(config.drifted_features(params)) <= set(config.injected_features(params))
    same = config.DriftParams(income_infl=0.3, credit_infl=0.3, annuity_infl=0.3)
    f_same = config.drifted_features(same)
    assert "AMT_CREDIT" in f_same and not {"CREDIT_TERM", *RATIO_FEATURES} & set(f_same)
    assert "AMT_CREDIT" in f_same and not {"CREDIT_TERM", *RATIO_FEATURES} & set(f_same)
