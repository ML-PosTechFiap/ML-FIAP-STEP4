import json

import pandas as pd
import pytest

from credit_monitor.config import MIN_BATCH_ROWS
from credit_monitor.contracts.validate import (
    ContractResult,
    quarantine,
    validate_batch,
    write_failure_report,
)
from tests.test_observability import _settings


@pytest.fixture(scope="module")
def valid(synthetic_raw):
    from credit_monitor.contracts.schema import domains_from_reference
    from credit_monitor.privacy.pseudonymize import pseudonymize

    df = synthetic_raw[(synthetic_raw.AMT_INCOME_TOTAL < 1e7) & (synthetic_raw.CODE_GENDER != "XNA")
                       & synthetic_raw.AMT_ANNUITY.notna()]
    df = pseudonymize(df, "t").reset_index(drop=True)
    return df, domains_from_reference(df)


def test_valid_passes(valid):
    df, dom = valid
    r = validate_batch(df, dom)
    assert r.passed and r.violations == {} and r.n_rows == len(df)
    assert r.failures.empty and len(r.failing_index) == 0


@pytest.mark.parametrize("mutate,rule", [
    (lambda d: d.assign(AMT_INCOME_TOTAL=d.AMT_INCOME_TOTAL.where(d.index != 0)), "income_not_null"),
    (lambda d: d.assign(AMT_INCOME_TOTAL=d.AMT_INCOME_TOTAL.where(d.index != 0, 2e7)), "income_range"),
    (lambda d: d.assign(DAYS_BIRTH=d.DAYS_BIRTH.where(d.index != 0, -15 * 365.25)), "age_range"),
    (lambda d: d.assign(DAYS_EMPLOYED=d.DAYS_EMPLOYED.where(d.index != 0, 500)), "days_employed_valid"),
    (lambda d: d.assign(AMT_CREDIT=d.AMT_CREDIT.where(d.index != 0, -1)), "credit_positive"),
    (lambda d: d.assign(EXT_SOURCE_2=d.EXT_SOURCE_2.where(d.index != 0, 1.5)), "ext_source_range"),
    (lambda d: d.assign(client_key=d.client_key.where(d.index != 1, d.client_key[0])), "client_key_unique"),
    (lambda d: d.assign(client_key=d.client_key.astype(object).where(d.index != 0, None)),
     "client_key_not_null"),
    (lambda d: d.assign(AMT_ANNUITY=d.AMT_ANNUITY.where(d.index != 0)), "annuity_not_null"),
    (lambda d: d.assign(AMT_ANNUITY=d.AMT_ANNUITY.where(d.index != 0, 0.0)), "annuity_positive"),
    (lambda d: d.astype({"NAME_CONTRACT_TYPE": object}).assign(
        NAME_CONTRACT_TYPE=lambda x: x.NAME_CONTRACT_TYPE.where(x.index != 0, "Crypto")),
     "domain_NAME_CONTRACT_TYPE"),
    (lambda d: d.astype({"CODE_GENDER": object}).assign(
        CODE_GENDER=lambda x: x.CODE_GENDER.where(x.index != 0, "XNA")),
     "domain_CODE_GENDER"),
    (lambda d: d.assign(TARGET=d.TARGET.where(d.index != 0, 2)), "target_binary"),
    (lambda d: d.head(MIN_BATCH_ROWS - 1), "min_rows"),
])
def test_each_rule_fails(valid, mutate, rule):
    df, dom = valid
    r = validate_batch(mutate(df.copy()), dom)
    assert not r.passed and r.violations.get(rule, 0) >= 1


def test_annuity_null_counts_only_not_null_rule(valid):
    df, dom = valid
    r = validate_batch(df.assign(AMT_ANNUITY=df.AMT_ANNUITY.where(df.index > 2)), dom)
    assert r.violations == {"annuity_not_null": 3}
    assert list(r.failing_index) == [0, 1, 2]


def test_category_dtype_outside_domain_is_caught(valid):
    df, dom = valid
    bad = df.copy()
    bad["NAME_INCOME_TYPE"] = bad["NAME_INCOME_TYPE"].astype(str)
    bad.loc[3, "NAME_INCOME_TYPE"] = "Astronaut"
    bad["NAME_INCOME_TYPE"] = bad["NAME_INCOME_TYPE"].astype("category")
    r = validate_batch(bad, dom)
    assert r.violations == {"domain_NAME_INCOME_TYPE": 1}


def test_target_is_optional(valid):
    df, dom = valid
    assert validate_batch(df.drop(columns=["TARGET"]), dom).passed


def test_missing_required_column_is_reported(valid):
    df, dom = valid
    r = validate_batch(df.drop(columns=["AMT_CREDIT"]), dom)
    assert not r.passed and sum(r.violations.values()) >= 1
    assert r.failures["column"].eq("AMT_CREDIT").any()


def test_string_in_numeric_column_does_not_raise(valid):
    df, dom = valid
    bad = df.copy()
    bad["AMT_CREDIT"] = bad["AMT_CREDIT"].astype(object)
    bad.loc[5, "AMT_CREDIT"] = "abc"
    r = validate_batch(bad, dom)
    assert not r.passed
    assert 5 in set(r.failing_index)
    assert r.failures["column"].eq("AMT_CREDIT").any()
    assert r.failures.loc[r.failures["column"] == "AMT_CREDIT", "failure_value"].eq("abc").any()


def test_extra_columns_are_allowed(valid):
    df, dom = valid
    assert validate_batch(df.assign(EXTRA=1), dom).passed


def test_failures_never_contain_raw_id(valid):
    df, dom = valid
    r = validate_batch(df.assign(AMT_CREDIT=-1.0), dom)
    assert "SK_ID_CURR" not in r.failures.columns
    assert set(r.failures.columns) == {"rule", "column", "failure_value", "client_key"}


def test_failures_carry_client_key_and_index(valid):
    df, dom = valid
    r = validate_batch(df.assign(AMT_CREDIT=df.AMT_CREDIT.where(df.index != 7, -1)), dom)
    row = r.failures.iloc[0]
    assert row["rule"] == "credit_positive" and row["column"] == "AMT_CREDIT"
    assert row["failure_value"] == "-1.0" and row["client_key"] == df.client_key[7]
    assert list(r.failing_index) == [7]


def test_frame_level_failure_has_no_client_key_nor_index(valid):
    df, dom = valid
    r = validate_batch(df.head(10), dom)
    assert r.violations == {"min_rows": 1}
    assert r.failures["client_key"].isna().all() and len(r.failing_index) == 0


def _failed_result():
    failures = pd.DataFrame({
        "rule": ["credit_positive"] * 600,
        "column": ["AMT_CREDIT"] * 600,
        "failure_value": ["-1.0"] * 600,
        "client_key": [f"k{i}" for i in range(600)],
    })
    return ContractResult(passed=False, n_rows=20_000, violations={"credit_positive": 600},
                          failures=failures, failing_index=pd.Index(range(600)))


def test_write_failure_report(tmp_path):
    out = tmp_path / "novo" / "contracts"  # nao existe ainda
    json_path, html_path = write_failure_report(_failed_result(), "03-dirty", out)
    assert json_path == out / "03-dirty_failures.json"
    assert html_path == out / "03-dirty_failures.html"
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["contract_version"] == "1.0" and payload["batch_id"] == "03-dirty"
    assert payload["passed"] is False and payload["n_rows"] == 20_000
    assert payload["violations"] == {"credit_positive": 600}
    assert len(payload["failures"]) == 500
    assert set(payload["failures"][0]) == {"rule", "column", "failure_value", "client_key"}
    html = html_path.read_text(encoding="utf-8")
    assert "credit_positive" in html and "600" in html and "<table" in html
    assert "SK_ID_CURR" not in json_path.read_text(encoding="utf-8") + html


def test_write_failure_report_for_passing_result(tmp_path, valid):
    df, dom = valid
    json_path, html_path = write_failure_report(validate_batch(df, dom), "01", tmp_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["passed"] is True and payload["violations"] == {} and payload["failures"] == []
    assert html_path.exists()


def test_quarantine_copies_and_keeps_source(tmp_path):
    s = _settings(tmp_path)
    src = tmp_path / "batches" / "03-dirty.parquet"
    src.parent.mkdir()
    pd.DataFrame({"a": [1, 2]}).to_parquet(src)
    dest = quarantine(src, "03-dirty", s)
    assert dest == tmp_path / "quarantine" / "03-dirty.parquet"
    assert src.exists() and dest.exists()
    pd.testing.assert_frame_equal(pd.read_parquet(dest), pd.read_parquet(src))
    # reexecucao sobrescreve sem erro
    assert quarantine(src, "03-dirty", s) == dest


def test_mixed_frame_and_row_failures_map_client_key(valid):
    df, dom = valid
    small = df.head(10).assign(AMT_CREDIT=lambda d: d.AMT_CREDIT.where(d.index != 4, -1))
    r = validate_batch(small, dom)
    assert r.violations == {"min_rows": 1, "credit_positive": 1}
    keyed = r.failures.set_index("rule")["client_key"]
    assert keyed["credit_positive"] == df.client_key[4] and keyed["min_rows"] is None
    assert list(r.failing_index) == [4]


def test_synthetic_anomalies_are_counted_per_row(synthetic_raw, valid):
    from credit_monitor.privacy.pseudonymize import pseudonymize

    _, dom = valid
    r = validate_batch(pseudonymize(synthetic_raw, "t"), dom)
    assert r.violations == {
        "income_range": 1,
        "domain_CODE_GENDER": 4,
        "annuity_not_null": 12,
    }
    assert len(r.failing_index) == 17
