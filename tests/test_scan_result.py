"""Unit tests for the structured scan layer: Finding, ScanResult, collect()."""
import dataclasses

import pytest
from fastapi import FastAPI

from fastapi_safeguard import (
    DependencySecurityCheck,
    FastAPISafeguard,
    Finding,
    PaginationEnforcementCheck,
    ScanResult,
    open_route,
)


@pytest.fixture
def insecure_app():
    app = FastAPI()

    @app.get("/items")
    async def items() -> list:
        return []

    @app.get("/users")
    async def users():
        return []

    return app


def make_safeguard(baseline, **kwargs):
    checks = kwargs.pop("checks", [DependencySecurityCheck(), PaginationEnforcementCheck()])
    return FastAPISafeguard(checks=checks, baseline_path=str(baseline), **kwargs)


def test_finding_from_check_carries_metadata():
    check = DependencySecurityCheck()
    finding = Finding.from_check(check, "GET /x has no accepted security dependency")
    assert finding.check == "DependencySecurityCheck"
    assert finding.category == "auth"
    assert finding.owasp == ("API2", "API5")


def test_finding_is_immutable():
    finding = Finding(text="t", check="c")
    with pytest.raises(dataclasses.FrozenInstanceError):
        finding.text = "other"


def test_collect_is_pure(insecure_app, baseline, capsys):
    safeguard = make_safeguard(baseline, update_baseline=True)
    result = safeguard.collect(insecure_app)

    assert not result.ok
    assert len(result.new) == 3  # 2 auth findings + 1 pagination
    assert capsys.readouterr().out == ""  # no printing
    assert not baseline.exists()  # no baseline write, even with update_baseline


def test_collect_counts_routes_and_checks(insecure_app, baseline):
    result = make_safeguard(baseline).collect(insecure_app)
    assert result.route_count == 2
    assert result.checks_count == 2


def test_scan_result_len_and_iter(insecure_app, baseline):
    result = make_safeguard(baseline).collect(insecure_app)
    assert len(result) == 3
    assert [f.text for f in result] == list(result.texts)


def test_scan_result_by_category(insecure_app, baseline):
    result = make_safeguard(baseline).collect(insecure_app)
    grouped = result.by_category()
    assert {f.text for f in grouped["auth"]} | {f.text for f in grouped["performance"]} == set(result.texts)


def test_baseline_split_accepted_resolved(insecure_app, baseline):
    # Accept the current findings, then change the app: one finding resolves,
    # one new appears.
    make_safeguard(baseline, update_baseline=True).process(insecure_app)

    app = FastAPI()

    @open_route
    @app.get("/items")  # auth finding resolved via open_route
    async def items(limit: int = 10) -> list:  # pagination finding resolved
        return []

    @app.get("/users")
    async def users():  # still unsecured -> accepted
        return []

    @app.get("/orders")
    async def orders():  # new unsecured route
        return []

    result = make_safeguard(baseline).collect(app)
    assert not result.ok
    assert {f.text for f in result.new} == {"GET /orders has no accepted security dependency"}
    assert {f.text for f in result.accepted_findings} == {"GET /users has no accepted security dependency"}
    assert len(result.resolved) == 2


def test_ok_when_all_accepted(insecure_app, baseline):
    make_safeguard(baseline, update_baseline=True).process(insecure_app)
    result = make_safeguard(baseline).collect(insecure_app)
    assert result.ok
    assert result.new == ()
    assert len(result.accepted_findings) == 3


def test_process_returns_exit_codes(insecure_app, baseline, capsys):
    failing = make_safeguard(baseline)
    assert failing.process(insecure_app) == 1
    assert "❌ Security check failed" in capsys.readouterr().out

    accepting = make_safeguard(baseline, update_baseline=True)
    assert accepting.process(insecure_app) == 0
    assert "accepted into baseline" in capsys.readouterr().out

    clean = make_safeguard(baseline)
    assert clean.process(insecure_app) == 0


# ----------------- baseline & reporting edges -----------------

def test_validate_baseline_path_rejects_relative_traversal(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match="outside working directory"):
        FastAPISafeguard(baseline_path="../escape.json")


def test_load_baseline_ignores_non_list_payload(insecure_app, baseline):
    import json
    baseline.write_text(json.dumps({"accepted_findings": "not-a-list"}))
    result = make_safeguard(baseline).collect(insecure_app)
    assert len(result.new) == 3  # malformed baseline accepted nothing


def test_write_baseline_noop_without_path(baseline):
    safeguard = make_safeguard(baseline)
    safeguard.baseline_path = ""
    safeguard._write_baseline(["finding"])
    assert not baseline.exists()


def test_print_summary_with_no_findings(baseline, capsys):
    safeguard = make_safeguard(baseline)
    safeguard._print_category_summary(safeguard.collect(FastAPI()))
    assert "No findings to summarize" in capsys.readouterr().out


def test_failure_output_lists_previously_accepted(insecure_app, baseline, capsys):
    make_safeguard(baseline, update_baseline=True).process(insecure_app)
    capsys.readouterr()

    @insecure_app.get("/extra")
    async def extra():
        return []

    assert make_safeguard(baseline).process(insecure_app) == 1
    out = capsys.readouterr().out
    assert "+ GET /extra has no accepted security dependency" in out
    assert "ℹ️  Previously accepted findings (baseline):" in out
    assert "= GET /users has no accepted security dependency" in out


def test_resolved_hint_without_update(insecure_app, baseline, capsys):
    make_safeguard(baseline, update_baseline=True).process(insecure_app)
    capsys.readouterr()

    app = FastAPI()

    @app.get("/users")
    async def users():  # accepted finding remains; /items findings resolved
        return []

    assert make_safeguard(baseline).process(app) == 0
    out = capsys.readouterr().out
    assert "match accepted baseline (1 accepted)" in out
    assert "previously accepted finding(s) resolved" in out
