"""Tests for the assert_safeguard test-suite helper."""
import pytest
from fastapi import FastAPI

from fastapi_safeguard import (
    DependencySecurityCheck,
    FastAPISafeguard,
    ScanResult,
    assert_safeguard,
    open_route,
)


@pytest.fixture
def insecure_app():
    app = FastAPI()

    @app.get("/unsecured")
    async def unsecured():
        return {"ok": True}

    return app


@pytest.fixture
def clean_app():
    app = FastAPI()

    @open_route
    @app.get("/health")
    async def health() -> dict:
        return {"status": "healthy"}

    return app


def test_raises_on_new_findings(insecure_app, baseline):
    with pytest.raises(AssertionError) as excinfo:
        assert_safeguard(insecure_app, baseline_path=str(baseline))
    message = str(excinfo.value)
    assert "1 new security finding(s) not in baseline:" in message
    assert "+ GET /unsecured has no accepted security dependency" in message


def test_passes_and_returns_result(clean_app, baseline):
    result = assert_safeguard(clean_app, baseline_path=str(baseline))
    assert isinstance(result, ScanResult)
    assert result.ok
    assert result.route_count == 1


def test_respects_baseline(insecure_app, baseline):
    # Accept the finding first, then the assertion passes.
    FastAPISafeguard.recommended(baseline_path=str(baseline), update_baseline=True).process(insecure_app)
    result = assert_safeguard(insecure_app, baseline_path=str(baseline))
    assert result.ok
    assert len(result.accepted_findings) == 1


def test_never_writes_baseline(insecure_app, baseline):
    with pytest.raises(AssertionError):
        assert_safeguard(insecure_app, baseline_path=str(baseline))
    assert not baseline.exists()


def test_custom_checks(insecure_app, baseline):
    class NeverFails(DependencySecurityCheck):
        def check_route(self, route):
            return None

    result = assert_safeguard(insecure_app, checks=[NeverFails()], baseline_path=str(baseline))
    assert result.ok


def test_custom_safeguard(insecure_app, baseline):
    safeguard = FastAPISafeguard.recommended(baseline_path=str(baseline), update_baseline=True)
    safeguard.process(insecure_app)
    result = assert_safeguard(insecure_app, safeguard)
    assert result.ok


def test_safeguard_and_checks_are_mutually_exclusive(insecure_app, baseline):
    safeguard = FastAPISafeguard.recommended(baseline_path=str(baseline))
    with pytest.raises(TypeError, match="not both"):
        assert_safeguard(insecure_app, safeguard, checks=[DependencySecurityCheck()])


def test_no_stdout_noise(insecure_app, baseline, capsys):
    with pytest.raises(AssertionError):
        assert_safeguard(insecure_app, baseline_path=str(baseline))
    assert capsys.readouterr().out == ""
