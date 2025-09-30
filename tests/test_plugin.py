import json
import re
import types
from pathlib import Path
from contextlib import asynccontextmanager

import pytest
from fastapi import FastAPI, Depends, Body
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fastapi_safeguard import (
    FastAPISafeguard,
    DependencySecurityCheck,
    ResponseModelSecurityCheck,
    UnsecuredAllowedMethodsCheck,
    CORSMisconfigurationCheck,
    DebugModeCheck,
    BodyModelEnforcementCheck,
    PaginationEnforcementCheck,
    WildcardPathCheck,
    SensitiveFieldExposureCheck,
    ReturnTypeAnnotationCheck,
    SensitiveQueryParamCheck,
    HTTPSRedirectMiddlewareCheck,
    TrustedHostMiddlewareCheck,
    RateLimitingPresenceCheck,
    open_route,
    disable_security_checks,
)


# ----------------- Helpers -----------------
async def run_startup(app: FastAPI):
    async with app.router.lifespan_context(app):
        pass

@pytest.fixture(autouse=True)
def clear_env(tmp_path, monkeypatch):
    # Ensure no global baseline interferes
    for k in ["SECURITY_BASELINE_UPDATE", "SECURITY_BASELINE_PATH"]:
        monkeypatch.delenv(k, raising=False)
    # Remove default baseline file if created by prior runs
    default_file = Path("security_baseline.json")
    if default_file.exists():
        default_file.unlink()

# Provide a unique baseline path per test
@pytest.fixture
def baseline(tmp_path):
    return tmp_path / "baseline.json"

# Simple accepted dependency class for dependency check
class AllowedDep:
    def __call__(self):
        return True

# ----------------- Tests -----------------
@pytest.mark.asyncio
async def test_dependency_check_failure_then_accept(baseline, capsys):
    plugin = FastAPISafeguard(checks=[DependencySecurityCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/noauth")
    async def noauth():
        return {"ok": True}

    with pytest.raises(SystemExit):
        await run_startup(app)
    assert "no accepted security dependency" in capsys.readouterr().out

    # Accept via update_baseline flag (constructor arg)
    plugin2 = FastAPISafeguard(checks=[DependencySecurityCheck()], baseline_path=str(baseline), update_baseline=True)
    app2 = FastAPI(lifespan=plugin2.get_lifespan())

    @app2.get("/noauth")
    async def noauth2():
        return {"ok": True}

    await run_startup(app2)
    out = capsys.readouterr().out
    assert "accepted into baseline" in out
    with open(baseline) as f:
        data = json.load(f)
    assert any("no accepted security dependency" in f for f in data["accepted_findings"])

@pytest.mark.asyncio
async def test_dependency_check_pass_with_dependency(baseline, capsys):
    plugin = FastAPISafeguard(checks=[DependencySecurityCheck(extra_dependencies={AllowedDep})], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    dep_instance = AllowedDep()

    @app.get("/secure")
    async def secure(_: bool = Depends(dep_instance)):
        return {"ok": True}

    await run_startup(app)
    out = capsys.readouterr().out
    assert "no accepted security dependency" not in out

@pytest.mark.asyncio
async def test_open_route_and_disable_security(baseline, capsys):
    plugin = FastAPISafeguard(checks=[DependencySecurityCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @open_route
    @app.get("/public")
    async def public():
        return {"p": True}

    @disable_security_checks
    @app.get("/skip")
    async def skip():
        return {"s": True}

    await run_startup(app)
    out = capsys.readouterr().out
    assert "/public" not in out and "/skip" not in out

@pytest.mark.asyncio
async def test_response_model_and_return_annotation_fail(baseline, capsys):
    plugin = FastAPISafeguard(checks=[ResponseModelSecurityCheck(), ReturnTypeAnnotationCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.post("/create")
    async def create():  # unsafe method, no response model
        return {"id": 1}

    with pytest.raises(SystemExit):
        await run_startup(app)
    out = capsys.readouterr().out
    assert "missing response_model" in out

@pytest.mark.asyncio
async def test_return_type_only_pass(baseline, capsys):
    plugin = FastAPISafeguard(checks=[ReturnTypeAnnotationCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/ok")
    async def ok() -> dict:
        return {"x": 1}

    await run_startup(app)
    assert "has neither response_model" not in capsys.readouterr().out

@pytest.mark.asyncio
async def test_body_model_enforcement_fail(baseline, capsys):
    plugin = FastAPISafeguard(checks=[BodyModelEnforcementCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.patch("/raw")
    async def raw(payload: dict = Body(...)):
        return payload

    with pytest.raises(SystemExit):
        await run_startup(app)
    assert "uses non-model raw body param(s): payload" in capsys.readouterr().out

@pytest.mark.asyncio
async def test_body_model_enforcement_pass(baseline, capsys):
    class Model(BaseModel):
        name: str
    plugin = FastAPISafeguard(checks=[BodyModelEnforcementCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.patch("/raw")
    async def raw(payload: Model):
        return payload

    await run_startup(app)
    assert "uses non-model" not in capsys.readouterr().out

@pytest.mark.asyncio
async def test_pagination_enforcement_fail(baseline, capsys):
    plugin = FastAPISafeguard(checks=[PaginationEnforcementCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/items")
    async def items() -> list[int]:
        return [1, 2]

    with pytest.raises(SystemExit):
        await run_startup(app)
    assert "returns a collection without pagination params" in capsys.readouterr().out

@pytest.mark.asyncio
async def test_pagination_enforcement_pass(baseline, capsys):
    plugin = FastAPISafeguard(checks=[PaginationEnforcementCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/items")
    async def items(limit: int = 10) -> list[int]:
        return [1]

    await run_startup(app)
    assert "returns a collection without" not in capsys.readouterr().out

@pytest.mark.asyncio
async def test_wildcard_path_check_fail(baseline, capsys):
    plugin = FastAPISafeguard(checks=[WildcardPathCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/files/{path:path}")
    async def files(path: str):
        return path

    with pytest.raises(SystemExit):
        await run_startup(app)
    assert "uses broad wildcard" in capsys.readouterr().out

@pytest.mark.asyncio
async def test_sensitive_field_and_query_fail(baseline, capsys):
    class User(BaseModel):
        username: str
        password: str

    plugin = FastAPISafeguard(checks=[SensitiveFieldExposureCheck(), SensitiveQueryParamCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/account", response_model=User)
    async def account(password_token: str):
        return User(username="u", password="p")

    with pytest.raises(SystemExit):
        await run_startup(app)
    out = capsys.readouterr().out
    assert "response_model exposes potentially sensitive fields" in out
    assert "exposes potentially sensitive data via query params" in out

@pytest.mark.asyncio
async def test_sensitive_query_allowlist(baseline, capsys):
    plugin = FastAPISafeguard(checks=[SensitiveQueryParamCheck(allowlist=["password_token"])], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/account")
    async def account(password_token: str):
        return {"ok": True}

    await run_startup(app)
    assert "exposes potentially sensitive data" not in capsys.readouterr().out

@pytest.mark.asyncio
async def test_cors_misconfiguration_fail(baseline, capsys):
    # Use manual user_middleware injection for deterministic detection across Starlette versions
    plugin = FastAPISafeguard(checks=[CORSMisconfigurationCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())
    app.user_middleware.append(types.SimpleNamespace(cls=CORSMiddleware, options={
        'allow_origins': ['*'], 'allow_methods': ['*'], 'allow_headers': ['*'], 'allow_credentials': True
    }))

    @app.get("/ping")
    async def ping():
        return {"pong": True}

    with pytest.raises(SystemExit):
        await run_startup(app)
    assert "CORS misconfiguration" in capsys.readouterr().out

@pytest.mark.asyncio
async def test_cors_ok(baseline, capsys):
    plugin = FastAPISafeguard(checks=[CORSMisconfigurationCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())
    app.add_middleware(CORSMiddleware, allow_origins=["https://example.com"], allow_methods=["GET"], allow_headers=["X-Req"], allow_credentials=False)

    @app.get("/ok")
    async def ok():
        return {"pong": True}

    await run_startup(app)
    assert "CORS misconfiguration" not in capsys.readouterr().out

@pytest.mark.asyncio
async def test_https_trustedhost_ratelimit_fail(baseline, capsys):
    plugin = FastAPISafeguard(checks=[HTTPSRedirectMiddlewareCheck(), TrustedHostMiddlewareCheck(), RateLimitingPresenceCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/x")
    async def x():
        return {}

    with pytest.raises(SystemExit):
        await run_startup(app)
    out = capsys.readouterr().out
    assert "HTTPS redirect" in out
    assert "TrustedHostMiddleware not configured" in out
    assert "No apparent rate limiting" in out

@pytest.mark.asyncio
async def test_https_trustedhost_ratelimit_pass(baseline, capsys):
    class RateLimiterMiddleware:
        async def __call__(self, scope, receive, send):
            await self.app(scope, receive, send)
        def __init__(self, app):
            self.app = app

    plugin = FastAPISafeguard(checks=[HTTPSRedirectMiddlewareCheck(), TrustedHostMiddlewareCheck(), RateLimitingPresenceCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())
    # deterministic manual injection
    app.user_middleware.append(types.SimpleNamespace(cls=HTTPSRedirectMiddleware, options={}))
    app.user_middleware.append(types.SimpleNamespace(cls=TrustedHostMiddleware, options={'allowed_hosts': ['*']}))
    app.user_middleware.append(types.SimpleNamespace(cls=RateLimiterMiddleware, options={}))

    @app.get("/ok")
    async def ok():
        return {}

    await run_startup(app)
    out = capsys.readouterr().out
    assert "HTTPS redirect" not in out
    assert "TrustedHostMiddleware not configured" not in out
    assert "No apparent rate limiting" not in out

@pytest.mark.asyncio
async def test_debug_mode_check_fail(baseline, capsys):
    plugin = FastAPISafeguard(checks=[DebugModeCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())
    app.debug = True  # force debug flag

    @app.get("/d")
    async def d():
        return {}

    with pytest.raises(SystemExit):
        await run_startup(app)
    assert "Application running in debug mode" in capsys.readouterr().out

@pytest.mark.asyncio
async def test_baseline_resolved_and_prune(baseline, capsys):
    # Start by accepting a finding
    plugin_a = FastAPISafeguard(checks=[DependencySecurityCheck()], baseline_path=str(baseline), update_baseline=True)
    app_a = FastAPI(lifespan=plugin_a.get_lifespan())

    @app_a.get("/missing")
    async def missing():
        return {}

    await run_startup(app_a)

    # Now secure route -> finding resolved
    plugin_b = FastAPISafeguard(checks=[DependencySecurityCheck(extra_dependencies={AllowedDep})], baseline_path=str(baseline))
    app_b = FastAPI(lifespan=plugin_b.get_lifespan())
    dep_instance = AllowedDep()

    @app_b.get("/missing")
    async def missing_ok(dep: bool = Depends(dep_instance)):
        return {}

    await run_startup(app_b)
    out = capsys.readouterr().out
    assert any(msg in out for msg in ["previously accepted finding(s) resolved", "All security findings match", "No security findings."])

    # Prune baseline
    plugin_c = FastAPISafeguard(checks=[DependencySecurityCheck(extra_dependencies={AllowedDep})], baseline_path=str(baseline), update_baseline=True)
    app_c = FastAPI(lifespan=plugin_c.get_lifespan())

    @app_c.get("/missing")
    async def missing_ok2(dep: bool = Depends(dep_instance)):
        return {}

    await run_startup(app_c)
    with open(baseline) as f:
        data = json.load(f)
    assert data["accepted_findings"] == []

@pytest.mark.asyncio
async def test_run_checks_with_custom_lifespan(baseline, capsys):
    """Test that run_checks() works when called from a custom lifespan."""
    plugin = FastAPISafeguard(checks=[DependencySecurityCheck()], baseline_path=str(baseline), update_baseline=True)

    @asynccontextmanager
    async def custom_lifespan(app: FastAPI):
        # Custom startup
        plugin.run_checks(app)
        yield
        # Custom shutdown (nothing here)

    app = FastAPI(lifespan=custom_lifespan)

    @app.get("/noauth")
    async def noauth():
        return {"ok": True}

    await run_startup(app)
    out = capsys.readouterr().out
    assert "Security checks passed with new findings accepted into baseline" in out
    assert baseline.exists()

    # Verify baseline was created
    with open(baseline) as f:
        data = json.load(f)
    assert len(data["accepted_findings"]) == 1
    assert "no accepted security dependency" in data["accepted_findings"][0]

@pytest.mark.asyncio
async def test_invalid_baseline_parsing(baseline, capsys):
    baseline.write_text("{ invalid json")
    plugin = FastAPISafeguard(checks=[DependencySecurityCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/a")
    async def a():
        return {"a":1}

    with pytest.raises(SystemExit):
        await run_startup(app)
    assert "Could not parse baseline file" in capsys.readouterr().out

@pytest.mark.asyncio
async def test_category_summary_and_owasp_codes(baseline, capsys):
    plugin = FastAPISafeguard(checks=[DependencySecurityCheck(), PaginationEnforcementCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/list")
    async def list_items() -> list[int]:
        return [1]

    @app.get("/data")
    async def data():
        return {"x":1}

    with pytest.raises(SystemExit):
        await run_startup(app)
    out = capsys.readouterr().out
    assert re.search(r"Category\s+Total\s+New\s+Accepted", out)
    assert "auth" in out and "performance" in out
    assert "API2" in out

@pytest.mark.asyncio
async def test_disable_security_checks_skips_all(baseline, capsys):
    plugin = FastAPISafeguard(checks=[DependencySecurityCheck(), ResponseModelSecurityCheck(), PaginationEnforcementCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @disable_security_checks
    @app.post("/insecure")
    async def insecure(body: dict = Body(...)):
        return body

    await run_startup(app)
    out = capsys.readouterr().out
    assert "/insecure" not in out
    assert "All security checks passed" in out or "No security findings" in out

@pytest.mark.asyncio
async def test_unsecured_allowed_methods_fail(baseline, capsys):
    plugin = FastAPISafeguard(
        checks=[UnsecuredAllowedMethodsCheck(allowed_unsecured=["/openapi.json", "/docs", "/redoc", "/unsecured"])],
        baseline_path=str(baseline),
    )
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.post("/unsecured")
    async def unsecured():
        return {"x": 1}

    with pytest.raises(SystemExit):
        await run_startup(app)
    assert "exposes unsafe method" in capsys.readouterr().out

@pytest.mark.asyncio
async def test_unsecured_allowed_methods_pass(baseline, capsys):
    plugin = FastAPISafeguard(
        checks=[UnsecuredAllowedMethodsCheck(allowed_unsecured=["/unsecured"])],
        baseline_path=str(baseline),
    )
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/unsecured")
    async def unsecured():
        return {"x": 1}

    await run_startup(app)
    assert "exposes unsafe method" not in capsys.readouterr().out

@pytest.mark.asyncio
async def test_allowed_unsecured_skips_response_model_check(baseline, capsys):
    plugin = FastAPISafeguard(
        checks=[ResponseModelSecurityCheck(enforce_methods=["POST"], allowed_unsecured=["/skipresp"])],
        baseline_path=str(baseline),
    )
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.post("/skipresp")
    async def skipresp():
        return {"ok": True}

    await run_startup(app)
    assert "missing response_model" not in capsys.readouterr().out

@pytest.mark.asyncio
async def test_open_route_still_enforces_other_checks(baseline, capsys):
    plugin = FastAPISafeguard(
        checks=[ResponseModelSecurityCheck(enforce_methods=["POST"]), DependencySecurityCheck()],
        baseline_path=str(baseline),
    )
    app = FastAPI(lifespan=plugin.get_lifespan())

    @open_route
    @app.post("/openmissing")
    async def openmissing():
        return {"no": "model"}

    with pytest.raises(SystemExit):
        await run_startup(app)
    out = capsys.readouterr().out
    # dependency check skipped, but response model check should fail
    assert "missing response_model" in out
    assert "no accepted security dependency" not in out

@pytest.mark.asyncio
async def test_disable_security_checks_multi(baseline, capsys):
    plugin = FastAPISafeguard(
        checks=[DependencySecurityCheck(), ResponseModelSecurityCheck(), BodyModelEnforcementCheck()],
        baseline_path=str(baseline),
    )
    app = FastAPI(lifespan=plugin.get_lifespan())

    @disable_security_checks
    @app.post("/multi")
    async def multi(payload: dict = Body(...)):
        return payload

    await run_startup(app)
    out = capsys.readouterr().out
    assert "/multi" not in out

@pytest.mark.asyncio
async def test_cors_no_middleware_no_findings(baseline, capsys):
    plugin = FastAPISafeguard(checks=[CORSMisconfigurationCheck()], baseline_path=str(baseline))
    app = FastAPI(lifespan=plugin.get_lifespan())

    @app.get("/plain")
    async def plain():
        return {"ok": True}

    await run_startup(app)
    out = capsys.readouterr().out
    assert "CORS misconfiguration" not in out
