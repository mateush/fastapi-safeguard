import json
import types

from fastapi import FastAPI
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

from fastapi_safeguard import (
    DependencySecurityCheck,
    ResponseModelSecurityCheck,
    UnsecuredAllowedMethodsCheck,
    BodyModelEnforcementCheck,
    PaginationEnforcementCheck,
    WildcardPathCheck,
    SensitiveFieldExposureCheck,
    ReturnTypeAnnotationCheck,
    SensitiveQueryParamCheck,
    CORSMisconfigurationCheck,
    DebugModeCheck,
    HTTPSRedirectMiddlewareCheck,
    TrustedHostMiddlewareCheck,
    RateLimitingPresenceCheck,
    FastAPISafeguard,
    DangerousMethodExposureCheck,
    SSRFParameterCheck,
    AdminRouteOpenCheck,
    recommended_checks,
)


# ------------------ Helpers ------------------
class FakeParam:
    def __init__(self, name, type_):
        self.name = name
        self.type_ = type_

class FakeDep:
    def __init__(self, call):
        self.call = call

class FakeDependant:
    def __init__(self, dependencies=None, body_params=None, query_params=None):
        self.dependencies = dependencies or []
        self.body_params = body_params or []
        self.query_params = query_params or []

class FakeRoute:
    def __init__(self, path="/r", methods=None, response_model=None, endpoint=None, dependant=None, app=None):
        self.path = path
        self.methods = set(methods or {"GET"})
        self.response_model = response_model
        self.endpoint = endpoint or (lambda: None)
        self.dependant = dependant or FakeDependant()
        self.app = app

# Simple pydantic models
class SimpleModel(BaseModel):
    title: str

class SensitiveModel(BaseModel):
    username: str
    password: str

# ------------------ Unit Tests ------------------

def test_dependency_callable_only_branch():
    def func_dep():
        pass
    check = DependencySecurityCheck(extra_dependencies={func_dep})
    # Force no type dependencies to exercise callable branch with empty tuple
    check.accepted_type_dependencies.clear()
    route = FakeRoute(dependant=FakeDependant(dependencies=[FakeDep(func_dep)]))
    assert check.check_route(route) is None  # passes because callable matches


def test_dependency_extra_dependencies_as_list():
    def custom_dep():
        pass
    # Test that extra_dependencies accepts a list
    check = DependencySecurityCheck(extra_dependencies=[custom_dep])
    route = FakeRoute(dependant=FakeDependant(dependencies=[FakeDep(custom_dep)]))
    assert check.check_route(route) is None  # passes because callable matches


def test_dependency_missing():
    check = DependencySecurityCheck()
    route = FakeRoute()
    msg = check.check_route(route)
    assert "no accepted security dependency" in msg


def test_response_model_skip_safe_method():
    check = ResponseModelSecurityCheck(enforce_methods=["POST"])  # only POST enforced
    route = FakeRoute(methods={"GET"})
    assert check.check_route(route) is None


def test_response_model_missing_on_post():
    check = ResponseModelSecurityCheck(enforce_methods=["POST"])
    route = FakeRoute(methods={"POST"})
    assert "missing response_model" in check.check_route(route)


def test_unsecured_allowed_methods_skip_when_not_listed():
    check = UnsecuredAllowedMethodsCheck(allowed_unsecured=["/open"])  # route not listed
    route = FakeRoute(path="/else", methods={"POST"})
    assert check.check_route(route) is None


def test_body_model_enforcement_skips_upload_and_bytes():
    check = BodyModelEnforcementCheck(enforce_methods=["POST"])  # method enforced
    route = FakeRoute(
        methods={"POST"},
        dependant=FakeDependant(body_params=[
            FakeParam("file", __import__('fastapi').UploadFile),
            FakeParam("raw", dict),
            FakeParam("data", bytes),
        ])
    )
    msg = check.check_route(route)
    assert "raw body param(s): raw" in msg and "file" not in msg and "data" not in msg


def test_pagination_not_list_return():
    check = PaginationEnforcementCheck()
    def endpoint() -> int:  # not list
        return 1
    endpoint.__annotations__["return"] = int
    route = FakeRoute(endpoint=endpoint)
    assert check.check_route(route) is None


def test_pagination_list_missing_params():
    check = PaginationEnforcementCheck()
    def endpoint() -> list[int]:
        return []
    endpoint.__annotations__["return"] = list[int]  # ensure annotation presence
    route = FakeRoute(endpoint=endpoint)
    assert "returns a collection without pagination" in check.check_route(route)


def test_wildcard_path_no_issue():
    check = WildcardPathCheck()
    route = FakeRoute(path="/alpha")
    assert check.check_route(route) is None


def test_sensitive_field_exposure_none():
    check = SensitiveFieldExposureCheck()
    route = FakeRoute(response_model=SimpleModel)
    assert check.check_route(route) is None


def test_sensitive_field_exposure_hit():
    check = SensitiveFieldExposureCheck()
    route = FakeRoute(response_model=SensitiveModel)
    msg = check.check_route(route)
    assert "sensitive fields" in msg and "password" in msg


def test_return_type_annotation_present():
    check = ReturnTypeAnnotationCheck()
    def endpoint() -> dict:
        return {}
    endpoint.__annotations__["return"] = dict
    route = FakeRoute(endpoint=endpoint)
    assert check.check_route(route) is None


def test_return_type_annotation_missing():
    check = ReturnTypeAnnotationCheck()
    def endpoint():
        return {}
    route = FakeRoute(endpoint=endpoint)
    assert "neither response_model" in check.check_route(route)


def test_sensitive_query_allowlist_skips():
    check = SensitiveQueryParamCheck(allowlist=["api_key"])
    route = FakeRoute(dependant=FakeDependant(query_params=[FakeParam("api_key", str)]))
    assert check.check_route(route) is None


def test_sensitive_query_flag():
    check = SensitiveQueryParamCheck()
    route = FakeRoute(dependant=FakeDependant(query_params=[FakeParam("user_token", str)]))
    assert "sensitive data via query params" in check.check_route(route)


def _make_app_with_middleware(entries):
    app = FastAPI()
    app.user_middleware.extend(entries)
    return app


def test_cors_single_run_and_wildcard_detection():
    check = CORSMisconfigurationCheck()
    app = FastAPI()
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)
    # invoke app_check directly
    first = check.app_check(app)
    second = check.app_check(app)
    assert first and "CORS misconfiguration" in first
    assert second is None


def test_debug_mode_single_run():
    check = DebugModeCheck()
    app = FastAPI()
    app.debug = True
    msg1 = check.app_check(app)
    msg2 = check.app_check(app)
    assert msg1 and "debug mode" in msg1
    assert msg2 is None


def test_https_redirect_single_run():
    check = HTTPSRedirectMiddlewareCheck()
    app = FastAPI()
    app.add_middleware(HTTPSRedirectMiddleware)
    assert check.app_check(app) is None  # middleware present
    assert check.app_check(app) is None  # second call noop


def test_https_redirect_missing():
    check = HTTPSRedirectMiddlewareCheck()
    app = FastAPI()
    assert "HTTPS redirect" in check.app_check(app)


def test_trusted_host_present():
    check = TrustedHostMiddlewareCheck()
    app = FastAPI()
    app.add_middleware(TrustedHostMiddleware, allowed_hosts=["*"])
    assert check.app_check(app) is None


def test_trusted_host_missing():
    check = TrustedHostMiddlewareCheck()
    app = FastAPI()
    assert "TrustedHostMiddleware not configured" in check.app_check(app)


def test_rate_limiting_present_produces_finding():
    class Limiter: ...
    check = RateLimitingPresenceCheck()
    app = FastAPI()
    # class name doesn't include keyword – expect finding
    app.user_middleware.append(types.SimpleNamespace(cls=Limiter, options={}))
    assert "rate limiting" in check.app_check(app)


def test_rate_limiting_detected():
    class RateLimitShield: ...
    check = RateLimitingPresenceCheck()
    app = FastAPI()
    app.user_middleware.append(types.SimpleNamespace(cls=RateLimitShield, options={}))
    assert check.app_check(app) is None


def test_baseline_refresh_removing_resolved(tmp_path, capsys):
    # Setup baseline with two findings
    baseline = tmp_path / "baseline.json"
    accepted = ["GET /a has no accepted security dependency", "GET /b has no accepted security dependency"]
    baseline.write_text(json.dumps({"schema_version":1,"generated_at":"x","accepted_findings":accepted,"checks_count":1}))
    # Plugin with update, only one current finding produced
    check = DependencySecurityCheck()
    plugin = FastAPISafeguard(checks=[check], baseline_path=str(baseline), update_baseline=True)

    # Fake routes: /a missing dependency, /b now secure (simulate by putting /b in allowed_unsecured list)
    check.allowed_unsecured.add("/b")

    from fastapi.routing import APIRoute
    app = FastAPI(lifespan=plugin.get_lifespan())

    async def _ep():
        return None
    app.router.routes.append(APIRoute(path="/a", endpoint=_ep, methods=["GET"]))
    app.router.routes.append(APIRoute(path="/b", endpoint=_ep, methods=["GET"]))

    # Run lifespan without expecting failure (no new findings; baseline will be refreshed)
    import asyncio
    async def run_app():
        async with app.router.lifespan_context(app):
            pass
    asyncio.run(run_app())

    out = capsys.readouterr().out
    # Should mention refreshed baseline
    assert any(x in out for x in ["baseline refreshed", "accepted baseline"])  # flexible wording
    with open(baseline) as f:
        data = json.load(f)
    assert data["accepted_findings"] == ["GET /a has no accepted security dependency"]

# Helper simple namespace route for baseline refresh test
class SimpleNamespaceForRoute:
    def __init__(self, path):
        self.path = path
        self.methods = {"GET"}
        self.dependant = FakeDependant()
        self.endpoint = lambda: None


def test_dangerous_method_exposure():
    check = DangerousMethodExposureCheck()
    route = FakeRoute(path="/trace", methods={"GET", "TRACE"})
    msg = check.check_route(route)
    assert "dangerous HTTP method" in msg and "TRACE" in msg
    safe_route = FakeRoute(path="/safe", methods={"GET"})
    assert check.check_route(safe_route) is None


def test_ssrf_parameter_check():
    check = SSRFParameterCheck()
    route = FakeRoute(dependant=FakeDependant(query_params=[FakeParam("url", str), FakeParam("name", str)]))
    msg = check.check_route(route)
    assert "potential SSRF" in msg and "url" in msg
    # allowlist case
    check2 = SSRFParameterCheck(allowlist=["url"])
    route2 = FakeRoute(dependant=FakeDependant(query_params=[FakeParam("url", str)]))
    assert check2.check_route(route2) is None


def test_admin_route_open_check():
    check = AdminRouteOpenCheck()
    open_admin = FakeRoute(path="/admin/dashboard", methods={"GET"})
    msg = check.check_route(open_admin)
    assert "admin route" in msg
    # secure admin (simulate dependency presence)
    secure_admin = FakeRoute(path="/admin/panel", methods={"GET"}, dependant=FakeDependant(dependencies=[FakeDep(lambda: True)]))
    assert check.check_route(secure_admin) is None


def test_recommended_preset_includes_heuristics():
    # Heuristics flag exists but currently doesn't add any checks (all removed due to low value)
    plugin = FastAPISafeguard.recommended()
    names = {type(c).__name__ for c in plugin.checks}
    # Core checks always present
    assert "DependencySecurityCheck" in names and "ResponseModelSecurityCheck" in names
    # Heuristic checks no longer included by default
    assert not ({"RateLimitingPresenceCheck", "DangerousMethodExposureCheck", "SSRFParameterCheck", "AdminRouteOpenCheck"} & names)


def test_recommended_preset_excludes_heuristics():
    plugin = FastAPISafeguard.recommended()
    names = {type(c).__name__ for c in plugin.checks}
    assert not ({"RateLimitingPresenceCheck", "DangerousMethodExposureCheck", "SSRFParameterCheck", "AdminRouteOpenCheck"} & names)
    assert "DependencySecurityCheck" in names


def test_recommended_checks_function_equivalence():
    a = recommended_checks()
    b = FastAPISafeguard.recommended().checks
    assert {type(c).__name__ for c in a} == {type(c).__name__ for c in b}
