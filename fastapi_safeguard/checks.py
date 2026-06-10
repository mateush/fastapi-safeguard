"""Security checks and the decorators that scope them.

Every check is a small, single-purpose class implementing the `SecurityCheck`
contract: inspect one route (or, via ``app_check``, the whole application) and
return ``None`` when compliant or a stable, deterministic finding string when
not. Baselines match on the exact finding text, so wording must never depend
on runtime state such as timestamps or ordering.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import (
    Any,
    Callable,
    Iterable,
    List,
    Optional,
    Sequence,
    Set,
    Type,
    Union,
    get_origin,
)

from fastapi import FastAPI, UploadFile
from fastapi.routing import APIRoute
from fastapi.security import (
    APIKeyCookie,
    APIKeyHeader,
    APIKeyQuery,
    HTTPBasic,
    HTTPBearer,
    OAuth2PasswordBearer,
    OAuth2PasswordRequestForm,
)
from pydantic import BaseModel
from starlette.middleware.cors import CORSMiddleware
from starlette.middleware.httpsredirect import HTTPSRedirectMiddleware
from starlette.middleware.trustedhost import TrustedHostMiddleware

# --------------------------------------------------------------------------------------
# Decorator markers
# --------------------------------------------------------------------------------------
_OPEN_ATTR = "_secure_open"          # Only bypasses dependency/auth check
_SKIP_ALL_ATTR = "_secure_skip_all"  # Bypasses every check


def open_route(func: Callable) -> Callable:
    """Mark a route as intentionally open (auth dependency check skipped, others enforced)."""
    setattr(func, _OPEN_ATTR, True)
    return func


def disable_security_checks(func: Callable) -> Callable:
    """Disable ALL security checks for this route."""
    setattr(func, _SKIP_ALL_ATTR, True)
    return func


def _is_open(route: APIRoute) -> bool:
    return bool(getattr(route.endpoint, _OPEN_ATTR, False))


def _skip_all(route: APIRoute) -> bool:
    return bool(getattr(route.endpoint, _SKIP_ALL_ATTR, False))


# --------------------------------------------------------------------------------------
# Shared constants & helpers
# --------------------------------------------------------------------------------------
DEFAULT_ALLOWED_UNSECURED: frozenset = frozenset({"/openapi.json", "/docs", "/redoc"})
SUSPICIOUS_FIELD_PARTS: frozenset = frozenset({
    "password", "passwd", "secret", "token", "api_key", "apikey",
    "key", "credential", "auth", "private",
})
SUSPICIOUS_QUERY_PARTS = SUSPICIOUS_FIELD_PARTS  # reuse same list
RATE_LIMIT_KEYWORDS = ("ratelimit", "throttle")


def _route_dependencies(route: APIRoute) -> List[Callable]:
    """Collect dependency callables for a route, including nested sub-dependencies.

    Walks the full dependant tree so a security scheme wrapped inside a custom
    dependency (e.g. ``Depends(get_current_user)`` where ``get_current_user``
    itself depends on ``OAuth2PasswordBearer``) is still discovered.
    """
    collected: List[Callable] = []
    seen: Set[int] = set()
    stack = list(route.dependant.dependencies)
    while stack:
        dependant = stack.pop()
        if id(dependant) in seen:
            continue
        seen.add(id(dependant))
        if dependant.call is not None:
            collected.append(dependant.call)
        stack.extend(dependant.dependencies)
    return collected


# --------------------------------------------------------------------------------------
# Base classes
# --------------------------------------------------------------------------------------
class SecurityCheck(ABC):
    """Contract for a security check.

    Attributes:
        CATEGORY: Classification category for reporting (e.g., 'auth', 'schema').
        OWASP: List of related OWASP API Security Top 10 identifiers (e.g., ['API3']).

    Returns:
        check_route: None if the route passes the check, otherwise a string describing
                     the security issue.
    """
    CATEGORY = "general"
    OWASP: List[str] = []  # e.g. ["API3"]

    @abstractmethod
    def check_route(self, route: APIRoute) -> Optional[str]:  # pragma: no cover - interface
        ...


class RouteCheck(SecurityCheck):
    """Base for checks needing allowed_unsecured handling and skip_all support.

    Args:
        allowed_unsecured: Paths that should be excluded from checks
                          (defaults to /openapi.json, /docs, /redoc).
    """

    def __init__(self, allowed_unsecured: Optional[Sequence[str]] = None) -> None:
        self.allowed_unsecured: Set[str] = set(allowed_unsecured or DEFAULT_ALLOWED_UNSECURED)

    @abstractmethod
    def _analyze(self, route: APIRoute) -> Optional[str]:  # pragma: no cover - interface
        """Core check logic implemented by subclasses.

        Args:
            route: The FastAPI route to analyze.

        Returns:
            None if check passes, otherwise a finding description.
        """
        ...

    def check_route(self, route: APIRoute) -> Optional[str]:
        if _skip_all(route):
            return None
        if route.path in self.allowed_unsecured:
            return None
        return self._analyze(route)


class SingleRunMixin:
    """Mixin for checks that should run only once per application lifecycle.

    Correct usage: subclasses perform detection when not yet _done; subsequent invocations skip.
    """
    def __init__(self) -> None:
        self._done = False

    def should_run(self) -> bool:
        if self._done:
            return False
        self._done = True
        return True


# --------------------------------------------------------------------------------------
# Concrete Checks
# --------------------------------------------------------------------------------------
class DependencySecurityCheck(SecurityCheck):
    # OWASP: API2 (Broken Authentication), API5 (Broken Function Level Authorization)
    CATEGORY = "auth"
    OWASP = ["API2", "API5"]

    """Ensure at least one accepted auth/security dependency is present.

    open_route decorator: bypasses this check only.
    disable_security_checks decorator: bypasses all checks (handled globally by _skip_all).
    """

    DEFAULT_SECURITY_DEPENDENCIES: Set[Type] = {
        OAuth2PasswordBearer,
        OAuth2PasswordRequestForm,
        HTTPBasic,
        HTTPBearer,
        APIKeyHeader,
        APIKeyQuery,
        APIKeyCookie,
    }

    def __init__(
        self,
        allowed_unsecured: Optional[Sequence[str]] = None,
        extra_dependencies: Optional[Union[List[Any], Set[Any]]] = None,
    ) -> None:
        self.allowed_unsecured = set(allowed_unsecured or DEFAULT_ALLOWED_UNSECURED)
        raw = set(self.DEFAULT_SECURITY_DEPENDENCIES)
        if extra_dependencies:
            raw |= set(extra_dependencies)
        self.accepted_type_dependencies: Set[Type] = {d for d in raw if isinstance(d, type)}
        self.accepted_callable_dependencies: Set[Callable] = {d for d in raw if not isinstance(d, type)}

    def _has_accepted_dependency(self, deps: List[Callable]) -> bool:
        """Check if any dependency matches accepted security dependencies.

        Args:
            deps: List of dependency callables from route.

        Returns:
            True if at least one accepted dependency is found.
        """
        types_tuple = tuple(self.accepted_type_dependencies)
        for dep in deps:
            if types_tuple and isinstance(dep, types_tuple):
                return True
            if dep in self.accepted_callable_dependencies:
                return True
        return False

    def check_route(self, route: APIRoute) -> Optional[str]:
        if _skip_all(route):
            return None
        if _is_open(route):  # explicit open route
            return None
        if route.path in self.allowed_unsecured:
            return None
        deps = _route_dependencies(route)
        if self._has_accepted_dependency(deps):
            return None
        return f"{','.join(route.methods)} {route.path} has no accepted security dependency"


class ResponseModelSecurityCheck(RouteCheck):
    # OWASP: API3 (Broken Object Property Level Authorization / Excessive Data Exposure)
    CATEGORY = "schema"
    OWASP = ["API3"]
    def __init__(self, enforce_methods: Optional[Iterable[str]] = None, allowed_unsecured: Optional[Sequence[str]] = None) -> None:
        super().__init__(allowed_unsecured)
        self.methods = {m.upper() for m in (enforce_methods or ["POST", "PUT", "PATCH", "DELETE"])}

    def _analyze(self, route: APIRoute) -> Optional[str]:
        if not (self.methods & route.methods):
            return None
        if route.response_model is None:
            return f"{','.join(route.methods)} {route.path} missing response_model for unsafe method(s)"
        return None


class UnsecuredAllowedMethodsCheck(RouteCheck):
    # OWASP: API5 (Broken Function Level Authorization)
    CATEGORY = "auth"
    OWASP = ["API5"]
    def __init__(self, allowed_unsecured: Optional[Sequence[str]] = None, safe_methods: Optional[Iterable[str]] = None) -> None:
        # Here allowed_unsecured means explicit open paths list.
        super().__init__(allowed_unsecured)
        self.safe = {m.upper() for m in (safe_methods or ["GET", "HEAD", "OPTIONS"])}

    def _analyze(self, route: APIRoute) -> Optional[str]:
        """Check if allowed unsecured paths expose unsafe methods.

        This check has inverted logic: it ONLY checks routes in allowed_unsecured.
        """
        # This will never be called by RouteCheck.check_route() for paths in allowed_unsecured
        # So we override check_route() instead
        return None

    def check_route(self, route: APIRoute) -> Optional[str]:
        """Override to invert the allowed_unsecured logic."""
        if _skip_all(route):
            return None
        # Only check routes that ARE in allowed_unsecured
        if route.path not in self.allowed_unsecured:
            return None
        unsafe = [m for m in route.methods if m not in self.safe]
        if unsafe:
            return f"{','.join(route.methods)} {route.path} exposes unsafe method(s) without security (allowed_unsecured)"
        return None


class CORSMisconfigurationCheck(SingleRunMixin, SecurityCheck):
    # OWASP: API8 (Security Misconfiguration)
    CATEGORY = "config"
    OWASP = ["API8"]
    def __init__(self, allow_wildcards: bool = False) -> None:
        super().__init__()
        self.allow_wildcards = allow_wildcards

    # Route-level invocation now no-op; detection done in app_check
    def check_route(self, route: APIRoute) -> Optional[str]:
        return None

    def app_check(self, app: FastAPI) -> Optional[str]:
        if not self.should_run():  # single-run guard
            return None
        issues: List[str] = []

        def is_wild(v: Any) -> bool:
            return v == "*" or v == ["*"]

        for mw in getattr(app, "user_middleware", []):
            if mw.cls is CORSMiddleware:
                opt = getattr(mw, "options", None) or getattr(mw, "kwargs", {}) or {}
                origins = opt.get("allow_origins")
                methods = opt.get("allow_methods")
                headers = opt.get("allow_headers")
                credentials = opt.get("allow_credentials")
                if not self.allow_wildcards:
                    if is_wild(origins):
                        issues.append("allow_origins='*'")
                    if is_wild(methods):
                        issues.append("allow_methods='*'")
                    if is_wild(headers):
                        issues.append("allow_headers='*'")
                if credentials and is_wild(origins):
                    issues.append("credentials allowed with wildcard origins")
        if issues:
            return "CORS misconfiguration: " + ", ".join(issues)
        return None


class DebugModeCheck(SingleRunMixin, SecurityCheck):
    # OWASP: API8 (Security Misconfiguration)
    CATEGORY = "config"
    OWASP = ["API8"]
    def __init__(self) -> None:
        super().__init__()

    def check_route(self, route: APIRoute) -> Optional[str]:
        return None

    def app_check(self, app: FastAPI) -> Optional[str]:
        if not self.should_run():
            return None
        if getattr(app, "debug", False):
            return "Application running in debug mode"
        return None


class BodyModelEnforcementCheck(RouteCheck):
    # OWASP: API6 (Mass Assignment), API3 (Excessive Data Exposure)
    CATEGORY = "schema"
    OWASP = ["API6", "API3"]
    def __init__(self, enforce_methods: Optional[Iterable[str]] = None, allowed_unsecured: Optional[Sequence[str]] = None) -> None:
        super().__init__(allowed_unsecured)
        self.methods = {m.upper() for m in (enforce_methods or ["POST", "PUT", "PATCH"])}

    def _analyze(self, route: APIRoute) -> Optional[str]:
        if not (self.methods & route.methods):
            return None
        raw_names: List[str] = []
        for p in route.dependant.body_params:  # type: ignore[attr-defined]
            t = getattr(p, "type_", None)
            if t in (None, UploadFile, bytes):
                continue
            origin = getattr(t, "__origin__", None)
            if t in (dict, list, Any) or origin in (dict, list):
                raw_names.append(p.name)
        if raw_names:
            return f"{','.join(route.methods)} {route.path} uses non-model raw body param(s): {','.join(raw_names)}"
        return None


class PaginationEnforcementCheck(RouteCheck):
    # OWASP: API4 (Unrestricted Resource Consumption)
    CATEGORY = "performance"
    OWASP = ["API4"]
    def __init__(self, pagination_param_names: Optional[Iterable[str]] = None, allowed_unsecured: Optional[Sequence[str]] = None) -> None:
        super().__init__(allowed_unsecured)
        self.pagination_params = set(pagination_param_names or ["limit", "offset", "page", "page_size"])

    def _analyze(self, route: APIRoute) -> Optional[str]:
        if "GET" not in route.methods:
            return None
        # Prefer the return annotation; fall back to response_model so routes
        # declared via response_model=List[X] without an annotation are covered.
        ann = route.endpoint.__annotations__.get("return")
        if ann is None:
            ann = route.response_model
        if ann is None:
            return None
        origin = get_origin(ann)
        try:
            is_list = origin in (list, List) or ann in (list, List)
        except TypeError:
            is_list = False
        if not is_list:
            return None
        query_names = {p.name for p in route.dependant.query_params}
        if self.pagination_params.isdisjoint(query_names):
            return f"GET {route.path} returns a collection without pagination params ({'/'.join(sorted(self.pagination_params))})"
        return None


class WildcardPathCheck(RouteCheck):
    # OWASP: API5 (Broken Function Level Authorization), API3 (Excessive Data Exposure)
    CATEGORY = "routing"
    OWASP = ["API5", "API3"]
    def _analyze(self, route: APIRoute) -> Optional[str]:
        if ":path}" in route.path:
            return f"{','.join(route.methods)} {route.path} uses broad wildcard path parameter (:path)"
        return None


class SensitiveFieldExposureCheck(RouteCheck):
    # OWASP: API3 (Excessive Data Exposure)
    CATEGORY = "data_exposure"
    OWASP = ["API3"]
    def _analyze(self, route: APIRoute) -> Optional[str]:
        model = route.response_model
        if not (model and isinstance(model, type)):
            return None
        try:
            if not issubclass(model, BaseModel):  # type: ignore[arg-type]
                return None
        except TypeError:  # pragma: no cover - defensive against exotic metaclasses
            return None
        field_names = (
            list(getattr(model, "model_fields", {}).keys())
            if hasattr(model, "model_fields")
            else list(getattr(model, "__fields__", {}).keys())
        )
        hits = {
            name_lower
            for name in field_names
            if (name_lower := name.lower())
            and any(sub in name_lower for sub in SUSPICIOUS_FIELD_PARTS)
        }
        if hits:
            return f"{','.join(route.methods)} {route.path} response_model exposes potentially sensitive fields: {','.join(sorted(hits))}"
        return None


class ReturnTypeAnnotationCheck(RouteCheck):
    # OWASP: API3 (Excessive Data Exposure)
    CATEGORY = "schema"
    OWASP = ["API3"]
    def _analyze(self, route: APIRoute) -> Optional[str]:
        if route.response_model is not None:
            return None
        ann = route.endpoint.__annotations__.get("return") if hasattr(route.endpoint, "__annotations__") else None
        if ann is None:
            return f"{','.join(route.methods)} {route.path} has neither response_model nor return type annotation"
        return None


class SensitiveQueryParamCheck(RouteCheck):
    # OWASP: API3 (Excessive Data Exposure)
    CATEGORY = "data_exposure"
    OWASP = ["API3"]
    def __init__(self, allowed_unsecured: Optional[Sequence[str]] = None, allowlist: Optional[Iterable[str]] = None) -> None:
        super().__init__(allowed_unsecured)
        self.allowlist = {a.lower() for a in (allowlist or [])}

    def _analyze(self, route: APIRoute) -> Optional[str]:
        hits = []
        for qp in route.dependant.query_params:
            name_l = qp.name.lower()
            if name_l in self.allowlist:
                continue
            if any(sub in name_l for sub in SUSPICIOUS_QUERY_PARTS):
                hits.append(qp.name)
        if hits:
            return f"{','.join(route.methods)} {route.path} exposes potentially sensitive data via query params: {','.join(sorted(set(hits)))}"
        return None


class HTTPSRedirectMiddlewareCheck(SingleRunMixin, SecurityCheck):
    # OWASP: API8 (Security Misconfiguration)
    CATEGORY = "config"
    OWASP = ["API8"]
    def __init__(self) -> None:
        super().__init__()

    def check_route(self, route: APIRoute) -> Optional[str]:
        return None

    def app_check(self, app: FastAPI) -> Optional[str]:
        if not self.should_run():
            return None
        for mw in getattr(app, "user_middleware", []):
            if mw.cls is HTTPSRedirectMiddleware:
                return None
        return "HTTPS redirect middleware not configured (consider HTTPSRedirectMiddleware or upstream TLS enforcement)"


class TrustedHostMiddlewareCheck(SingleRunMixin, SecurityCheck):
    # OWASP: API8 (Security Misconfiguration)
    CATEGORY = "config"
    OWASP = ["API8"]
    def __init__(self) -> None:
        super().__init__()

    def check_route(self, route: APIRoute) -> Optional[str]:
        return None

    def app_check(self, app: FastAPI) -> Optional[str]:
        if not self.should_run():
            return None
        for mw in getattr(app, "user_middleware", []):
            if mw.cls is TrustedHostMiddleware:
                return None
        return "TrustedHostMiddleware not configured (consider restricting allowed hosts)"


class RateLimitingPresenceCheck(SingleRunMixin, SecurityCheck):
    # OWASP: API4 (Unrestricted Resource Consumption)
    CATEGORY = "performance"
    OWASP = ["API4"]
    def __init__(self) -> None:
        super().__init__()

    def check_route(self, route: APIRoute) -> Optional[str]:
        return None

    def app_check(self, app: FastAPI) -> Optional[str]:
        if not self.should_run():
            return None
        for mw in getattr(app, "user_middleware", []):
            name = mw.cls.__name__.lower()
            full = str(mw.cls).lower()
            if any(k in name or k in full for k in RATE_LIMIT_KEYWORDS):
                return None
        return "No apparent rate limiting middleware detected (consider adding to mitigate abuse)"


# ---------------- Additional (non-OWASP-top10-specific) Checks ----------------
class DangerousMethodExposureCheck(RouteCheck):
    """Flag usage of rarely needed and potentially unsafe HTTP methods (TRACE/CONNECT).
    These methods are almost never required in public APIs and can aid in fingerprinting or tunneling.
    """
    CATEGORY = "http_methods"
    OWASP: List[str] = []  # informational
    DANGEROUS = {"TRACE", "CONNECT"}

    def _analyze(self, route: APIRoute) -> Optional[str]:
        exposed = self.DANGEROUS & route.methods
        if exposed:
            return f"{','.join(route.methods)} {route.path} exposes dangerous HTTP method(s): {','.join(sorted(exposed))}"
        return None


class SSRFParameterCheck(RouteCheck):
    """Detect query parameters that commonly indicate potential SSRF vectors (e.g. 'url', 'uri', 'target').
    Purely heuristic – encourages explicit allowlists or validation for remote resource fetches.
    """
    CATEGORY = "ssrf"
    OWASP: List[str] = []  # informational
    RISKY = {"url", "uri", "target", "endpoint", "callback"}

    def __init__(self, allowed_unsecured: Optional[Sequence[str]] = None, allowlist: Optional[Iterable[str]] = None) -> None:
        super().__init__(allowed_unsecured)
        self.allowlist = {a.lower() for a in (allowlist or [])}

    def _analyze(self, route: APIRoute) -> Optional[str]:
        hits: List[str] = []
        for qp in route.dependant.query_params:
            name_l = qp.name.lower()
            if name_l in self.allowlist:
                continue
            if name_l in self.RISKY:
                hits.append(qp.name)
        if hits:
            return f"{','.join(route.methods)} {route.path} contains potential SSRF parameter(s): {','.join(sorted(set(hits)))}"
        return None


class AdminRouteOpenCheck(RouteCheck):
    """Flag admin-related routes that appear to lack any dependency-based security.
    Heuristic: path contains '/admin' and dependant.dependencies is empty.
    """
    CATEGORY = "auth"
    OWASP: List[str] = []  # informational

    def _analyze(self, route: APIRoute) -> Optional[str]:
        if "/admin" in route.path.lower():
            if not getattr(route.dependant, "dependencies", []):
                return f"{','.join(route.methods)} {route.path} admin route without explicit security dependencies"
        return None


# ---------------- Recommended preset utilities ----------------

def recommended_checks(
    *,
    allowed_unsecured: Optional[Sequence[str]] = None,
    extra_dependencies: Optional[Union[List[Any], Set[Any]]] = None,
) -> List[SecurityCheck]:
    """Return a curated set of checks considered a strong default.

    Core checks (always included):
    - Authentication & authorization enforcement
    - Response model & body validation (prevents data leaks & mass assignment)
    - Sensitive data exposure detection
    - CORS & debug mode misconfiguration

    Optional checks (infrastructure-level or heuristic-based rules such as
    HTTPS redirect, rate limiting, SSRF params, admin routes) are not included;
    add them explicitly when relevant to your deployment.
    """
    allowed_unsecured = allowed_unsecured or DEFAULT_ALLOWED_UNSECURED

    # High-value core checks: critical security issues with low false positives
    core: List[SecurityCheck] = [
        # Authentication & Authorization (OWASP API2, API5)
        DependencySecurityCheck(allowed_unsecured=allowed_unsecured, extra_dependencies=extra_dependencies),
        UnsecuredAllowedMethodsCheck(allowed_unsecured=allowed_unsecured),

        # Data Exposure Prevention (OWASP API3, API6)
        ResponseModelSecurityCheck(allowed_unsecured=allowed_unsecured),
        BodyModelEnforcementCheck(allowed_unsecured=allowed_unsecured),
        SensitiveFieldExposureCheck(allowed_unsecured=allowed_unsecured),
        SensitiveQueryParamCheck(allowed_unsecured=allowed_unsecured),

        # Resource Consumption (OWASP API4)
        PaginationEnforcementCheck(allowed_unsecured=allowed_unsecured),

        # Configuration Issues (OWASP API8)
        CORSMisconfigurationCheck(),
        DebugModeCheck(),
    ]
    return core
