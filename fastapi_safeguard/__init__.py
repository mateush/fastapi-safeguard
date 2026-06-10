"""FastAPI Safeguard - Security auditing framework for FastAPI applications.

This package provides a comprehensive security checking system for FastAPI
applications, detecting common vulnerabilities aligned with OWASP API Security
Top 10.

The framework includes:
- Authentication and authorization enforcement
- Data exposure prevention
- Resource consumption protection
- Security misconfiguration detection
- Baseline management for accepted findings
- A CLI (``fastapi-safeguard check app.main:app``) and a test helper
  (:func:`assert_safeguard`) for CI pipelines and test suites

Example:
    >>> from fastapi import FastAPI
    >>> from fastapi_safeguard import FastAPISafeguard
    >>>
    >>> app = FastAPI(lifespan=FastAPISafeguard.recommended().lifespan())

For more information, see the documentation at https://github.com/mateush/fastapi-safeguard
"""
from .checks import (
    AdminRouteOpenCheck,
    BodyModelEnforcementCheck,
    CORSMisconfigurationCheck,
    DangerousMethodExposureCheck,
    DebugModeCheck,
    DependencySecurityCheck,
    HTTPSRedirectMiddlewareCheck,
    PaginationEnforcementCheck,
    RateLimitingPresenceCheck,
    ResponseModelSecurityCheck,
    ReturnTypeAnnotationCheck,
    RouteCheck,
    SecurityCheck,
    SensitiveFieldExposureCheck,
    SensitiveQueryParamCheck,
    SSRFParameterCheck,
    TrustedHostMiddlewareCheck,
    UnsecuredAllowedMethodsCheck,
    WildcardPathCheck,
    disable_security_checks,
    open_route,
    recommended_checks,
)
from .scanner import FastAPISafeguard, Finding, ScanResult
from .testing import assert_safeguard

__version__ = "0.2.0"

__all__ = [
    # Decorators
    "open_route",
    "disable_security_checks",
    # Core types
    "FastAPISafeguard",
    "SecurityCheck",
    "RouteCheck",
    "Finding",
    "ScanResult",
    # Checks
    "DependencySecurityCheck",
    "ResponseModelSecurityCheck",
    "UnsecuredAllowedMethodsCheck",
    "CORSMisconfigurationCheck",
    "DebugModeCheck",
    "BodyModelEnforcementCheck",
    "PaginationEnforcementCheck",
    "WildcardPathCheck",
    "SensitiveFieldExposureCheck",
    "ReturnTypeAnnotationCheck",
    "SensitiveQueryParamCheck",
    "HTTPSRedirectMiddlewareCheck",
    "TrustedHostMiddlewareCheck",
    "RateLimitingPresenceCheck",
    "DangerousMethodExposureCheck",
    "SSRFParameterCheck",
    "AdminRouteOpenCheck",
    # Preset helpers
    "recommended_checks",
    # Test-suite integration
    "assert_safeguard",
]
