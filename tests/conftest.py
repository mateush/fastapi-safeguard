import os
import json
import contextlib
import pytest
import sys
from pathlib import Path
from fastapi import FastAPI
from pydantic import BaseModel

# Ensure project root on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from fastapi_safeguard import (
    FastAPISafeguard,
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
    HTTPSRedirectMiddlewareCheck,
    TrustedHostMiddlewareCheck,
    RateLimitingPresenceCheck,
    DebugModeCheck,
    open_route,
    disable_security_checks,
)

class DummyDep:
    def __call__(self):
        return True

@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    # Ensure no global baseline configuration interferes
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

@pytest.fixture
def tmp_baseline(tmp_path):
    return tmp_path / "security_baseline.json"

@pytest.fixture
def BookModel():
    class Book(BaseModel):
        title: str
        rating: float
    return Book

@pytest.fixture
def SensitiveModel():
    class User(BaseModel):
        username: str
        password: str
    return User
