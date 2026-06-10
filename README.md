# FastAPI Safeguard

FastAPI Safeguard scans your routes at startup and fails on security issues — missing auth, exposed sensitive fields, CORS misconfigs, debug mode left on — before your application serves a single request.

Add it once. New unsecured routes fail the build.

**Requires Python 3.9+**

## Highlights

- **Baseline lock file** — Like `package-lock.json`, but for security findings. Accept existing tech debt; fail only on *new* regressions. Every change to security posture shows up as a diff in `security_baseline.json`.
- **Fail fast** — Startup exits non-zero if new issues are found. Works as a CI gate out of the box.
- **CLI & test helper** — `fastapi-safeguard check app.main:app` scans without running your app; `assert_safeguard(app)` does the same inside pytest.
- **Explicit intent** — Mark public endpoints with `@open_route`. Everything else is expected to have auth.
- **Zero runtime overhead** — Checks run once at startup, never per-request.
- **Extensible** — 9 core checks, 8 optional, and a simple base class for your own rules.

## Quick Start

```bash
pip install fastapi-safeguard
```

```python
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi_safeguard import FastAPISafeguard, open_route

app = FastAPI(lifespan=FastAPISafeguard.recommended().lifespan())
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@open_route  # Explicitly marked as public — no warning
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/legacy-endpoint")  # Forgot to add auth — will be flagged!
async def legacy():
    return {"data": "visible to all"}

@app.get("/protected")
async def protected(token: str = Depends(oauth2_scheme)):
    return {"data": "secret"}
```

**First run** — accept current state and create a baseline:
```bash
SECURITY_BASELINE_UPDATE=1 uvicorn main:app --reload
```

This creates `security_baseline.json`:
```json
{
  "schema_version": 1,
  "generated_at": "2025-01-15T10:30:00Z",
  "accepted_findings": [
    "GET /legacy-endpoint has no accepted security dependency"
  ],
  "checks_count": 9
}
```

**Subsequent runs** fail only on *new* findings:
```bash
uvicorn main:app --reload
```

✅ Startup succeeds if findings match baseline
❌ Startup fails if new security issues are detected

---

## Table of Contents

- [Highlights](#highlights)
- [Quick Start](#quick-start)
- [Why](#why)
- [Configuration](#configuration)
- [Baseline Workflow](#baseline-workflow)
- [CLI](#cli)
- [Using in Tests](#using-in-tests)
- [Decorators](#decorators)
- [Security Checks Reference](#security-checks-reference)
- [OWASP API Top 10 Coverage](#owasp-api-top-10-coverage)
- [Writing a Custom Check](#writing-a-custom-check)
- [CI/CD Integration](#cicd-integration)
- [Startup Output Examples](#startup-output-examples)
- [Environment Variables](#environment-variables)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Why

FastAPI treats every new endpoint as open by default. That's fine for prototyping. In production it means a forgotten `Depends()` silently exposes a route — and nobody notices until a pentest or an incident.

The failure modes are mundane and common:
- A mutation endpoint ships without auth.
- A response model is missing, so the ORM object serializes more fields than intended.
- A list endpoint has no pagination, and a single request dumps the whole table.
- CORS is set to `*` with `allow_credentials=True`.
- Debug mode is on in a staging environment that shares production data.

FastAPI Safeguard catches these at startup. It enforces that openness is intentional: unsecured routes must either carry a recognized auth dependency, be decorated with `@open_route`, or be explicitly accepted in the security baseline.

The baseline is the key part. Once you accept a finding, it's pinned. Future runs fail only on *new* findings. Adding an unsecured route shows up as a diff in the baseline file. Fixing one shows up as a removal. Security posture becomes reviewable in pull requests.

---

## Configuration

### Custom auth dependencies

Built-in schemes (`OAuth2*`, `HTTPBasic`, `HTTPBearer`, `APIKey*`) are recognized even when nested inside your own dependencies — the common `Depends(get_current_user)` pattern, where `get_current_user` itself depends on `oauth2_scheme`, works out of the box. Only fully custom auth needs registering so the check recognizes it:

```python
from fastapi_safeguard import (
    FastAPISafeguard,
    DependencySecurityCheck,
    ResponseModelSecurityCheck,
    PaginationEnforcementCheck,
)

safeguard = FastAPISafeguard(checks=[
    DependencySecurityCheck(
        allowed_unsecured=["/openapi.json", "/docs", "/health"],
        extra_dependencies=[my_custom_auth_dep],
    ),
    ResponseModelSecurityCheck(),
    PaginationEnforcementCheck(),
])
app = FastAPI(lifespan=safeguard.lifespan())
```

### Adding optional checks

The `recommended()` preset includes only the 9 core checks. Add optional ones explicitly:

```python
from fastapi_safeguard import FastAPISafeguard, HTTPSRedirectMiddlewareCheck

safeguard = FastAPISafeguard(checks=[
    *FastAPISafeguard.recommended().checks,
    HTTPSRedirectMiddlewareCheck(),
])
app = FastAPI(lifespan=safeguard.lifespan())
```

### Existing lifespan

If your app already has a lifespan, call `run_checks()` directly:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi_safeguard import FastAPISafeguard

safeguard = FastAPISafeguard.recommended()

@asynccontextmanager
async def lifespan(app: FastAPI):
    safeguard.run_checks(app)
    db = await setup_database()
    yield
    await db.close()

app = FastAPI(lifespan=lifespan)
```

---

## Baseline Workflow

### Mental model

| You do this | The plugin does this |
|---|---|
| Run app with no baseline | Scans, prints findings, exits with error |
| Set `SECURITY_BASELINE_UPDATE=1` | Writes a snapshot of current findings |
| Add a new unsecured route | Fails: finding not in baseline |
| Fix a finding | Old entry stays in file until you refresh |
| Refresh with update flag | Prunes resolved entries, accepts new ones |

### Typical flow

```bash
# 1. First scan (will fail if there are findings)
uvicorn main:app --reload

# 2. Accept the current snapshot
SECURITY_BASELINE_UPDATE=1 uvicorn main:app --reload

# 3. Subsequent runs fail only on NEW findings
uvicorn main:app --reload

# 4. After fixing issues, prune resolved ones
SECURITY_BASELINE_UPDATE=1 uvicorn main:app --reload
```

You can also set `update_baseline=True` in code: `FastAPISafeguard(..., update_baseline=True)`.

### Reviewing baseline diffs

- **Added lines** = new risks being accepted.
- **Removed lines** = fixed issues.
- **No change** = security posture unchanged.

If a diff adds `"GET /internal-metrics has no accepted security dependency"`, ask: was that endpoint meant to be open? If not, fix it instead of accepting.

> Commit the baseline to git. Treat it like a contract — change it only when you're consciously accepting or removing risk.

---

## CLI

Scan an app without wiring the lifespan or starting a server — the app is only imported, never run. Use it when you want the security gate in CI (or locally) instead of, or in addition to, the startup gate:

```bash
fastapi-safeguard check app.main:app                  # attribute defaults to :app
fastapi-safeguard check app.main:create_app --factory # app factories
fastapi-safeguard check main:app --update-baseline    # accept current findings
```

| Flag | Purpose |
|---|---|
| `--app-dir DIR` | Directory to add to the import path (default: `.`) |
| `--baseline PATH` | Baseline file (default: `SECURITY_BASELINE_PATH` or `security_baseline.json`) |
| `--update-baseline` | Accept current findings and write/refresh the baseline |
| `--factory` | Call the imported attribute to get the app |

Exit codes: `0` pass, `1` new findings, `2` app could not be loaded. Also available as `python -m fastapi_safeguard`.

---

## Using in Tests

`assert_safeguard(app)` runs the same scan as a plain assertion — no `SystemExit`, no stdout noise, baseline respected but never written:

```python
from fastapi_safeguard import assert_safeguard
from app.main import app

def test_security_posture():
    assert_safeguard(app)  # raises AssertionError listing any new findings
```

It returns a `ScanResult` for further assertions, and accepts `checks=[...]`, `baseline_path=...`, or a preconfigured `safeguard=` for custom setups.

---

## Decorators

| Decorator | Effect | Other checks still run? | Use case |
|---|---|---|---|
| `@open_route` | Skips auth check only | Yes | Public status pages, health checks |
| `@disable_security_checks` | Skips all checks | No | Internal webhooks, exotic routes |

```python
from fastapi_safeguard import open_route, disable_security_checks

@open_route
@app.get("/public-catalog")
async def catalog():
    return {"items": []}

@disable_security_checks
@app.post("/internal-webhook")
async def webhook():
    return {"ok": True}
```

Decorator placement above or below `@app.get(...)` doesn't matter — both attach to the same function.

### Other ways to accept findings

| Strategy | When to use |
|---|---|
| `@open_route` | Intentionally public endpoint. Other checks still apply. |
| `@disable_security_checks` | Exceptional route. Use sparingly. |
| Baseline accept | Known tech debt. Persists until pruned. |
| Omit check from config | The check isn't relevant to your org. |

---

## Security Checks Reference

### Core checks (in `recommended()`)

| Check | OWASP | What it catches | Example finding |
|---|---|---|---|
| `DependencySecurityCheck` | API2, API5 | Routes with no auth dependency | `GET /items has no accepted security dependency` |
| `UnsecuredAllowedMethodsCheck` | API5 | Unsafe methods on declared-open paths | `POST /status exposes unsafe method(s) without security` |
| `ResponseModelSecurityCheck` | API3 | Mutation endpoints missing `response_model` | `POST /users missing response_model for unsafe method(s)` |
| `BodyModelEnforcementCheck` | API3, API6 | Raw `dict`/`list` body params (mass assignment risk) | `PATCH /users uses non-model raw body param(s): payload` |
| `SensitiveFieldExposureCheck` | API3 | Response fields named `password`, `token`, `secret`, etc. | `GET /auth/me response_model exposes potentially sensitive fields: password` |
| `SensitiveQueryParamCheck` | API3 | Sensitive data in query strings | `GET /login exposes potentially sensitive data via query params: token` |
| `PaginationEnforcementCheck` | API4 | List endpoints without `limit`/`skip` params | `GET /reports returns a collection without pagination params` |
| `CORSMisconfigurationCheck` | API8 | Wildcard origins + `allow_credentials=True` | `CORS misconfiguration: allow_origins='*', credentials allowed` |
| `DebugModeCheck` | API8 | `app.debug = True` | `Application running in debug mode` |

### Optional checks

These have higher false positive rates or are typically handled at the infrastructure level:

| Check | OWASP | What it catches | Why not default |
|---|---|---|---|
| `HTTPSRedirectMiddlewareCheck` | API8 | Missing HTTPS redirect middleware | Usually handled by load balancer |
| `TrustedHostMiddlewareCheck` | API8 | Missing host header protection | Usually handled upstream |
| `RateLimitingPresenceCheck` | API4 | No rate-limiting middleware detected | Often external (API gateway); weak heuristic |
| `ReturnTypeAnnotationCheck` | API3 | Missing return type annotations | Noisy; not a security issue |
| `WildcardPathCheck` | API3, API5 | Broad `{path:path}` params | Legitimate use cases exist |
| `DangerousMethodExposureCheck` | — | TRACE/CONNECT methods | Rarely exposed via FastAPI |
| `SSRFParameterCheck` | — | Params named `url`, `uri`, `target`, etc. | Too many false positives |
| `AdminRouteOpenCheck` | — | Unprotected `/admin*` routes | Heuristic based on path naming |

---

## OWASP API Top 10 Coverage

| OWASP | Risk | Core checks | Optional checks |
|---|---|---|---|
| API2 | Broken Authentication | DependencySecurityCheck | — |
| API3 | Excessive Data Exposure | ResponseModelSecurityCheck, BodyModelEnforcementCheck, SensitiveFieldExposureCheck, SensitiveQueryParamCheck | ReturnTypeAnnotationCheck, WildcardPathCheck |
| API4 | Unrestricted Resource Consumption | PaginationEnforcementCheck | RateLimitingPresenceCheck |
| API5 | Broken Function Level Authorization | DependencySecurityCheck, UnsecuredAllowedMethodsCheck | WildcardPathCheck |
| API6 | Mass Assignment | BodyModelEnforcementCheck | — |
| API8 | Security Misconfiguration | CORSMisconfigurationCheck, DebugModeCheck | HTTPSRedirectMiddlewareCheck, TrustedHostMiddlewareCheck |

---

## Writing a Custom Check

```python
from typing import Optional

from fastapi.routing import APIRoute
from fastapi_safeguard import FastAPISafeguard, RouteCheck

class MaxPathDepthCheck(RouteCheck):
    """Flag routes with deeply nested paths (more than 4 segments)."""
    CATEGORY = "routing"
    OWASP = ["API5"]

    def _analyze(self, route: APIRoute) -> Optional[str]:
        depth = len(route.path.strip("/").split("/"))
        if depth > 4:
            return f"{route.path} has {depth} path segments (max 4)"
        return None

safeguard = FastAPISafeguard(checks=[
    *FastAPISafeguard.recommended().checks,
    MaxPathDepthCheck(),
])
```

Rules:
- Subclass `RouteCheck` (not `SecurityCheck`) and implement `_analyze` — the base class handles `@disable_security_checks` and `allowed_unsecured` paths for you.
- Return `None` when compliant, a string when not.
- Finding strings must be **stable and deterministic** — the baseline matches on exact text.
- Set `CATEGORY` and `OWASP` for the summary output.

---

## CI/CD Integration

```yaml
# GitHub Actions — scans the app without running it; exits 1 on new findings
- name: Security route check
  run: |
    pip install fastapi-safeguard
    fastapi-safeguard check main:app
```

To accept a new baseline in a controlled PR:

```yaml
- name: Update baseline (manual trigger)
  if: github.event_name == 'workflow_dispatch'
  run: |
    fastapi-safeguard check main:app --update-baseline
    git add security_baseline.json
    git commit -m "chore: update security baseline" || echo "No changes"
```

---

## Startup Output Examples

**New findings (startup fails):**
```
Category Summary:
Category            Total   New  Accepted  OWASP
auth                   1     1         0   API2/API5
schema                 2     1         1   API3/API6

❌ Security check failed: new findings detected (not in baseline):
  + POST /users missing response_model for unsafe method(s)
  + GET /items has no accepted security dependency

To accept current findings run with SECURITY_BASELINE_UPDATE=1 or set update_baseline=True.
```

**All findings accepted:**
```
Category Summary:
Category            Total   New  Accepted  OWASP
auth                   1     0         1   API2/API5
schema                 2     0         2   API3/API6

✅ All security findings match accepted baseline (3 accepted).
```

**Clean:**
```
✅ All security checks passed (0 findings, 12 routes, 14 checks).
```

---

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `SECURITY_BASELINE_PATH` | Path to baseline file | `security_baseline.json` |
| `SECURITY_BASELINE_UPDATE` | Set to `1` to accept current findings and write/refresh the baseline | `0` |

---

## FAQ

**Does this replace runtime auth?**
No. It catches common omissions at startup. You still need runtime authorization.

**Will it block production if I forget to update the baseline?**
Yes, intentionally. Incorporate baseline updates into the PR that introduces the change.

**Can I have different baselines per environment?**
Yes — set `SECURITY_BASELINE_PATH` to point at different files.

**What if I rename a route?**
The old finding becomes resolved; refresh with `SECURITY_BASELINE_UPDATE=1` to prune it.

**How do I disable a specific check?**
Omit it from the `checks=[...]` list when constructing `FastAPISafeguard`.

---

## Roadmap

- Severity levels (warn vs error) and downgrade mechanism
- JSON output for machine parsing (SARIF-compatible)
- Category include/exclude filters via env vars
- `@warn_only` decorator for non-fatal findings on selected routes
- Scope / role enforcement scaffold

---

## Contributing

1. Fork & branch (`feat/my-improvement`).
2. Keep PRs focused — one conceptual change per PR.
3. Update README examples if behavior changes.
4. Prefer small, composable checks over multi-purpose ones.
5. Finding strings must be deterministic — no timestamps or randomness.

---

## License

MIT — see [LICENSE](LICENSE).
