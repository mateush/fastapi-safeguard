# FastAPI Safeguard

**A zero-runtime-overhead security linting and hardening plugin for FastAPI applications.**

Catches security issues at startup—before serving traffic. Prevents accidental exposure of unsecured endpoints through baseline lock files and explicit intent markers.

> *"Errors should never pass silently." — Unless explicitly accepted and version‑locked in your baseline.*

## ✨ Highlights

- 🔒 **Baseline Lock File** — Like `package-lock.json` but for security findings. Accept current tech debt; fail only on *new* regressions.
- 🚨 **Fail Fast** — Detects missing auth, exposed sensitive fields, CORS misconfigs, debug mode, and more at startup.
- 🎯 **Explicit Intent** — `@open_route` for public endpoints, `@disable_security_checks` for exceptions. No accidental exposures.
- 📊 **OWASP API Top 10** — Covers API2, API3, API4, API5, API6, API8 with 9 high-value core checks.
- ⚡ **Zero Runtime Overhead** — All checks run once at startup, not per-request.
- 🔧 **Extensible** — Write custom checks, configure per-org rules. 8+ optional checks available.

## 🚀 Quick Start

```bash
pip install fastapi-safeguard
```

```python
from fastapi import FastAPI, Depends
from fastapi.security import OAuth2PasswordBearer
from fastapi_safeguard import FastAPISafeguard, open_route

app = FastAPI(lifespan=FastAPISafeguard.recommended().lifespan())
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

@open_route  # Explicitly marked as public - no warning
@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.get("/legacy-endpoint")  # Forgot to add auth - will be flagged!
async def legacy():
    return {"data": "visible to all"}

@app.get("/protected")
async def protected(token: str = Depends(oauth2_scheme)):
    return {"data": "secret"}
```

**First run** - Accept current state and create baseline:
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

**Subsequent runs** - Fail only on *new* findings:
```bash
uvicorn main:app --reload
```

✅ Startup succeeds if findings match baseline
❌ Startup fails if new security issues detected

---
## Table of Contents
- [Highlights](#-highlights)
- [Quick Start](#-quick-start)
- [Why](#why)
- [Key Features](#key-features)
- [Installation & Usage](#installation--usage)
- [Baseline (Lock) File Workflow](#baseline-lock-file-workflow)
- [Decorators](#decorators)
- [Configuration & Extensibility](#configuration--extensibility)
- [Writing a Custom Check](#writing-a-custom-check)
- [Startup Output Examples](#startup-output-examples)
- [Security Checks Reference](#security-checks-reference)
- [OWASP API Top 10 Coverage](#owasp-api-top-10-coverage)
- [CI/CD Integration](#cicd-integration)
- [Environment Variables](#environment-variables)
- [FAQ](#faq)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [Security Policy](#security-policy)
- [License](#license)

---
## Why
Securing an API is more than adding a single dependency or middleware. Repeated patterns cause recurring mistakes:
- Accidentally shipping a mutating endpoint with no auth.
- Returning whole ORM objects / dict bodies without explicit response models.
- Forgetting pagination on collection endpoints.
- Permissive CORS + credentials.
- Leaking sensitive fields or query parameters.
- Running with debug settings in production.
- Accepting past risk *intentionally*, but wanting to fail only on **new** regressions.

### Philosophy: Security is a First‑Class Feature (Not an Optional Afterthought)
FastAPI (like most Python web frameworks) treats every new endpoint as **open by default** unless you *remember* to add a dependency, middleware, or explicit security layer. That default is great for exploration—but dangerous for serious applications where an unguarded route can leak data, expose internal mechanics, or become a lateral movement foothold.

This project flips that implicit default:
- Closed (or at least scrutinized) is the baseline; openness must be **intentional and explicit**.
- Any route lacking an accepted security dependency is treated as a potential regression unless you *consciously* accept it into the security baseline.
- Public endpoints are allowed—but must be justified (e.g. decorated with `@open_route` and/or accepted in the lock file).

### Open by Accident vs. Open on Purpose
Implicitly exposed endpoints are a common source of:
- “Why is staging data visible in prod?” moments
- Shadow APIs (endpoints added in a refactor, never reviewed)
- Drift between expected security posture and reality

By making openness explicit, you reduce the cognitive load during review: **it’s harder to smuggle in a silent, unsecured path.**

### The Baseline Lock File as a PR Gate
This mirrors how dependency lock files work: *only deliberate changes* modify the security posture. Accidental exposures surface immediately instead of months later in a pentest report.

### Secure‑by‑Default Principles Enforced Here
- ✔️ **Visibility:** Every risk is enumerated at startup—no silent failures.
- ✔️ **Friction for risk; low friction for safety:** Secure endpoints pass quietly; new exposures demand action.
- ✔️ **Deterministic strings:** Findings are stable -> easy to review / baseline.
- ✔️ **Intent markers:** `@open_route` vs `@disable_security_checks` communicate design decisions.
- ✔️ **Fail fast:** CI / local dev breaks on unaccepted regressions.
- ✔️ **Scalable:** Add or drop checks without changing application logic.

If your application handles *anything* sensitive (users, tokens, internal metadata, usage analytics), default-open behavior is not good enough. Treat security posture the way you treat schema migrations or dependency bumps: **explicit, reviewed, versioned.**

---
## Key Features

### 🔒 Baseline / Lock File (Core Feature)
Security findings as version-controlled artifacts:
- **Accept existing tech debt** while blocking new vulnerabilities
- **Fail only on new findings** — stabilizes CI without compromising security
- **Diff-able PR changes** — every new risk requires explicit approval
- Automatically updated with `SECURITY_BASELINE_UPDATE=1`

### 🎯 Explicit Intent Markers
- `@open_route` — marks intentionally public endpoints (auth check skipped, others still enforced)
- `@disable_security_checks` — disables all checks for exceptional routes (use sparingly)

### 📊 High-Value Security Checks
9 core checks covering critical security issues with low false positives:
- Missing auth dependencies (OWASP API2, API5)
- Response model & body validation (OWASP API3, API6)
- Sensitive field/query param exposure (OWASP API3)
- Unbounded pagination (OWASP API4)
- CORS & debug mode misconfigurations (OWASP API8)
- See [Security Checks Reference](#security-checks-reference) for complete list

### 🔧 Extensible & Configurable
- Add/remove checks via constructor
- Write custom organizational rules
- Enable/disable heuristic checks
- Each check self-identifies with `CATEGORY` and `OWASP` codes

---
## Installation & Usage

### Installation
```bash
pip install fastapi-safeguard
```

### Basic Usage
```python
from fastapi import FastAPI
from fastapi_safeguard import FastAPISafeguard

app = FastAPI(lifespan=FastAPISafeguard.recommended().lifespan())
```

### With Additional Heuristic Checks
The default includes only high-value checks. Add optional heuristic checks manually if needed:
```python
from fastapi_safeguard import (
    FastAPISafeguard,
    HTTPSRedirectMiddlewareCheck,
    RateLimitingPresenceCheck,
)

safeguard = FastAPISafeguard(checks=[
    *FastAPISafeguard.recommended().checks,
    HTTPSRedirectMiddlewareCheck(),  # Often handled by infrastructure
    RateLimitingPresenceCheck(),      # Often external (API gateway)
])
app = FastAPI(lifespan=safeguard.lifespan())
```

### Custom Configuration
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
        extra_dependencies=[my_custom_auth_dep]  # accepts list or set
    ),
    ResponseModelSecurityCheck(),
    PaginationEnforcementCheck(),
])
app = FastAPI(lifespan=safeguard.lifespan())
```

### Integration with Custom Lifespan
If your app already has a custom lifespan context manager, use `run_checks()` instead:

```python
from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi_safeguard import FastAPISafeguard

safeguard = FastAPISafeguard.recommended()

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Run security checks at startup
    safeguard.run_checks(app)

    # Your custom startup logic
    print("Starting up with custom logic...")
    db_connection = await setup_database()

    yield  # App is running

    # Your custom shutdown logic
    await db_connection.close()
    print("Shutting down...")

app = FastAPI(lifespan=lifespan)
```

---
## Security Checks Reference

### ✅ Core Checks (Included in Recommended Preset)

High-value checks with low false positives, included by default when using `FastAPISafeguard.recommended()`:

| Check | Category | OWASP | Purpose | Example Finding |
|-------|----------|-------|---------|-----------------|
| **DependencySecurityCheck** | auth | API2, API5 | Ensures at least one accepted auth/security dependency | `GET /items has no accepted security dependency` |
| **UnsecuredAllowedMethodsCheck** | auth | API5 | Prevents unsafe methods on declared unsecured paths | `POST /status exposes unsafe method(s) without security` |
| **ResponseModelSecurityCheck** | schema | API3 | Enforces explicit response models for unsafe methods | `POST /users missing response_model...` |
| **BodyModelEnforcementCheck** | schema | API3, API6 | Prevents raw dict/list bodies (mass assignment) | `PATCH /users uses non-model raw body param(s): payload` |
| **SensitiveFieldExposureCheck** | data_exposure | API3 | Flags sensitive-looking response fields | `GET /auth/me response_model exposes potentially sensitive fields: password` |
| **SensitiveQueryParamCheck** | data_exposure | API3 | Flags sensitive-looking query parameter names | `GET /login exposes potentially sensitive data via query params: token` |
| **PaginationEnforcementCheck** | performance | API4 | Promotes pagination controls on list endpoints | `GET /reports returns a collection without pagination params` |
| **CORSMisconfigurationCheck** | config | API8 | Detects overly permissive CORS with credentials | `CORS misconfiguration: allow_origins='*', credentials allowed` |
| **DebugModeCheck** | config | API8 | Fails if running with `debug=True` | `Application running in debug mode` |

### 🔧 Optional Checks (Not Included by Default)

Available for manual addition. These have higher false positive rates or are often handled at infrastructure level:

| Check | Category | OWASP | Purpose | Why Not Default |
|-------|----------|-------|---------|-----------------|
| **HTTPSRedirectMiddlewareCheck** | config | API8 | Suggests enforcing HTTPS redirect | Usually handled by load balancer/reverse proxy |
| **TrustedHostMiddlewareCheck** | config | API8 | Suggests host header protection | Usually handled upstream |
| **RateLimitingPresenceCheck** | performance | API4 | Heuristic: missing rate limiting middleware | Often external (API gateway, nginx), weak heuristic |
| **ReturnTypeAnnotationCheck** | schema | API3 | Encourages type annotations when no response_model | Too noisy, not a security issue |
| **WildcardPathCheck** | routing | API3, API5 | Warns on broad catch-all path params | Legitimate use cases (file serving) |
| **DangerousMethodExposureCheck** | http_methods | - | Flags TRACE/CONNECT methods | Rarely exposed via FastAPI, non-issue in practice |
| **SSRFParameterCheck** | ssrf | - | Detects query params like 'url', 'uri', 'target' | Too many false positives |
| **AdminRouteOpenCheck** | auth | - | Flags routes with '/admin' in path without deps | Weak heuristic based on path naming |

**To use optional checks:**
```python
from fastapi_safeguard import FastAPISafeguard, HTTPSRedirectMiddlewareCheck

safeguard = FastAPISafeguard(checks=[
    *FastAPISafeguard.recommended().checks,
    HTTPSRedirectMiddlewareCheck(),  # Add as needed
])
```

---
## OWASP API Top 10 Coverage
Coverage by **core checks only** (included in recommended preset):

| OWASP Code | Risk (Short) | Core Checks | Optional Checks Available |
|------------|--------------|-------------|---------------------------|
| API2 | Broken Authentication | DependencySecurityCheck | - |
| API3 | Excessive Data / Object Property Level Exposure | ResponseModelSecurityCheck, BodyModelEnforcementCheck, SensitiveFieldExposureCheck, SensitiveQueryParamCheck | ReturnTypeAnnotationCheck, WildcardPathCheck |
| API4 | Unrestricted Resource Consumption | PaginationEnforcementCheck | RateLimitingPresenceCheck |
| API5 | Broken Function Level Authorization | DependencySecurityCheck, UnsecuredAllowedMethodsCheck | WildcardPathCheck |
| API6 | Mass Assignment | BodyModelEnforcementCheck | - |
| API8 | Security Misconfiguration | CORSMisconfigurationCheck, DebugModeCheck | HTTPSRedirectMiddlewareCheck, TrustedHostMiddlewareCheck |

> **Note:** Core checks provide solid coverage out-of-the-box. Optional checks can supplement coverage but have higher false positive rates.

---
## Decorators

| Decorator | Effect | Still Enforces Other Checks? | Use Case |
|-----------|--------|-------------------------------|----------|
| `@open_route` | Bypasses only the authentication dependency check | Yes | Public catalog, status page, landing page |
| `@disable_security_checks` | Skips *all* checks for that route | No | Internal metrics, intentionally exotic endpoint |

### Example
```python
from fastapi_safeguard import open_route, disable_security_checks

@open_route
@app.get("/public-catalog")
async def catalog():
    """Public endpoint - auth check skipped, other checks still enforced"""
    return {"items": []}

@disable_security_checks
@app.get("/internal-metrics")
async def metrics():
    """All security checks disabled for this route"""
    return {"cpu": 0.5}
```

**Note:** Decorator placement above `@app.get/...` is conventional but either order works; attributes attach to the underlying function.

---
## Baseline (Lock) File Workflow
Think of `security_baseline.json` the same way you think of `package-lock.json`, `poetry.lock`, or `Pipfile.lock` — but instead of pinning dependency versions, it pins the exact set of *currently accepted security findings*.

If something new (an unsecured route, missing response model, etc.) appears, the diff to this file makes that change obvious in a pull request. No silent drift.

### Mental Model
| You do this | The plugin does this |
|-------------|----------------------|
| Run app first time (no baseline) | Scans, prints findings, exits with error (unless update flag supplied) |
| Accept current state | Writes a lock file snapshot of findings |
| Add a risky change | Fails fast: new finding not in baseline |
| Fix a finding | Old one stays in file until refreshed so you can see improvement diff |
| Refresh with update flag | Prunes resolved entries / accepts new ones |

### Typical Flow
1. First scan (will fail if there are findings):
   ```bash
   uvicorn main:app --reload
   ```
2. Accept the current snapshot (write / update lock file):
   ```bash
   SECURITY_BASELINE_UPDATE=1 uvicorn main:app --reload
   # or in code: FastAPISafeguard(..., update_baseline=True)
   ```
3. Subsequent runs now fail **only** if *new* findings appear:
   ```bash
   uvicorn main:app --reload
   ```
4. After fixing issues, prune resolved ones (removes them from the file):
   ```bash
   SECURITY_BASELINE_UPDATE=1 uvicorn main:app --reload
   ```
5. If there are zero findings and you refresh, the file becomes empty (clean slate).

### Pull Request Review Shortcut
- Added lines in `security_baseline.json` = **new risks being accepted**.
- Removed lines = **improvements / fixes**.
- No file change = **no security posture change**.

### When to Reject a Baseline Update
If a baseline diff shows something like:
```
+  "GET /internal-metrics has no accepted security dependency"
```
Ask: *Was that endpoint meant to be open?* If not, fix instead of accepting.

### Quick Commands
```bash
# Fail on new findings only (normal dev / CI)
uvicorn main:app --reload

# Accept current findings (snapshot / refresh)
SECURITY_BASELINE_UPDATE=1 uvicorn main:app --reload
```

> Treat the baseline like a living contract: change it only when you are consciously accepting or removing risk.
---
## Startup Output Examples
### With new findings (will fail)
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
### After acceptance
```
Category Summary:
Category            Total   New  Accepted  OWASP
auth                   1     0         1   API2/API5
schema                 2     0         2   API3/API6

✅ All security findings match accepted baseline (3 accepted).
```
### Fully clean
```
✅ All security checks passed (0 findings, 12 routes, 14 checks).
```

---
## Environment Variables
| Variable | Purpose | Default |
|----------|---------|---------|
| `SECURITY_BASELINE_PATH` | Path to baseline file | `security_baseline.json` |
| `SECURITY_BASELINE_UPDATE` | If `1`, accept current findings (write / refresh baseline) | `0` |

(Planned) Additional env toggles (see Roadmap) could include JSON output, category include/exclude filters, severity levels.

---
## Configuration & Extensibility
```python
safeguard = FastAPISafeguard(
    checks=[
        DependencySecurityCheck(extra_dependencies={my_custom_dep}),
        ResponseModelSecurityCheck(),
        PaginationEnforcementCheck(pagination_param_names=["limit", "offset"]),
        # Add / remove checks freely
    ]
)
```
All checks expose constants:
- `CATEGORY`: used for grouped summary.
- `OWASP`: list of relevant OWASP API Top 10 codes.

Remove noisy checks (e.g. heuristics) simply by omitting them from the list.

---
## Writing a Custom Check
Minimal example:
```python
from fastapi.routing import APIRoute
from fastapi_safeguard import SecurityCheck

class EnforceJsonCheck(SecurityCheck):
    CATEGORY = "content"
    OWASP = ["API8"]  # misconfiguration flavor

    def check_route(self, route: APIRoute):
        if any("/legacy" in route.path for _ in [0]):  # sample logic
            return None
        for response in route.response_models or []:  # pseudocode
            pass
        # Return a finding string or None
        return None

safeguard = FastAPISafeguard(checks=[EnforceJsonCheck(), ...])
```
Rules:
- Return `None` when compliant.
- Return **short, stable strings** for findings (baseline matches on exact text).
- Add `CATEGORY` & `OWASP` for better summaries.

To reuse skip semantics: subclass `RouteCheck` instead of `SecurityCheck` to inherit `allowed_unsecured` & `@disable_security_checks` behavior.

---
## Ignoring / Accepting Findings
| Strategy | When to Use | Notes |
|----------|-------------|-------|
| Baseline accept | Existing known tech debt | Persists until you prune |
| `@open_route` | Legit public read-only endpoint | Still validated by other checks |
| `@disable_security_checks` | Exceptional / internal route | Use sparingly; documents intent |
| Remove check from config | Org chooses out-of-scope rule | Consider replacing with softer variant |

---
## CI/CD Integration
Example GitHub Actions step:
```yaml
- name: Security route check
  run: |
    pip install fastapi uvicorn
    python -c "import main"  # triggers lifespan startup logic
```
To accept current baseline in a controlled PR:
```yaml
- name: Update baseline (opt-in)
  if: github.event_name == 'workflow_dispatch'
  run: |
    SECURITY_BASELINE_UPDATE=1 python -c "import main"
    git add security_baseline.json
    git commit -m "chore: update security baseline" || echo "No changes"
```
Failing builds will show new findings distinctly (`+` prefixed rows).

---
## Performance Notes
- All checks execute **once at startup**; zero per-request overhead.
- Complexity roughly: O(routes × checks). Typical microservice: negligible (< few ms).
- Single-run checks short‑circuit after first execution.

---
## Roadmap
- Severity levels (warn vs error) & downgrade mechanism.
- JSON output mode for machine parsing (SBOM / SARIF style).
- Category include / exclude filters (env-based).
- `@warn_only` decorator (non-fatal for selected routes).
- Scope / role enforcement scaffold (e.g. `@require_scopes([...])`).
- Packaged distribution (PyPI) with plugin discovery entry points.
- Optional HTML or Markdown report emitter.

---
## Contributing
1. Fork & branch (`feat/my-improvement`).
2. Keep PRs focused (one conceptual change).
3. Add or update README examples if behavior / options change.
4. Prefer small, composable checks over giant multi-purpose ones.
5. Use clear finding strings (deterministic, no timestamps or randomness).

Feel free to open an issue proposing new categories or OWASP mappings first.

---
## Security Policy
Found a security issue *with the plugin itself* (e.g., code execution via baseline parsing)?
- Please open a GitHub issue or contact the maintainer
- For sensitive security issues, avoid including exploit details in public issues

---
## FAQ
**Q: Does this replace runtime authorization?**  
A: No. It prevents common *omissions* early. You still need robust runtime auth.

**Q: Will it block production if I forget to update the baseline?**  
A: Yes—intentionally. Incorporate baseline refresh into the PR that introduces the accepted risk.

**Q: Can I have different baselines per environment?**  
A: Yes—point `SECURITY_BASELINE_PATH` at different files (e.g. `security_baseline.dev.json`).

**Q: Are findings deduplicated?**  
A: Yes—each route/check pair should produce at most one stable string.

**Q: How do I silence RateLimitingPresenceCheck but keep others?**
A: Omit it from the `checks=[...]` list when constructing `FastAPISafeguard`.

**Q: What if I refactor a route path?**  
A: The old finding becomes *resolved*; run with `SECURITY_BASELINE_UPDATE=1` to prune it.

---
## License
MIT License - see [LICENSE](LICENSE) file for details.

---
**Happy hardening!**
