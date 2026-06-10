"""Scan orchestration: structured findings, baseline management and reporting.

The scanning pipeline is split into layers so each consumer takes only what
it needs:

* ``FastAPISafeguard.collect()`` — pure: scans the app and returns a
  :class:`ScanResult`. No printing, no baseline writes, no process exit.
* ``FastAPISafeguard.process()`` — collects, reports to stdout, manages the
  baseline file, and returns a process exit code (used by the CLI).
* ``FastAPISafeguard.run_checks()`` / ``lifespan()`` — startup-gate behavior:
  like ``process()`` but exits the interpreter on new findings.
"""
from __future__ import annotations

import json
import os
import sys
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterator, List, Optional, Sequence, Set, Tuple, Union

from fastapi import FastAPI
from fastapi.routing import APIRoute

from .checks import DependencySecurityCheck, SecurityCheck, recommended_checks


@dataclass(frozen=True)
class Finding:
    """One security finding: a check's verdict on one route or on the app.

    ``text`` is the stable identifier — baselines match on this exact string,
    so it must stay deterministic across runs.
    """

    text: str
    check: str
    category: str = "general"
    owasp: Tuple[str, ...] = ()

    @classmethod
    def from_check(cls, check: SecurityCheck, text: str) -> "Finding":
        """Build a Finding from the check instance that produced ``text``."""
        return cls(
            text=text,
            check=type(check).__name__,
            category=getattr(check, "CATEGORY", "general"),
            owasp=tuple(getattr(check, "OWASP", ()) or ()),
        )


@dataclass(frozen=True)
class ScanResult:
    """Outcome of scanning an application, with baseline context applied.

    Attributes:
        findings: Every finding produced by the scan, in scan order.
        accepted: Baseline entries that match a current finding.
        resolved: Baseline entries with no matching current finding (fixed).
        route_count: Number of API routes inspected.
        checks_count: Number of checks that ran.
    """

    findings: Tuple[Finding, ...]
    accepted: frozenset
    resolved: frozenset
    route_count: int
    checks_count: int

    def __len__(self) -> int:
        return len(self.findings)

    def __iter__(self) -> Iterator[Finding]:
        return iter(self.findings)

    @property
    def texts(self) -> Tuple[str, ...]:
        """Finding texts in scan order — what baselines store."""
        return tuple(finding.text for finding in self.findings)

    @property
    def new(self) -> Tuple[Finding, ...]:
        """Findings not covered by the baseline; these fail the scan."""
        return tuple(f for f in self.findings if f.text not in self.accepted)

    @property
    def accepted_findings(self) -> Tuple[Finding, ...]:
        """Findings covered by the baseline (known, accepted tech debt)."""
        return tuple(f for f in self.findings if f.text in self.accepted)

    @property
    def ok(self) -> bool:
        """True when there are no new findings (accepted ones are fine)."""
        return not self.new

    def by_category(self) -> Dict[str, Tuple[Finding, ...]]:
        """Group findings by check category, preserving scan order."""
        grouped: Dict[str, List[Finding]] = {}
        for finding in self.findings:
            grouped.setdefault(finding.category, []).append(finding)
        return {category: tuple(items) for category, items in grouped.items()}


class FastAPISafeguard:
    """Run registered security checks and manage a baseline (accepted findings) file.

    Baseline logic:
      * If baseline exists, findings listed there are accepted.
      * Startup fails only on NEW findings unless update_baseline / SECURITY_BASELINE_UPDATE=1.
      * With update flag, current findings overwrite the baseline.
      * Resolved (previously accepted but now gone) findings can be pruned with update.
    """

    def __init__(
        self,
        checks: Optional[List[SecurityCheck]] = None,
        baseline_path: Optional[str] = None,
        update_baseline: Optional[bool] = None,
    ) -> None:
        self.checks: List[SecurityCheck] = checks or [DependencySecurityCheck()]
        raw_path = (
            baseline_path
            or os.environ.get("SECURITY_BASELINE_PATH")
            or "security_baseline.json"
        )
        # Validate baseline path to prevent path traversal attacks
        self.baseline_path = self._validate_baseline_path(raw_path)
        if update_baseline is None:
            self.update_baseline = os.environ.get("SECURITY_BASELINE_UPDATE") == "1"
        else:
            self.update_baseline = update_baseline

    def _validate_baseline_path(self, path: str) -> str:
        """Validate and normalize baseline file path for security.

        Args:
            path: Raw path from user input or environment.

        Returns:
            Validated absolute path.

        Raises:
            ValueError: If path attempts traversal outside working directory.
        """
        abs_path = os.path.abspath(path)
        cwd = os.getcwd()
        # Ensure the path is within the current working directory or is absolute
        # This prevents malicious paths like "../../etc/passwd"
        # (cwd + os.sep avoids false prefix matches like /foo vs /foobar)
        inside_cwd = abs_path == cwd or abs_path.startswith(cwd + os.sep)
        if not inside_cwd:
            # Allow absolute paths outside cwd only if explicitly provided
            if not os.path.isabs(path):
                raise ValueError(
                    f"Baseline path '{path}' resolves outside working directory. "
                    f"Use absolute path if intentional."
                )
        return abs_path

    @classmethod
    def recommended(
        cls,
        *,
        allowed_unsecured: Optional[Sequence[str]] = None,
        extra_dependencies: Optional[Union[List[Any], Set[Any]]] = None,
        baseline_path: Optional[str] = None,
        update_baseline: Optional[bool] = None,
    ) -> "FastAPISafeguard":
        """Instantiate plugin with the recommended preset of checks."""
        checks = recommended_checks(
            allowed_unsecured=allowed_unsecured,
            extra_dependencies=extra_dependencies,
        )
        return cls(
            checks=checks,
            baseline_path=baseline_path,
            update_baseline=update_baseline,
        )

    # -------- Baseline helpers --------
    def _load_baseline(self) -> Set[str]:
        if not (self.baseline_path and os.path.exists(self.baseline_path)):
            return set()
        try:
            with open(self.baseline_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            accepted = data.get("accepted_findings")
            if isinstance(accepted, list):
                return set(accepted)
        except (OSError, json.JSONDecodeError, ValueError, KeyError) as exc:  # pragma: no cover - defensive
            print(f"⚠️  Could not parse baseline file '{self.baseline_path}': {exc}")
        return set()

    def _write_baseline(self, findings: Sequence[str]) -> None:
        if not self.baseline_path:
            return
        payload = {
            "schema_version": 1,
            "generated_at": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
            "accepted_findings": sorted(set(findings)),
            "checks_count": len(self.checks),
        }
        try:
            with open(self.baseline_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            # Set secure permissions: owner read/write only
            try:
                os.chmod(self.baseline_path, 0o600)
            except OSError:  # pragma: no cover - Windows / restricted environments
                pass  # Best effort
            print(f"💾 Updated security baseline written to {self.baseline_path}")
        except (OSError, TypeError, ValueError) as exc:  # pragma: no cover - defensive
            print(f"⚠️  Failed to write baseline file '{self.baseline_path}': {exc}")

    # -------- Scanning --------
    def collect(self, app: FastAPI) -> ScanResult:
        """Scan ``app`` and return a :class:`ScanResult`.

        Pure with respect to the filesystem and process: reads the baseline
        for comparison but never writes it, prints nothing, never exits.
        """
        findings: List[Finding] = []
        # App-level checks
        for check in self.checks:
            app_check_fn = getattr(check, "app_check", None)
            if callable(app_check_fn):
                text = app_check_fn(app)
                if text:
                    findings.append(Finding.from_check(check, text))
        route_count = 0
        for route in app.routes:
            if isinstance(route, APIRoute):
                route_count += 1
                for check in self.checks:
                    text = check.check_route(route)
                    if text:
                        findings.append(Finding.from_check(check, text))

        baseline = self._load_baseline()
        current = {finding.text for finding in findings}
        return ScanResult(
            findings=tuple(findings),
            accepted=frozenset(baseline & current),
            resolved=frozenset(baseline - current),
            route_count=route_count,
            checks_count=len(self.checks),
        )

    # -------- Reporting --------
    def _print_category_summary(self, result: ScanResult) -> None:
        if not result.findings:
            print("ℹ️  No findings to summarize by category.")
            return
        new_texts = {finding.text for finding in result.new}
        print("\nCategory Summary:")
        header = f"{'Category':<15} {'Total':>5} {'New':>5} {'Accepted':>9}  {'OWASP':<25}"
        print(header)
        print("-" * len(header))
        for category, findings in sorted(result.by_category().items()):
            total = len(findings)
            new_cnt = sum(1 for f in findings if f.text in new_texts)
            accepted_cnt = total - new_cnt
            owasp_codes = sorted({code for f in findings for code in f.owasp})
            owasp_str = "/".join(owasp_codes)[:25]
            print(f"{category:<15} {total:>5} {new_cnt:>5} {accepted_cnt:>9}  {owasp_str:<25}")
        print()

    def process(self, app: FastAPI) -> int:
        """Scan, report to stdout, and manage the baseline file.

        Returns:
            A process exit code: 0 when the scan passes (no new findings, or
            findings were accepted into the baseline), 1 on new findings.
        """
        result = self.collect(app)
        new = {finding.text for finding in result.new}
        baseline = result.accepted | result.resolved

        if result.findings:
            self._print_category_summary(result)
            if new:
                if self.update_baseline:
                    self._write_baseline(result.texts)
                    print("✅ Security checks passed with new findings accepted into baseline.")
                else:
                    print("❌ Security check failed: new findings detected (not in baseline):")
                    for text in sorted(new):
                        print(f"  + {text}")
                    if baseline and result.accepted:
                        print("ℹ️  Previously accepted findings (baseline):")
                        for text in sorted(result.accepted):
                            print(f"    = {text}")
                    print("\nTo accept current findings run with SECURITY_BASELINE_UPDATE=1 or set update_baseline=True.")
                    return 1
            else:
                if self.update_baseline and result.resolved:
                    self._write_baseline(result.texts)
                    print("✅ All security findings match baseline (baseline refreshed removing resolved items).")
                else:
                    print(f"✅ All security findings match accepted baseline ({len(result.findings)} accepted).")
                    if result.resolved:
                        print(f"ℹ️  {len(result.resolved)} previously accepted finding(s) resolved; run with SECURITY_BASELINE_UPDATE=1 to prune baseline.")
        else:
            if baseline:
                if self.update_baseline:
                    self._write_baseline([])
                    print("✅ No security findings. Baseline cleared (was non-empty).")
                else:
                    print("✅ No security findings. (Baseline exists – run with SECURITY_BASELINE_UPDATE=1 to clear.)")
            else:
                print(f"✅ All security checks passed (0 findings, {result.route_count} routes, {result.checks_count} checks).")
        return 0

    # -------- Lifespan --------
    def run_checks(self, app: FastAPI) -> None:
        """Run all security checks on the FastAPI app and exit on failures.

        This method can be called from within a custom lifespan context manager
        for apps that already have their own lifespan logic.

        Args:
            app: The FastAPI application instance to check.

        Raises:
            SystemExit: If new security findings are detected and not in baseline.
        """
        if self.process(app):
            sys.exit(1)

    def lifespan(self):
        """Return an async context manager for FastAPI lifespan integration.

        Usage:
            app = FastAPI(lifespan=safeguard.lifespan())

        Returns:
            An async context manager that runs security checks on startup.
        """
        @asynccontextmanager
        async def _lifespan(app: FastAPI):
            self.run_checks(app)
            yield

        return _lifespan

    def get_lifespan(self):
        """Deprecated alias for lifespan()."""
        return self.lifespan()
