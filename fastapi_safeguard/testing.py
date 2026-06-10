"""Test-suite integration.

Prefer :func:`assert_safeguard` over wiring the lifespan in tests: it raises a
plain ``AssertionError`` (no ``SystemExit``, no stdout side effects), so it
behaves like any other failing assertion under pytest or unittest.
"""
from __future__ import annotations

from typing import Optional, Sequence

from fastapi import FastAPI

from .checks import SecurityCheck
from .scanner import FastAPISafeguard, ScanResult


def assert_safeguard(
    app: FastAPI,
    safeguard: Optional[FastAPISafeguard] = None,
    *,
    checks: Optional[Sequence[SecurityCheck]] = None,
    baseline_path: Optional[str] = None,
) -> ScanResult:
    """Scan ``app`` and raise ``AssertionError`` on new security findings.

    Findings accepted by the baseline pass, exactly as at startup; the
    baseline file is never written.

    Args:
        app: The FastAPI application to scan.
        safeguard: A preconfigured ``FastAPISafeguard``. Mutually exclusive
            with ``checks``/``baseline_path``; defaults to the recommended preset.
        checks: Explicit check list, if no ``safeguard`` is given.
        baseline_path: Baseline file location, if no ``safeguard`` is given.

    Returns:
        The :class:`ScanResult`, so tests can assert further on it.

    Raises:
        AssertionError: One line per new finding.

    Example:
        def test_security_posture():
            assert_safeguard(app)
    """
    if safeguard is not None and (checks is not None or baseline_path is not None):
        raise TypeError("Pass either 'safeguard' or 'checks'/'baseline_path', not both.")
    if safeguard is None:
        if checks is not None:
            safeguard = FastAPISafeguard(checks=list(checks), baseline_path=baseline_path)
        else:
            safeguard = FastAPISafeguard.recommended(baseline_path=baseline_path)
    result = safeguard.collect(app)
    if not result.ok:
        lines = "\n".join(f"  + {finding.text}" for finding in result.new)
        raise AssertionError(
            f"{len(result.new)} new security finding(s) not in baseline:\n{lines}"
        )
    return result
