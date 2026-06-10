"""Command-line interface: scan an application without touching its lifespan.

Usage:
    fastapi-safeguard check app.main:app
    fastapi-safeguard check app.main:create_app --factory
    python -m fastapi_safeguard check main:app --baseline security_baseline.json

Exit codes:
    0 — scan passed (no new findings, or findings accepted into the baseline)
    1 — new security findings detected
    2 — the application could not be loaded, or usage error
"""
from __future__ import annotations

import argparse
import importlib
import os
import sys
from typing import Optional, Sequence

from fastapi import FastAPI

from .scanner import FastAPISafeguard


class CLIError(Exception):
    """Raised when the target application cannot be loaded."""


def load_app(spec: str, *, factory: bool = False) -> FastAPI:
    """Resolve an import string like ``package.module:attribute`` to a FastAPI app.

    The attribute defaults to ``app`` and may be dotted (``module:obj.app``).
    With ``factory=True`` the resolved attribute is called (no arguments) and
    the returned application is used.

    Raises:
        CLIError: With a user-facing message when resolution fails.
    """
    module_name, _, attr_path = spec.partition(":")
    attr_path = attr_path or "app"
    if not module_name:
        raise CLIError(f"Invalid application spec {spec!r}; expected 'module[:attribute]'.")
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise CLIError(f"Could not import module {module_name!r}: {exc}") from exc
    target: object = module
    for attr in attr_path.split("."):
        try:
            target = getattr(target, attr)
        except AttributeError as exc:
            raise CLIError(f"Module {module_name!r} has no attribute {attr_path!r}.") from exc
    if factory:
        if not callable(target):
            raise CLIError(f"{spec!r} is not callable; --factory expects an application factory.")
        target = target()
    if not isinstance(target, FastAPI):
        hint = " Did you mean to pass --factory?" if callable(target) else ""
        raise CLIError(
            f"{spec!r} resolved to {type(target).__name__}, not a FastAPI application.{hint}"
        )
    return target


def _build_parser() -> argparse.ArgumentParser:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="fastapi-safeguard",
        description="Startup-time security checks for FastAPI applications.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    check = subparsers.add_parser(
        "check",
        help="Scan an application and exit non-zero on new security findings.",
        description=(
            "Import the application, run the recommended security checks against it, "
            "and report findings against the baseline. The app's lifespan is NOT run."
        ),
    )
    check.add_argument(
        "app",
        help="Application import string, e.g. 'app.main:app' (attribute defaults to 'app').",
    )
    check.add_argument(
        "--factory",
        action="store_true",
        help="Treat the imported attribute as an application factory and call it.",
    )
    check.add_argument(
        "--app-dir",
        default=".",
        help="Directory to add to the import path before loading the app (default: current directory).",
    )
    check.add_argument(
        "--baseline",
        default=None,
        help="Path to the baseline file (default: SECURITY_BASELINE_PATH or security_baseline.json).",
    )
    check.add_argument(
        "--update-baseline",
        action="store_true",
        help="Accept current findings and write/refresh the baseline file.",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    """CLI entry point. Returns a process exit code (see module docstring)."""
    args = _build_parser().parse_args(argv)

    sys.path.insert(0, os.path.abspath(args.app_dir))
    try:
        app = load_app(args.app, factory=args.factory)
    except CLIError as exc:
        print(f"❌ {exc}", file=sys.stderr)
        return 2

    safeguard = FastAPISafeguard.recommended(
        baseline_path=args.baseline,
        # None keeps the SECURITY_BASELINE_UPDATE env var fallback active.
        update_baseline=True if args.update_baseline else None,
    )
    return safeguard.process(app)


if __name__ == "__main__":  # pragma: no cover - exercised via __main__.py
    raise SystemExit(main())
