"""Tests for the fastapi-safeguard CLI."""
import json
import sys

import pytest

from fastapi_safeguard import __version__
from fastapi_safeguard.cli import CLIError, load_app, main

SECURE_APP = """\
from fastapi import Depends, FastAPI
from fastapi.security import HTTPBearer

app = FastAPI()
bearer = HTTPBearer()

@app.get("/secure")
async def secure(credentials=Depends(bearer)):
    return {"ok": True}
"""

INSECURE_APP = """\
from fastapi import FastAPI

app = FastAPI()

@app.get("/unsecured")
async def unsecured():
    return {"ok": True}
"""

FACTORY_APP = """\
from fastapi import FastAPI

def create_app():
    return FastAPI()

not_an_app = object()
"""


@pytest.fixture(autouse=True)
def isolated_imports(tmp_path, monkeypatch):
    """Keep CLI-induced sys.path/sys.modules changes from leaking between tests."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    for name in ["secure_app", "insecure_app", "factory_app"]:
        sys.modules.pop(name, None)
    yield
    for name in ["secure_app", "insecure_app", "factory_app"]:
        sys.modules.pop(name, None)


@pytest.fixture
def app_dir(tmp_path):
    (tmp_path / "secure_app.py").write_text(SECURE_APP)
    (tmp_path / "insecure_app.py").write_text(INSECURE_APP)
    (tmp_path / "factory_app.py").write_text(FACTORY_APP)
    return tmp_path


def run_check(app_dir, spec, *extra, baseline=None):
    argv = ["check", spec, "--app-dir", str(app_dir)]
    if baseline is not None:
        argv += ["--baseline", str(baseline)]
    argv += list(extra)
    return main(argv)


# ----------------- check command -----------------

def test_check_passes_on_secure_app(app_dir, tmp_path, capsys):
    assert run_check(app_dir, "secure_app:app", baseline=tmp_path / "b.json") == 0
    assert "✅ All security checks passed" in capsys.readouterr().out


def test_check_fails_on_insecure_app(app_dir, tmp_path, capsys):
    assert run_check(app_dir, "insecure_app:app", baseline=tmp_path / "b.json") == 1
    out = capsys.readouterr().out
    assert "❌ Security check failed" in out
    assert "GET /unsecured has no accepted security dependency" in out


def test_check_attribute_defaults_to_app(app_dir, tmp_path):
    assert run_check(app_dir, "secure_app", baseline=tmp_path / "b.json") == 0


def test_check_update_baseline_then_pass(app_dir, tmp_path, capsys):
    baseline = tmp_path / "b.json"
    assert run_check(app_dir, "insecure_app:app", "--update-baseline", baseline=baseline) == 0
    assert "accepted into baseline" in capsys.readouterr().out

    with open(baseline) as f:
        accepted = json.load(f)["accepted_findings"]
    assert accepted == ["GET /unsecured has no accepted security dependency"]

    assert run_check(app_dir, "insecure_app:app", baseline=baseline) == 0
    assert "match accepted baseline" in capsys.readouterr().out


def test_check_factory(app_dir, tmp_path):
    assert run_check(app_dir, "factory_app:create_app", "--factory", baseline=tmp_path / "b.json") == 0


# ----------------- load errors (exit code 2) -----------------

def test_missing_module(app_dir, capsys):
    assert run_check(app_dir, "no_such_module:app") == 2
    assert "Could not import module 'no_such_module'" in capsys.readouterr().err


def test_missing_attribute(app_dir, capsys):
    assert run_check(app_dir, "secure_app:nope") == 2
    assert "has no attribute 'nope'" in capsys.readouterr().err


def test_not_a_fastapi_app(app_dir, capsys):
    assert run_check(app_dir, "factory_app:not_an_app") == 2
    assert "not a FastAPI application" in capsys.readouterr().err


def test_callable_without_factory_flag_hints(app_dir, capsys):
    assert run_check(app_dir, "factory_app:create_app") == 2
    assert "Did you mean to pass --factory?" in capsys.readouterr().err


def test_factory_flag_on_non_callable(app_dir, capsys):
    assert run_check(app_dir, "factory_app:not_an_app", "--factory") == 2
    assert "--factory expects an application factory" in capsys.readouterr().err


def test_empty_module_name():
    with pytest.raises(CLIError, match="expected 'module"):
        load_app(":app")


# ----------------- parser -----------------

def test_version_flag(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main(["--version"])
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_no_command_is_usage_error(capsys):
    with pytest.raises(SystemExit) as excinfo:
        main([])
    assert excinfo.value.code == 2


def test_python_dash_m_entrypoint(monkeypatch, capsys):
    import runpy
    monkeypatch.setattr(sys, "argv", ["fastapi_safeguard", "--version"])
    with pytest.raises(SystemExit) as excinfo:
        runpy.run_module("fastapi_safeguard", run_name="__main__")
    assert excinfo.value.code == 0
    assert __version__ in capsys.readouterr().out
