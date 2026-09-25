"""Static decoupling: components that must stay unaware of warm reset never import it (plan §4.9, §9.3)."""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]
PACKAGE = "openpi.cache.warm_reset"

UNAWARE = [
    "src/openpi/cache/components",
    "src/openpi/cache/orchestrator.py",
    "src/openpi/cache/cache_storage.py",
    "src/openpi/cache/backends",
    "src/openpi/conductor",
    "src/openpi/serving/websocket_policy_server.py",
    "examples/libero",
    "exp/robocasa365/episode_runner.py",
    # The interceptors receive the executor by injection and never import the package.
    "src/openpi/cache/interceptor.py",
    "src/openpi/cache/groot/interceptor.py",
    "src/openpi/cache/config.py",
]


def _files(relative: str) -> list[pathlib.Path]:
    path = REPO_ROOT / relative
    return sorted(path.rglob("*.py")) if path.is_dir() else [path]


def _imports(path: pathlib.Path) -> set[str]:
    names: set[str] = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("relative", UNAWARE)
def test_unaware_components_do_not_import_warm_reset(relative):
    files = _files(relative)
    assert files, relative
    offenders = {
        str(f.relative_to(REPO_ROOT)): sorted(n for n in _imports(f) if n == PACKAGE or n.startswith(PACKAGE + "."))
        for f in files
    }
    assert not {k: v for k, v in offenders.items() if v}


def test_the_scan_would_catch_an_import():
    """Reverse control: the serving entry point that assembles the feature is seen importing it."""
    assert any(n.startswith(PACKAGE) for n in _imports(REPO_ROOT / "scripts/serve_policy.py"))


def test_package_root_never_loads_an_executor():
    init = _imports(REPO_ROOT / "src/openpi/cache/warm_reset/__init__.py")
    assert not {f"{PACKAGE}.pi05", f"{PACKAGE}.groot"} & init
    groot = _imports(REPO_ROOT / "src/openpi/cache/warm_reset/groot.py")
    assert f"{PACKAGE}.pi05" not in groot
