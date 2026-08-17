"""The GR00T cache path must stay importable where jax is not installed.

The server runs in a virtualenv that has torch but no jax. A stray import of
`openpi.cache.interceptor`, `openpi.models`, `openpi.policies` or
`openpi.collect.collection_policy` would not fail here — it would fail on the
machine, at start-up, after the checkpoint has been loaded.

The check is static rather than an import attempt, because this test process
*does* have jax: importing the modules successfully would prove nothing.
"""

from __future__ import annotations

import ast
import pathlib

import pytest

REPO_ROOT = pathlib.Path(__file__).resolve().parents[3]

FORBIDDEN_PREFIXES = (
    "jax",
    "jaxlib",
    "flax",
    "openpi.models",
    "openpi.policies",
    "openpi.cache.interceptor",
    "openpi.collect.collection_policy",
)

GUARDED_FILES = [
    "src/openpi/cache/groot/__init__.py",
    "src/openpi/cache/groot/staged.py",
    "src/openpi/cache/groot/key_builder.py",
    "src/openpi/cache/groot/interceptor.py",
    "src/openpi/cache/groot/load_guard.py",
    "exp/robocasa365/groot_cache_collector.py",
]


def _imported_modules(path: pathlib.Path) -> set[str]:
    """Every module named by an import in the file, including inside functions."""
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


@pytest.mark.parametrize("relative", GUARDED_FILES)
def test_no_jax_bound_imports(relative: str) -> None:
    path = REPO_ROOT / relative
    assert path.exists(), relative
    offenders = sorted(
        name
        for name in _imported_modules(path)
        for prefix in FORBIDDEN_PREFIXES
        if name == prefix or name.startswith(prefix + ".")
    )
    assert not offenders, (
        f"{relative} imports {offenders}, which pull in jax or a Pi0.5-only "
        "module. The GR00T island has neither."
    )


def test_the_guard_would_actually_catch_something() -> None:
    """Reverse control: the Pi0.5 interceptor must trip the same check."""
    offenders = [
        name
        for name in _imported_modules(REPO_ROOT / "src/openpi/cache/interceptor.py")
        for prefix in FORBIDDEN_PREFIXES
        if name == prefix or name.startswith(prefix + ".")
    ]
    assert offenders, "the forbidden-prefix list no longer matches anything"
