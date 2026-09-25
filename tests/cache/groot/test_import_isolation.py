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
    "src/openpi/cache/groot/cp2_key_builder.py",
    # Trace serving mode (plan cache_trace_mode §9-9): the model-agnostic
    # runtime, the GR00T adapter and the jax-free batching core / server
    # lifecycle are imported by the island's servers.
    "src/openpi/cache/groot/batcher.py",
    "src/openpi/cache/trace/__init__.py",
    "src/openpi/cache/trace/types.py",
    "src/openpi/cache/trace/records.py",
    "src/openpi/cache/trace/runtime.py",
    "src/openpi/cache/trace/h5_sink.py",
    "src/openpi/cache/trace/groot.py",
    "src/openpi/serving/batching_core.py",
    "src/openpi/serving/trace_serving.py",
    # Warm reset continuation family (plan warm_continuation_first_class §5):
    # the model-agnostic package root and the GR00T executor.
    "src/openpi/cache/warm_reset/__init__.py",
    "src/openpi/cache/warm_reset/types.py",
    "src/openpi/cache/warm_reset/runtime.py",
    "src/openpi/cache/warm_reset/evidence.py",
    "src/openpi/cache/warm_reset/groot.py",
    # ActionCache-baseline island scripts (plan actioncache_baseline_groot §3.6/§3.9/§3.11).
    "exp/libero_groot/cp2_reconstruct.py",
    "exp/libero_groot/build_cp2_artifact_groot.py",
    "exp/libero_groot/build_shadow_table_groot.py",
    "exp/libero_groot/groot_cp2_parity.py",
    "exp/libero_groot/verify_shadow_h5.py",
    "exp/libero_groot/emit_task_map.py",
    "exp/libero_groot/bench_cp2_overhead_groot.py",
]

# Island entry points whose *transitive* module-level imports are checked too:
# they reuse ``exp.actioncache_baseline`` helpers, and a jax import added to one
# of those later would only fail on the machine.
TRANSITIVE_ROOTS = [
    "exp/libero_groot/build_cp2_artifact_groot.py",
    "exp/libero_groot/build_shadow_table_groot.py",
    "exp/libero_groot/groot_cp2_parity.py",
    "exp/libero_groot/bench_cp2_overhead_groot.py",
    "src/openpi/cache/groot/interceptor.py",
    "src/openpi/cache/groot/batcher.py",
    "src/openpi/cache/trace/groot.py",
    "src/openpi/cache/trace/runtime.py",
    "src/openpi/serving/trace_serving.py",
    "src/openpi/cache/warm_reset/__init__.py",
    "src/openpi/cache/warm_reset/groot.py",
]

#: The Pi0.5 trace adapter is loaded lazily by the Pi0.5 interceptor only; no
#: GR00T import chain may pass through it (plan §9-9). The Pi0.5 warm reset
#: executor imports the model and is assembled by ``serve_policy`` only.
PI05_ONLY_MODULES = ("openpi.cache.trace.pi05", "openpi.cache.warm_reset.pi05")


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


def _module_level_imports(path: pathlib.Path) -> set[str]:
    tree = ast.parse(path.read_text())
    names: set[str] = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            names.add(node.module)
    return names


def _module_path(name: str) -> pathlib.Path | None:
    for base in ("src", "."):
        for candidate in (
            REPO_ROOT / base / (name.replace(".", "/") + ".py"),
            REPO_ROOT / base / name.replace(".", "/") / "__init__.py",
        ):
            if candidate.exists():
                return candidate
    return None


def _transitive_offenders(start: pathlib.Path) -> list[tuple[str, str]]:
    """(offending module, importer) pairs reachable through module-level imports."""
    seen: set[str] = set()
    offenders: list[tuple[str, str]] = []
    stack: list[tuple[pathlib.Path, str]] = [(start, str(start.relative_to(REPO_ROOT)))]
    while stack:
        path, label = stack.pop()
        if label in seen:
            continue
        seen.add(label)
        for name in sorted(_module_level_imports(path)):
            if any(name == p or name.startswith(p + ".") for p in FORBIDDEN_PREFIXES):
                offenders.append((name, label))
            elif name.startswith(("openpi.", "exp.")):
                child = _module_path(name)
                if child is not None:
                    stack.append((child, name))
    return offenders


@pytest.mark.parametrize("relative", TRANSITIVE_ROOTS)
def test_island_scripts_stay_jax_free_transitively(relative: str) -> None:
    offenders = _transitive_offenders(REPO_ROOT / relative)
    assert not offenders, f"{relative} reaches {offenders} through module-level imports"


def _transitive_modules(start: pathlib.Path) -> set[str]:
    seen: set[str] = set()
    stack: list[tuple[pathlib.Path, str]] = [(start, str(start.relative_to(REPO_ROOT)))]
    reached: set[str] = set()
    while stack:
        path, label = stack.pop()
        if label in seen:
            continue
        seen.add(label)
        for name in _imported_modules(path):
            reached.add(name)
            if name.startswith(("openpi.", "exp.")):
                child = _module_path(name)
                if child is not None:
                    stack.append((child, name))
    return reached


@pytest.mark.parametrize(
    "relative",
    [
        "src/openpi/cache/groot/interceptor.py",
        "src/openpi/cache/groot/batcher.py",
        "src/openpi/cache/trace/groot.py",
        "src/openpi/cache/trace/runtime.py",
        "exp/libero_groot/serve_groot_libero.py",
        "exp/robocasa365/serve_groot_n15.py",
    ],
)
def test_groot_chain_never_reaches_the_pi05_adapter(relative: str) -> None:
    """Even through function-level imports: the GR00T path has no reason to load it."""
    reached = _transitive_modules(REPO_ROOT / relative)
    offenders = sorted(m for m in reached if m in PI05_ONLY_MODULES)
    assert not offenders, f"{relative} reaches {offenders}"


def test_the_transitive_guard_would_actually_catch_something() -> None:
    offenders = _transitive_offenders(REPO_ROOT / "exp/actioncache_baseline/build_cp2_artifact.py")
    assert offenders, "the Pi0.5 builder no longer reaches a forbidden module; the transitive check is inert"


def test_the_guard_would_actually_catch_something() -> None:
    """Reverse control: the Pi0.5 interceptor must trip the same check."""
    offenders = [
        name
        for name in _imported_modules(REPO_ROOT / "src/openpi/cache/interceptor.py")
        for prefix in FORBIDDEN_PREFIXES
        if name == prefix or name.startswith(prefix + ".")
    ]
    assert offenders, "the forbidden-prefix list no longer matches anything"
