"""What one gate-threshold Pareto run is bound to, in one reviewable table.

Every other module in this experiment takes a :class:`Binding` from here rather
than re-deriving a path, so swapping a library or mis-pointing a suite is a
one-line diff in a table instead of a string buried in a launch command.

Two conventions are load-bearing:

*   **The template is copied into the repo before it is used.** The winning
    search cell lives on ``/data``, which is a mutable scratch mount shared with
    the collection line; a run that generated its arms straight from there could
    not be reproduced once that file moved. ``template_path`` therefore points
    at the in-repo copy, and ``source_template`` records where it came from.
*   **Artifacts follow the four-slot layout** (``docs/experiments/artifact_layout.md``
    §1): generated YAML under ``config/``, run output under ``data/``, figures
    and analysis tools under ``analysis/``. The roots are computed here so no
    caller invents its own.

Coupling map:
  DEPENDS ON:  nothing (pure paths)
  CONSUMED BY: emit_gate_yamls, emit_warmup_pool, analysis/gate_pareto
  IF CHANGED:  the emitted arm YAMLs must be regenerated
"""

from __future__ import annotations

import dataclasses
import pathlib

REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]

#: Experiment id; also the sub-directory name inside each of the four slots.
EXPERIMENT = "gate_pareto"

#: Tasks per suite and evaluation episodes per task, for both LIBERO suites.
NUM_TASKS = 10
APOOL_TRIALS = 50

#: Episodes per task in the warmup phase (100 per suite, matching the pi0.5
#: line's warmup size so the solved quantiles rest on a comparable sample).
WARMUP_PER_TASK = 10


@dataclasses.dataclass(frozen=True)
class Binding:
    """Everything one suite's arms are generated and run against."""

    suite: str
    tag: str
    source_template: str
    library: str
    checkpoint: str
    apool_dir: str
    bpool_dir: str

    # -- four-slot roots ------------------------------------------------

    @property
    def config_root(self) -> pathlib.Path:
        """Generated YAML for this suite (in git; small and reproducible)."""
        return REPO_ROOT / "exp/libero_groot/config" / EXPERIMENT / self.suite

    @property
    def data_root(self) -> pathlib.Path:
        """Run output for this suite. ``exp/**/data/**`` is gitignored, and the
        repo's ``data`` is a symlink onto ``/data`` (the project's storage
        rule), so bytes never land on the system disk."""
        return REPO_ROOT / "exp/libero_groot/data" / EXPERIMENT / self.suite

    @property
    def analysis_root(self) -> pathlib.Path:
        """Figures, ``plot_data.json`` and the final report."""
        return REPO_ROOT / "exp/libero_groot/analysis" / EXPERIMENT

    @property
    def template_path(self) -> pathlib.Path:
        """In-repo copy of the winning search cell (see module docstring)."""
        return self.config_root / "template.yaml"


BINDINGS: tuple[Binding, ...] = (
    Binding(
        suite="libero_spatial",
        tag="sp",
        # ws_search round-2 marginal peak: vision_0 0.417 / vision_1 0.333 /
        # robot_state 0.250.
        source_template=(
            "/data/libero_cache/search/libero_spatial/r2/v0@5_v1@4_rs@3.yaml"
        ),
        library=(
            "/data/libero_cache/libraries/libero_spatial/"
            "libero_spatial_sp16_S3.pkl"
        ),
        checkpoint="/home/weiland/ckpt_n15_libero_spatial",
        apool_dir="exp/common/data/db_init/libero/libero_spatial_apool",
        bpool_dir="exp/common/data/db_init/libero/libero_spatial",
    ),
    Binding(
        suite="libero_10",
        tag="l10",
        # ws_search round-2 marginal peak: vision_0 0.500 / vision_1 0.417 /
        # robot_state 0.083.
        source_template="/data/libero_cache/search/libero_10/r2/v0@6_v1@5_rs@1.yaml",
        library="/data/libero_cache/libraries/libero_10/libero_10_sp16_S3.pkl",
        checkpoint="/home/weiland/ckpt_n15_libero_10",
        apool_dir="exp/common/data/db_init/libero/libero_10_apool",
        bpool_dir="exp/common/data/db_init/libero/libero_10",
    ),
)


def for_suite(suite: str) -> Binding:
    """Return the binding for ``suite``, or raise naming the known suites."""
    for binding in BINDINGS:
        if binding.suite == suite:
            return binding
    raise KeyError(f"unknown suite {suite!r}; known: {[b.suite for b in BINDINGS]}")


def for_tag(tag: str) -> Binding:
    """Return the binding for a short arm tag (``sp`` / ``l10``)."""
    for binding in BINDINGS:
        if binding.tag == tag:
            return binding
    raise KeyError(f"unknown tag {tag!r}; known: {[b.tag for b in BINDINGS]}")
