"""RrfReleaseVisionBackend — ROUND-RRF R5: weighted_rrf lean search + vision release.

Pure multiple-inheritance composition of two already-shipped, already-reviewed backends:
``ReleaseVisionBackend`` (the ~692MB vision release + fail-closed guard) and
``LeanRRFBackend`` (the lean weighted_rrf steady-state path). Zero new logic.

MRO (C3 linearization):
    RrfReleaseVisionBackend → ReleaseVisionBackend → LeanRRFBackend → LeanSearchBackend
    → PrenormDotBackend → PrebuiltMatrixBackend → InMemoryBackend → ...
so method resolution gives:
  - ``release_vision()`` + ``_compute_field_scores`` (fail-closed guard) ← ReleaseVisionBackend;
  - ``_search_weighted_rrf`` (lean RRF) ← LeanRRFBackend;
  - ``_lean_bucket`` / ``_prebuild_task_matrices`` / GEMV ← shared ancestors.

The guard's ``_compute_field_scores`` calls ``super()._compute_field_scores`` which, down the
MRO, reaches PrenormDot's GEMV — so a post-release fallback still raises ReleaseUnsafeError
before reading the empty sentinel. ``__init__`` chains through every ancestor via ``super()``
(ReleaseVision sets ``_vision_released``, LeanRRF sets ``_lean_rrf_hits``, etc.).

Both parents are unmodified; this round adds only the composing class + injector.
"""

from __future__ import annotations

from exp.cache_latency_bench.opt.lean_rrf_backend import LeanRRFBackend
from exp.cache_latency_bench.opt.safe_release_backend import ReleaseVisionBackend


class RrfReleaseVisionBackend(ReleaseVisionBackend, LeanRRFBackend):
    """weighted_rrf lean search + ~692MB entry-vision release, composed via MRO."""
