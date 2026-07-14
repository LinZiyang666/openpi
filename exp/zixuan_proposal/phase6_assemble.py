"""Enforceable projected-artifact build + serve chain (TRACER Phase 6, §6.3).

The Phase-6 consumers (`ical_filter`, `record_weights_digest`, `assert_serve_binding`, the
D- manifest) must not live as four disconnected helpers each touched only by a unit test.
This module ties them into TWO enforceable entry points that mirror the approved pipeline:

  * ``assemble_projected_artifact`` -- the BUILD chain (§6.3a-d): validate the machine-readable
    D- manifest against the authoritative provenance table, filter the merged pool to the I_cal
    (even-init) library with a zero-odd guard, stamp the IMMUTABLE weights digest, and attach
    both manifests. STOPS before producing an artifact if any manifest entry is missing /
    incomplete / identity-mismatched.
  * ``serve_init`` -- the SERVE hook (§6.3d): before any query is keyed, enforce the recorded
    digest <-> live weight bytes <-> serve-YAML binding AND re-assert I_cal purity + manifest
    presence, so a projected store can never be served against raw/other-head queries.

Parameterized on a provenance table + a per-H5 stat function so the chain is testable without
the 134 GB held-out D- dump (future numeric weights need not be fabricated to exercise the
code that consumes them).

``main()`` exposes these as the REAL executable pipeline the §6.3 runbook invokes: ``build``
(assemble + write the projected artifact) and ``serve-preflight`` (the MANDATORY digest + I_cal
gate run before ``serve_policy`` launches a projected lane). The Retrieval@K top-1-prefix
diagnostic (``strategy_topk_prefix_ok``) runs the real ``DualRetrievalKnnStrategy`` at K=1/K=5.
"""

from __future__ import annotations

import pathlib

from exp.zixuan_proposal.phase6_emit import ical_filter
from exp.zixuan_proposal.phase6_provenance import assert_serve_binding, record_weights_digest

# D- global-id -> identity: the Phase-4 held-out failure dump packs (task, init) as
# gid = task_id * INITS_PER_TASK + init_state_idx (provenance table `phase4_dminus_provenance.md`).
INITS_PER_TASK = 50

# FROZEN per-suite held-out D- totals (plan §6.3b): the contamination-fixed `failure_heldout`
# dump has exactly this many failure episodes and total steps. `validate_dminus_manifest`
# enforces BOTH so a manifest with the right shape but the wrong dataset (e.g. a truncated or
# padded build) is rejected -- equal counts alone cannot distinguish datasets.
DMINUS_TOTALS = {
    "libero_spatial": {"episodes": 18, "steps": 792},
    "libero_10": {"episodes": 85, "steps": 8840},
}


def resolve_dminus_ident(gid: int) -> tuple[int, int]:
    """(task_id, init_state_idx) for a D- global id (task = gid // 50, init = gid % 50)."""
    return (gid // INITS_PER_TASK, gid % INITS_PER_TASK)


def h5_source_key(name) -> str:
    """Canonical D- source key: the H5 basename with any trailing ``.h5`` removed.

    The projected artifact rows record the source as a ``trajectory_id`` STEM (no extension),
    while the D- manifest records the H5 ``h5_basename`` (with ``.h5``). Normalising BOTH sides to
    the stem here binds a correct 792-row build to its manifest regardless of the extension, while
    still rejecting a genuinely different source (different stem).
    """
    s = str(name)
    return s[:-3] if s.endswith(".h5") else s


# ------------------------------------------------------------------
# D- machine-readable manifest (§6.3b)
# ------------------------------------------------------------------
def build_dminus_manifest(provenance_rows: list, stat_fn) -> list:
    """One manifest record per expected D- H5, resolving identity + recording completeness.

    ``provenance_rows`` = [{"h5_basename": str, "gid": int}, ...] (the authoritative expected
    set). ``stat_fn(h5_basename) -> {"exists", "success", "n_steps", "complete"}`` reports the
    on-disk truth for one H5. Records the resolved ``(task_id, init_state_idx)``, the
    ``success`` flag, step count, and completeness -- counts alone cannot distinguish datasets
    (§6.3b), so identity + per-episode completeness are carried explicitly.
    """
    manifest = []
    for row in provenance_rows:
        base = row["h5_basename"]
        task_id, init = resolve_dminus_ident(row["gid"])
        st = stat_fn(base)
        manifest.append(
            {
                "h5_basename": base,
                "task_id": task_id,
                "init_state_idx": init,
                "success": st.get("success"),
                "n_steps": st.get("n_steps"),
                "exists": bool(st.get("exists")),
                "complete": bool(st.get("complete")),
            }
        )
    return manifest


def validate_dminus_manifest(manifest: list, provenance_table: set, *, suite: str) -> list:
    """Fail loud unless the manifest matches the authoritative provenance table AND the FROZEN
    per-suite totals EXACTLY (§6.3b).

    ``provenance_table`` = set of ``(h5_basename, task_id, init_state_idx)``. Raises unless:
      * ``suite`` is a known suite with frozen totals in ``DMINUS_TOTALS``;
      * ``len(manifest) == DMINUS_TOTALS[suite]['episodes']`` (spatial 18, l10 85) AND the
        summed step count equals ``DMINUS_TOTALS[suite]['steps']`` (spatial 792, l10 8840);
      * the manifest's ``(basename, task, init)`` set equals ``provenance_table`` exactly
        (no missing episode, no extra/mis-resolved episode);
      * every entry is a genuine failure (``success is False``);
      * every listed H5 ``exists`` AND is ``complete``.
    Returns the manifest unchanged when valid, so the caller can STOP the build on any raise.
    """
    if suite not in DMINUS_TOTALS:
        raise ValueError(f"unknown suite {suite!r}; frozen D- totals defined for {sorted(DMINUS_TOTALS)}")
    frozen = DMINUS_TOTALS[suite]
    if len(manifest) != frozen["episodes"]:
        raise ValueError(f"D- manifest has {len(manifest)} episodes, frozen {suite} total is {frozen['episodes']}")
    total_steps = sum(int(m.get("n_steps") or 0) for m in manifest)
    if total_steps != frozen["steps"]:
        raise ValueError(f"D- manifest step total {total_steps} != frozen {suite} total {frozen['steps']}")
    got = {(m["h5_basename"], m["task_id"], m["init_state_idx"]) for m in manifest}
    if got != provenance_table:
        missing = sorted(provenance_table - got)
        extra = sorted(got - provenance_table)
        raise ValueError(f"D- manifest != provenance table (missing={missing}, extra={extra})")
    bad_outcome = [m["h5_basename"] for m in manifest if m["success"] is not False]
    if bad_outcome:
        raise ValueError(f"D- manifest has non-failure episodes: {bad_outcome}")
    incomplete = [m["h5_basename"] for m in manifest if not (m["exists"] and m["complete"])]
    if incomplete:
        raise ValueError(f"D- manifest has missing/incomplete H5: {incomplete}")
    return manifest


# ------------------------------------------------------------------
# Build chain (§6.3a-d)
# ------------------------------------------------------------------
def assemble_projected_artifact(
    artifact: dict,
    ident_fn,
    weights_path,
    *,
    dminus_manifest: list,
    provenance_table: set,
    suite: str,
    outcome_fn,
    source_fn,
) -> dict:
    """Assemble a serve-ready projected artifact by running the full §6.3 build chain.

    Order: (b) validate the D- manifest against provenance + frozen suite totals -> STOP on any
    mismatch; (b') bind the manifest to the D- rows ACTUALLY present in ``artifact['entries']``
    by their SOURCE H5 basename and step indices, not merely resolved identity + multiplicity;
    (c) filter the merged pool to I_cal with the zero-odd guard; (d) stamp the immutable weights
    digest + attach both manifests. ``ident_fn(entry) -> (task, init) | None``;
    ``outcome_fn(entry) -> int`` (-1 for D-); ``source_fn(entry) -> (h5_basename, step_idx)``.
    Fails loud (never returns a half-built artifact).

    The per-H5 binding rejects the exact §6.3b contamination the identity/count check missed: an
    artifact from a DIFFERENT dump with the same identities and full row counts, but rows whose
    ``trajectory_id`` disagrees with the manifest ``h5_basename``, or with duplicate/missing
    ``step_idx``. For every manifest episode the artifact must contribute rows whose source H5 is
    that episode's basename, whose resolved identity matches, and whose step indices are exactly
    the contiguous unique set ``0..n_steps-1``. Source keys are compared on the ``.h5``-stripped
    stem (``h5_source_key``) so a row's ``trajectory_id`` stem binds to the manifest's ``.h5``
    basename.
    """
    validate_dminus_manifest(dminus_manifest, provenance_table, suite=suite)
    manifest_by_h5 = {h5_source_key(m["h5_basename"]): m for m in dminus_manifest}
    rows_by_h5: dict = {}
    for e in artifact["entries"]:
        if outcome_fn(e) == -1:
            h5, step_idx = source_fn(e)
            rows_by_h5.setdefault(h5_source_key(h5), []).append((ident_fn(e), step_idx))
    if set(rows_by_h5) != set(manifest_by_h5):
        missing = sorted(set(manifest_by_h5) - set(rows_by_h5))
        extra = sorted(set(rows_by_h5) - set(manifest_by_h5))
        raise ValueError(
            f"artifact D- source H5 basenames != manifest (wrong dump?): "
            f"missing_from_artifact={missing[:3]}, extra_in_artifact={extra[:3]}; refusing to stamp"
        )
    for h5, m in manifest_by_h5.items():
        want_ident = (m["task_id"], m["init_state_idx"])
        rows = rows_by_h5[h5]
        bad_ident = {r_ident for r_ident, _ in rows if r_ident != want_ident}
        if bad_ident:
            raise ValueError(f"D- H5 {h5} rows resolve to identities {sorted(bad_ident)[:3]} != manifest {want_ident}")
        steps = sorted(s for _, s in rows)
        if steps != list(range(int(m["n_steps"]))):
            raise ValueError(
                f"D- H5 {h5} step indices are not the contiguous unique set 0..{int(m['n_steps']) - 1} "
                f"(got {len(steps)} rows, sample {steps[:5]}); refusing to stamp a truncated/duplicated build"
            )
    total_rows = sum(len(v) for v in rows_by_h5.values())
    if total_rows != DMINUS_TOTALS[suite]["steps"]:
        raise ValueError(f"artifact has {total_rows} D- rows, frozen {suite} total is {DMINUS_TOTALS[suite]['steps']}")
    kept, ical_manifest = ical_filter(artifact["entries"], ident_fn)
    out = dict(artifact)
    out["entries"] = kept
    out["ical_manifest"] = ical_manifest
    out["dminus_manifest"] = dminus_manifest
    # Copy projection_params so stamping the digest does not mutate the caller's dict.
    out["projection_params"] = dict(artifact.get("projection_params") or {})
    record_weights_digest(out, weights_path)  # -> projection_params.projection_weights_sha256
    return out


# ------------------------------------------------------------------
# Serve hook (§6.3d)
# ------------------------------------------------------------------
def serve_init(artifact: dict, yaml_cfg: dict, weights_path, ident_fn) -> str:
    """Projected-lane serve-time init: enforce the artifact boundary BEFORE any query is keyed.

    (1) require the D- manifest be present (an artifact not built by ``assemble_projected_artifact``
    must not be served); (2) **REJECT** (not silently filter) a served library that contains any
    odd-init or unresolved row -- a served I_cal library with even one I_val (odd) trajectory is a
    self-retrieval leak, so serving is aborted rather than the row dropped; (3) enforce
    recorded-digest <-> live-bytes <-> YAML binding via ``assert_serve_binding``. Returns the
    enforced digest. Any break aborts serving.
    """
    if not artifact.get("dminus_manifest"):
        raise ValueError("projected artifact carries no D- manifest (not assembled via the build chain)")
    _, ical = ical_filter(artifact["entries"], ident_fn)  # inspect, do NOT mutate the served set
    if ical["n_odd_dropped"] or ical["n_unresolved_dropped"]:
        raise ValueError(
            f"served library is not pure I_cal: {ical['n_odd_dropped']} odd-init + "
            f"{ical['n_unresolved_dropped']} unresolved rows present -- aborting serve (odd={ical['odd_idents'][:3]})"
        )
    return assert_serve_binding(artifact, yaml_cfg, weights_path)


# ------------------------------------------------------------------
# Retrieval@K top-1-prefix diagnostic against the REAL strategy (§C)
# ------------------------------------------------------------------
def strategy_topk_prefix_ok(storage, strategy_kwargs: dict, ctx, *, production_k: int = 1, eval_k: int = 5) -> bool:
    """True iff the production ``top_k=1`` result is the prefix of the eval ``top_k=5`` result,
    both produced by the REAL ``DualRetrievalKnnStrategy`` on the same backend + query (§C).

    Takes ONE frozen ``strategy_kwargs`` (the production lane's fusion weights, normalizers,
    filters, LOEO, depth) and constructs both strategies from it, overriding ONLY ``top_k`` --
    so the diagnostic itself proves every other parameter is identical between K=1 and K=5 (the
    reviewer's auditability point), and any ``top_k`` in the passed kwargs is ignored. Unlike
    re-sorting a caller-supplied score list (tautological), this exercises the strategy's real
    filters / weighted score-sum / normalizers / over-fetch-and-slice.
    """
    from openpi.cache.components.search_strategy import DualRetrievalKnnStrategy

    frozen = {k: v for k, v in strategy_kwargs.items() if k != "top_k"}

    def _run(k):
        return [r.id for r in DualRetrievalKnnStrategy(storage, top_k=k, **frozen).search(ctx)]

    prod, wide = _run(production_k), _run(eval_k)
    if not prod or not wide:
        return False
    return wide[:production_k] == prod


# ------------------------------------------------------------------
# Executable entry points (§6.3) -- the REAL build + serve-preflight pipeline
# ------------------------------------------------------------------
def _load_pickle(path):
    import pickle

    with open(path, "rb") as fh:
        return pickle.load(fh)


def _ident_fns(artifact: dict, suite: str, l10_dplus_map: dict):
    """Build (ident_fn, outcome_fn) over an artifact's entries via the SAME resolver the
    trainset build uses (D+ init-map / l10 map, D- gid formula) -- no ad-hoc re-resolution."""
    from exp.zixuan_proposal.build_projection_trainset import _resolve_all

    resolved = _resolve_all(artifact, suite, l10_dplus_map or {})
    return (lambda e: resolved.get(id(e)), lambda e: int(getattr(e, "outcome", 0)))


def _cli_build(a) -> int:
    """`build`: assemble a serve-ready projected artifact from a merged projected pkl + a
    pre-validated D- manifest + provenance table, and write it. Invoked by the §6.3 runbook."""
    import json
    import pickle

    artifact = _load_pickle(a.merged_pkl)
    manifest = json.loads(pathlib.Path(a.dminus_manifest_json).read_text())
    prov = {tuple(x) for x in json.loads(pathlib.Path(a.provenance_json).read_text())}
    l10_map = json.loads(pathlib.Path(a.l10_dplus_map).read_text()) if a.l10_dplus_map else {}
    ident_fn, outcome_fn = _ident_fns(artifact, a.suite, l10_map)

    # A D- row's source H5 basename is its trajectory id (the failure_heldout episode file);
    # step_idx is the per-episode frame index used for the contiguity check.
    def source_fn(e):
        return (e.trajectory_id, int(e.step_idx))

    assembled = assemble_projected_artifact(
        artifact, ident_fn, a.weights, dminus_manifest=manifest,
        provenance_table=prov, suite=a.suite, outcome_fn=outcome_fn, source_fn=source_fn,
    )
    with open(a.out, "wb") as fh:
        pickle.dump(assembled, fh)
    print(f"assembled {a.out}: {len(assembled['entries'])} I_cal rows, "
          f"digest={assembled['projection_params']['projection_weights_sha256'][:12]}...")
    return 0


def _cli_serve_preflight(a) -> int:
    """`serve-preflight`: MANDATORY gate the runbook runs BEFORE launching serve_policy on a
    projected lane -- enforces digest binding + I_cal purity, aborting (exit 2) on any break."""
    import json

    import yaml

    artifact = _load_pickle(a.artifact_pkl)
    yaml_cfg = yaml.safe_load(pathlib.Path(a.yaml).read_text())
    l10_map = json.loads(pathlib.Path(a.l10_dplus_map).read_text()) if a.l10_dplus_map else {}
    ident_fn, _ = _ident_fns(artifact, a.suite, l10_map)
    digest = serve_init(artifact, yaml_cfg, a.weights, ident_fn)
    print(f"serve-preflight OK: bound digest={digest[:12]}...")
    return 0


def main(argv=None) -> int:
    import argparse

    p = argparse.ArgumentParser(description="TRACER Phase-6 projected-artifact build/serve chain")
    sub = p.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="assemble a serve-ready projected artifact")
    b.add_argument("--merged-pkl", required=True)
    b.add_argument("--weights", required=True)
    b.add_argument("--suite", required=True, choices=sorted(DMINUS_TOTALS))
    b.add_argument("--provenance-json", required=True, help="JSON list of [basename, task, init]")
    b.add_argument("--dminus-manifest-json", required=True, dest="dminus_manifest_json")
    b.add_argument("--l10-dplus-map", default=None, dest="l10_dplus_map")
    b.add_argument("--out", required=True)
    b.set_defaults(fn=_cli_build)
    s = sub.add_parser("serve-preflight", help="MANDATORY digest+I_cal gate before serving")
    s.add_argument("--artifact-pkl", required=True, dest="artifact_pkl")
    s.add_argument("--yaml", required=True)
    s.add_argument("--weights", required=True)
    s.add_argument("--suite", required=True, choices=sorted(DMINUS_TOTALS))
    s.add_argument("--l10-dplus-map", default=None, dest="l10_dplus_map")
    s.set_defaults(fn=_cli_serve_preflight)
    a = p.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
