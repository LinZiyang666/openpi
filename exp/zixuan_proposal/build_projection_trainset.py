"""Build the per-suite projection training set (TRACER Phase 6, §A/§C driver).

Ties the tested label/mask/fold core (``projection_labels``) and the identity resolvers
to the real merged dual cache artifact. End to end it:

  1. loads ``cp1_mean_pool_dual.pkl`` (D+ u D-) and OPTIONALLY merges the owner-ruled
     July-D+ control (§9-D1=(b)); refuses to proceed unless the Phase-6.0 batch-separability
     verdict is PASS;
  2. resolves each entry to (task_id, orig_init_state_idx) -- D+ spatial via the init map
     (with a held-out ``full_init_path`` coordinate assertion), D+ l10 via an AUTHORITATIVE
     map (recollected attrs / fingerprint, never collection order), D- via the Phase-4
     provenance formula (episode_id//50, episode_id%50); rejects duplicate identities;
  3. keeps only even ``orig_init_state_idx`` (I_cal), asserting membership against
     ``I_cal_even.json`` and zero intersection with ``I_val_odd.json``;
  4. whitens action chunks + denoise snapshots via the artifact ``library_stats``;
  5. assigns folds (D+ per-task §C rule; D- stable-hash train/val), fits sigma/rho on the
     train+early-stop-val folds ONLY, builds symmetric P/N masks (>=50% valid-anchor guard),
     applies the frozen transform to the mechanism-test fold;
  6. emits ``<suite>.pt`` (per-field raw features + masks + fold assignment) plus a
     provenance sidecar.

``--smoke <suite>`` loads + resolves + reports counts (no .pt), validating the
loading/resolution path against the real artifact without the heavy label build.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import pathlib
import pickle
import re

import numpy as np

from exp.zixuan_proposal.phase6_batch_separability import (
    APRIL,
    JULY,
    episode_feature_digest,
    episode_feature_from_entries,
    gate_input_digest,
)
from exp.zixuan_proposal.projection_labels import (
    MIN_VALID_ANCHOR_FRAC,
    RHO_MINUS_FALLBACK,
    RHO_PLUS_FALLBACK,
    Entry,
    assign_folds,
    build_masks,
    compat_matrix,
    fit_bandwidths,
    fit_thresholds,
    represented_tasks,
    whiten_flatten,
)

_REPO = pathlib.Path(__file__).resolve().parents[2]
_EPISODE_RE = re.compile(r"episode_0*(\d+)")


# ------------------------------------------------------------------
# Loading + identity resolution
# ------------------------------------------------------------------
def _load_pickle(path: pathlib.Path) -> dict:
    if not path.exists():
        raise SystemExit(f"artifact missing: {path}")
    with open(path, "rb") as fh:
        return pickle.load(fh)


def _gid(trajectory_id: str) -> int:
    m = _EPISODE_RE.search(trajectory_id or "")
    if not m:
        raise ValueError(f"cannot parse episode id from {trajectory_id!r}")
    return int(m.group(1))


def _assert_spatial_coord(init_rows: list[dict]) -> None:
    """Prove EVERY spatial init-map row's orig_init_state_idx indexes the held-out pool.

    Each row's ``full_init_path`` must be an existing ``db_init/libero/<suite>/<task>.init``
    (the same 50-state held-out pool the I_cal/I_val manifests index), so the D+
    orig_init_state_idx and the manifest coordinate are provably the same axis.
    """
    for r in init_rows:
        fip = r.get("full_init_path", "")
        if "/db_init/libero/" not in fip or not fip.endswith(".init"):
            raise SystemExit(f"spatial init-map full_init_path not a held-out .init: {fip!r}")
        if not (_REPO / fip).exists():
            raise SystemExit(f"spatial held-out .init missing on disk: {fip!r}")


def _resolve_all(artifact: dict, suite: str, l10_dplus_map: dict) -> dict:
    """Return {id(entry): (task_id, orig_init_state_idx)} for every resolvable entry.

    Keyed by ``id(entry)`` (unique per entry) so no silent overwrite occurs; multiple
    trajectories may legitimately share one (task, init) identity (collection retries) --
    the mask/LOEO logic keys on identity and excludes same-identity pairs, so retries are
    consistent. The trajectory_id -> identity map itself is deduplicated upstream by each
    resolver (init map / recollection attrs).
    """
    init_map_path = _REPO / f"exp/common/data/db/libero_cache/{suite}_init_map.json"
    rows = json.loads(init_map_path.read_text())
    if suite == "libero_spatial":
        _assert_spatial_coord(rows)
    tid_to_init = {r["trajectory_id"]: int(r["orig_init_state_idx"]) for r in rows}
    tid_to_task = {r["trajectory_id"]: int(r["task_id"]) for r in rows}
    ident_of: dict = {}
    for e in artifact["entries"]:
        tid = e.trajectory_id
        if e.outcome == -1:
            gid = _gid(tid)
            ident_of[id(e)] = (gid // 50, gid % 50)
        elif suite == "libero_spatial":
            if tid in tid_to_init:
                ident_of[id(e)] = (tid_to_task[tid], tid_to_init[tid])
        else:  # l10 D+ : authoritative map only, never collection order
            if tid in l10_dplus_map:
                ident_of[id(e)] = l10_dplus_map[tid]
    return ident_of


# ------------------------------------------------------------------
# Folds
# ------------------------------------------------------------------
def _dminus_fold(ident: tuple) -> str:
    """Stable-hash D- fold (plan §C): last-20% by hash -> early-stop-val, else train."""
    h = hashlib.sha1(f"{ident[0]}:{ident[1]}".encode()).hexdigest()
    return "val" if int(h, 16) % 5 == 0 else "train"


def _fold_map(entries: list[Entry]) -> list:
    """PER-ENTRY fold (aligned to ``entries``), so a D+ and D- sharing an identity differ.

    D+ entries follow the per-task §C rule; D- entries are **never** mechanism-test -- a D-
    at a D+ mechanism-test identity is LOEO-``"excluded"`` (dropped by the caller), otherwise
    stable-hash train/val.
    """
    dplus_even = [e.ident for e in entries if e.outcome == 1]
    dplus_fold = dict(assign_folds(dplus_even))  # identity -> train/val/test (D+ only)
    test_idents = {k for k, f in dplus_fold.items() if f == "test"}
    out = []
    for e in entries:
        if e.outcome == 1:
            out.append(dplus_fold.get(e.ident, "train"))
        else:  # D- : never test; excluded if it collides with a D+ mechanism-test identity
            out.append("excluded" if e.ident in test_idents else _dminus_fold(e.ident))
    return out


# ------------------------------------------------------------------
# Whitened entries
# ------------------------------------------------------------------
def _library_stats(artifact: dict):
    stats = artifact.get("library_stats")
    if stats is None:
        raise SystemExit("artifact has no library_stats (D+-only stats required)")
    sigma = np.asarray(getattr(stats, "action_sigma"))
    active = np.asarray(getattr(stats, "action_active_mask"))
    return sigma, active


def _build_entries(artifact: dict, ident_of: dict, active, sigma) -> tuple:
    """Return (entries, feature_rows) over even-init resolved entries, aligned in order.

    ``feature_rows`` is a list of dicts {field: raw pooled query_keys[field]} kept in the
    same order as ``entries`` so per-field trainset features stay row-aligned to the masks.
    """
    entries: list[Entry] = []
    feats: list[dict] = []
    for e in artifact["entries"]:
        ident = ident_of.get(id(e))
        if ident is None or ident[1] % 2 != 0:  # keep even orig_init_state_idx only
            continue
        chunk = np.asarray(e.payload.action_chunk)
        snaps = getattr(e.payload, "intermediates", None) or {}
        entries.append(
            Entry(
                ident=ident,
                outcome=int(e.outcome),
                action_flat=whiten_flatten(chunk, active, sigma),
                snap_flat={float(t): whiten_flatten(np.asarray(v), active, sigma) for t, v in snaps.items()},
            )
        )
        feats.append({f: np.asarray(e.query_keys[f]) for f in ("vision_0", "vision_1")})
    return entries, feats


# ------------------------------------------------------------------
# Full build
# ------------------------------------------------------------------
MIN_REPRESENTED_TASKS = 5


def _read_gate_verdict(path: pathlib.Path) -> dict:
    """Load a PERSISTED Phase-6.0 batch-separability verdict; require a genuine PASS.

    The verdict is produced by ``phase6_batch_separability.run_gate`` and written to disk;
    a caller cannot bypass the owner-ruled §9-D1=(b) gate by typing a status string. Beyond
    the status we re-check the FROZEN §6.0 conditions on the verdict's own numbers AND
    re-verify its tamper-evident ``input_digest`` (so a hand-edited ci/count/cell-manifest
    is caught). The matched-cell manifest is required so the caller can cross-check it against
    the actually-supplied artifacts (``_verify_verdict_binding``).
    """
    if path is None or not pathlib.Path(path).exists():
        raise SystemExit("Phase-6.0 requires a persisted batch-separability verdict (--batch-sep-verdict)")
    v = json.loads(pathlib.Path(path).read_text())
    if v.get("status") != "PASS":
        raise SystemExit(f"batch-separability gate is {v.get('status')!r}, not PASS -- training blocked")
    ci_high = v.get("ci_high")
    # A non-finite (NaN/inf) upper bound is NOT <= 0.55; guard explicitly so a NaN cannot
    # slip through the `> 0.55` comparison (NaN comparisons are always False).
    if ci_high is None or not math.isfinite(float(ci_high)) or float(ci_high) > 0.55:
        raise SystemExit(f"verdict ci_high {ci_high} is not a finite value <= 0.55; the gate did not actually pass")
    if int(v.get("n_april", 0)) < 10 or int(v.get("n_july", 0)) < 10:
        raise SystemExit(f"verdict cell counts below the min-10 floor (april={v.get('n_april')}, july={v.get('n_july')})")
    manifest = v.get("episode_manifest")
    if not manifest:
        raise SystemExit("verdict records no per-episode input manifest; cannot bind it to the supplied artifacts")
    if "input_digest" not in v:
        raise SystemExit("verdict records no input_digest; cannot verify it was not hand-edited")
    recomputed = gate_input_digest(manifest, v["n_april"], v["n_july"], ci_high)
    if recomputed != v["input_digest"]:
        raise SystemExit("verdict input_digest mismatch -- the recorded episodes/metrics were altered after the gate ran")
    return v


def _reconstruct_episodes(entries, ident_of, batch: int) -> dict:
    """{(batch, trajectory_id): (cell, feature_digest)} for every resolvable D+ episode.

    Groups the per-step D+ rows by trajectory_id (one entry per INDEPENDENT episode) and, for
    each, reconstructs the CANONICAL raw mean-pooled vision feature from the episode's rows and
    digests it -- so the returned map carries both the identity AND the classifier feature bytes
    the gate must have evaluated. A ``set`` over trajectory_ids means row multiplicity can never
    inflate the count.
    """
    rows: dict = {}
    for e in entries:
        if int(getattr(e, "outcome", 0)) == 1 and id(e) in ident_of:
            rows.setdefault(e.trajectory_id, []).append(e)
    out = {}
    for tid, es in rows.items():
        cell = ident_of[id(es[0])]
        out[(batch, tid)] = (cell, episode_feature_digest(episode_feature_from_entries(es)))
    return out


def _verify_verdict_binding(verdict: dict, ident_of: dict, control_entries: list, base_entries: list) -> int:
    """Bind the verdict to the EXACT independent episodes AND their classifier feature bytes.

    Reconstructs the per-episode manifest from the base(April D+) and control(July D+) pools --
    one ``[batch, trajectory_id, cell, feature_digest]`` per distinct episode, where the feature
    digest is recomputed from the artifact's actual vision rows -- and requires it to equal the
    verdict's persisted ``episode_manifest`` EXACTLY. This defeats two attacks the earlier checks
    missed: (i) inflation (a verdict claiming ``n_april=n_july=10`` backed by one base + one
    control row fails, reconstructed has 1 per batch); (ii) feature fabrication (a PASS obtained by
    feeding the gate constant features fails, because the recorded feature digest will not match
    the real mean-pooled vision reconstructed here). Also re-checks the recorded counts equal the
    distinct-episode counts. Returns the number of retained INDEPENDENT control episodes at
    even-init (I_cal) cells.
    """
    april = _reconstruct_episodes(base_entries, ident_of, APRIL)
    july = _reconstruct_episodes(control_entries, ident_of, JULY)
    reconstructed = {(b, tid, tuple(cell), fdig) for (b, tid), (cell, fdig) in {**april, **july}.items()}
    want = {(int(b), str(tid), (int(c[0]), int(c[1])), str(fdig)) for (b, tid, c, fdig) in verdict["episode_manifest"]}
    if reconstructed != want:
        missing = sorted((b, tid) for (b, tid, _c, _f) in want - reconstructed)[:3]
        extra = sorted((b, tid) for (b, tid, _c, _f) in reconstructed - want)[:3]
        raise SystemExit(
            "verdict episode_manifest does not correspond to the supplied base/control episodes "
            f"(identity or classifier-feature-digest mismatch; in_verdict_not_artifact={missing}, "
            f"in_artifact_not_verdict={extra})"
        )
    n_april_actual = len(april)
    n_july_actual = len(july)
    if n_april_actual != int(verdict["n_april"]) or n_july_actual != int(verdict["n_july"]):
        raise SystemExit(
            f"verdict counts (april={verdict['n_april']}, july={verdict['n_july']}) != distinct episodes "
            f"in the artifacts (april={n_april_actual}, july={n_july_actual})"
        )
    kept = len({tid for (b, tid), (cell, _f) in july.items() if cell[1] % 2 == 0})
    if kept == 0:
        raise SystemExit("no even-init (I_cal) independent control episode is present in the control artifact")
    return kept


_TS_RE = re.compile(r"episode_\d+_(\d{8})_")


def _verify_control(entries: list, *, expected_yyyymm: str = "202607") -> None:
    """Fail loud unless the control is D+-only AND from the expected (July) collection batch.

    Enforces that the owner-ruled control is genuinely batch-matched July successes, not a
    re-labelled slice of the original April D+ artifact.
    """
    for e in entries:
        if int(getattr(e, "outcome", 0)) != 1:
            raise SystemExit("control artifact must be D+-only (every outcome=+1)")
        m = _TS_RE.search(getattr(e, "trajectory_id", "") or "")
        ym = m.group(1)[:6] if m else None
        if ym != expected_yyyymm:
            raise SystemExit(f"control entry not from the {expected_yyyymm} batch (yyyymm={ym}): {getattr(e, 'trajectory_id', None)}")


def _anchor_valid_frac(pos, neg, anchor_idx: list[int]) -> float:
    """Fraction of the given ANCHOR rows (D+ only) with >=1 positive AND >=1 negative.

    Candidates are ALL rows (positives/negatives may be train, val, or D-); only the D+
    anchor population is the denominator (plan §A), so adding D- candidates cannot dilute it.
    """
    if not anchor_idx:
        return 0.0
    n = pos.shape[0]
    eye = np.eye(n, dtype=bool)
    valid = (pos & ~eye).any(axis=1) & (neg & ~eye).any(axis=1)  # [n]
    return float(valid[np.array(anchor_idx)].mean())


def build_trainset(
    suite: str,
    *,
    eta: float,
    control_artifact_path: pathlib.Path,
    batch_sep_verdict_path: pathlib.Path,
    l10_dplus_map: dict | None = None,
    seed: int = 7,
) -> dict:
    """Build the full per-suite trainset dict (§A/§C). Fails loud on the owner-ruled gates."""
    # Owner ruling §9-D1=(b): a PERSISTED PASS verdict AND the July-D+ control are mandatory,
    # machine-verifiable inputs (not caller-typed strings).
    verdict = _read_gate_verdict(batch_sep_verdict_path)
    if control_artifact_path is None:
        raise SystemExit("Phase-6.0 requires the July-D+ control artifact (--control-artifact)")
    control = _load_pickle(pathlib.Path(control_artifact_path))
    n_control_in = len(control.get("entries", []))
    if n_control_in == 0:
        raise SystemExit("control artifact has no entries")
    _verify_control(control["entries"])  # D+-only AND the expected July batch

    base = _load_pickle(_REPO / f"exp/common/data/cache_artifacts/{suite}/cp1_mean_pool_dual.pkl")
    artifact = {**base, "entries": list(base["entries"]) + list(control["entries"])}

    ident_of = _resolve_all(artifact, suite, l10_dplus_map or {})
    _assert_ical_membership(ident_of)
    sigma, active = _library_stats(base)  # D+-only stats from the ORIGINAL artifact
    entries, feats = _build_entries(artifact, ident_of, active, sigma)
    # Bind the verdict to the EXACT independent episodes in the supplied artifacts: the per-episode
    # manifest reconstructed from base(April D+) + control(July D+) must equal the verdict's, and
    # the recorded counts must equal the distinct-episode counts. Retained control count is the
    # number of independent even-init control episodes (not caller-controlled per-row fields).
    n_control_kept = _verify_verdict_binding(verdict, ident_of, control["entries"], base["entries"])

    if not any(e.outcome == 1 for e in entries):
        raise SystemExit(f"[{suite}] no resolved even-init D+ rows; check the l10 D+ map / control")
    return _finalize(
        suite,
        entries,
        feats,
        eta=eta,
        seed=seed,
        meta_extra={
            "control": {"path": str(control_artifact_path), "n_in": n_control_in, "n_kept": n_control_kept},
            "batch_sep_verdict": {"path": str(batch_sep_verdict_path), "auroc_ci_high": verdict.get("ci_high")},
        },
    )


def _finalize(suite, entries, feats, *, eta, seed, meta_extra=None):
    """Assemble folds + σ/ρ + symmetric masks into a trainset dict, enforcing §A/§C guards.

    Split out from ``build_trainset`` so the label/mask/fold assembly is testable on
    synthetic entries without the real artifact load. Fails loud on every completeness /
    leakage invariant (valid-anchor fraction, represented tasks, non-empty mechanism-test).
    """
    fold = _fold_map(entries)  # per-entry list
    keep = [i for i, f in enumerate(fold) if f != "excluded"]  # drop LOEO-excluded D-
    entries = [entries[i] for i in keep]
    feats = [feats[i] for i in keep]
    fold = [fold[i] for i in keep]

    trainval_entries = [entries[i] for i, f in enumerate(fold) if f in ("train", "val")]
    # D+ anchor indices (only D+ rows are anchors, plan §A).
    tv_anchors = [i for i, (f, e) in enumerate(zip(fold, entries)) if f in ("train", "val") and e.outcome == 1]
    val_anchors = [i for i, (f, e) in enumerate(zip(fold, entries)) if f == "val" and e.outcome == 1]

    bw = fit_bandwidths(trainval_entries, eta=eta, seed=seed)
    c = compat_matrix(entries, bw)  # full even-set matrix (frozen transform)
    # rho fitted on train+val D+ ONLY, then applied unchanged to the whole set.
    rp, rm = fit_thresholds(trainval_entries, compat_matrix(trainval_entries, bw))
    pos, neg = build_masks(entries, c, rp, rm)
    # Fallback decision on the EARLY-STOP-VAL anchors only (plan §A / G2R3).
    if _anchor_valid_frac(pos, neg, val_anchors) < MIN_VALID_ANCHOR_FRAC:
        rp, rm = RHO_PLUS_FALLBACK, RHO_MINUS_FALLBACK
        pos, neg = build_masks(entries, c, rp, rm)
    # Final guard over the D+ train+val ANCHOR population (D- candidates cannot dilute it).
    frac = _anchor_valid_frac(pos, neg, tv_anchors)
    if frac < MIN_VALID_ANCHOR_FRAC:
        raise SystemExit(f"[{suite}] D+ valid-anchor fraction {frac:.3f} < {MIN_VALID_ANCHOR_FRAC} even after fallback")

    rep = sorted({e.ident[0] for f, e in zip(fold, entries) if f == "test"})
    if len(rep) < MIN_REPRESENTED_TASKS:
        raise SystemExit(f"[{suite}] only {len(rep)} represented tasks (< {MIN_REPRESENTED_TASKS})")
    if "test" not in fold:
        raise SystemExit(f"[{suite}] mechanism-test fold is empty")

    fields = {f: {"features": np.stack([row[f] for row in feats])} for f in ("vision_0", "vision_1")}
    meta = {
        "eta": eta,
        "sigma_A_sq": bw["sigma_A_sq"],
        "sigma_X_sq": bw.get("sigma_X_sq"),
        "rho_plus": rp,
        "rho_minus": rm,
        "valid_anchor_frac": frac,
        "represented_tasks": rep,
        "n_entries": len(entries),
        "seed": seed,
        **(meta_extra or {}),
    }
    return {
        "suite": suite,
        "rows": [{"ident": list(e.ident), "outcome": e.outcome, "fold": f} for f, e in zip(fold, entries)],
        "fields": fields,
        "masks": {"pos": pos, "neg": neg},
        "meta": meta,
    }


def _assert_ical_membership(ident_of: dict) -> None:
    """The KEPT (even-init) identities must all be in I_cal_even and none in I_val_odd.

    Odd-init identities also appear in the resolved set (the library spans both parities);
    they are simply filtered out downstream, so they are NOT a leakage -- only the even
    (kept) set is validated here.
    """
    ical = {(r["task_id"], r["orig_init_state_idx"]) for r in json.loads(_FILTER("I_cal_even").read_text())}
    ival = {(r["task_id"], r["orig_init_state_idx"]) for r in json.loads(_FILTER("I_val_odd").read_text())}
    for ident in ident_of.values():
        if ident[1] % 2 != 0:
            continue  # odd-init entries are filtered out, not kept -> not leakage
        if ident not in ical:
            raise SystemExit(f"even identity {ident} not in I_cal_even manifest")
        if ident in ival:
            raise SystemExit(f"kept even identity {ident} is in I_val_odd -- leakage")


def _FILTER(name: str) -> pathlib.Path:
    return _REPO / f"exp/zixuan_proposal/data/filters/{name}.json"


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser(description="Build/inspect the projection training set")
    ap.add_argument("suite", choices=["libero_spatial", "libero_10"])
    ap.add_argument("--smoke", action="store_true", help="load+resolve+report counts only")
    ap.add_argument("--eta", type=float, default=1.0)
    ap.add_argument("--control-artifact", type=str, default=None, help="July-D+ control pkl (§6.0, required)")
    ap.add_argument("--batch-sep-verdict", type=str, default=None, help="persisted PASS verdict json (§6.0)")
    ap.add_argument("--l10-map", type=str, default=None, help="l10 D+ trajectory->identity map json")
    ap.add_argument("--out", type=str, default=None, help="output <suite>.pt path")
    args = ap.parse_args()

    if args.smoke:
        artifact = _load_pickle(_REPO / f"exp/common/data/cache_artifacts/{args.suite}/cp1_mean_pool_dual.pkl")
        ident_of = _resolve_all(artifact, args.suite, {})
        idents_even_dplus = [
            ident_of[id(e)]
            for e in artifact["entries"]
            if e.outcome == 1 and id(e) in ident_of and ident_of[id(e)][1] % 2 == 0
        ]
        fold = assign_folds(idents_even_dplus)
        print(f"[{args.suite}] entries={len(artifact['entries'])} resolved={len(ident_of)}")
        print(f"  even-D+ identities={len(set(idents_even_dplus))}; represented tasks={sorted(represented_tasks(fold))}")
        return

    import torch

    l10_map = None
    if args.l10_map:
        raw = json.loads(pathlib.Path(args.l10_map).read_text())
        l10_map = {k: tuple(v) for k, v in raw.items()}
    ts = build_trainset(
        args.suite,
        eta=args.eta,
        control_artifact_path=pathlib.Path(args.control_artifact) if args.control_artifact else None,
        batch_sep_verdict_path=pathlib.Path(args.batch_sep_verdict) if args.batch_sep_verdict else None,
        l10_dplus_map=l10_map,
    )
    out = pathlib.Path(args.out or (_REPO / f"exp/zixuan_proposal/data/projection_trainset/{args.suite}.pt"))
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(ts, out)
    sidecar = out.with_suffix(".provenance.json")
    sidecar.write_text(json.dumps(ts["meta"], indent=2, default=float))
    print(f"wrote {out} (n={ts['meta']['n_entries']}, valid_anchor_frac={ts['meta']['valid_anchor_frac']:.3f})")


if __name__ == "__main__":
    main()
