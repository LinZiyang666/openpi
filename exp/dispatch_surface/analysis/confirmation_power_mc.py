"""Full-adjudication power Monte Carlo (confirmation plan 3.4, G1R1-B6 / G1R2-B6,
G2R1-B2 / G2R1-B3).

For every candidate N and outer replicate r, the development cells are
resampled per task WITH replacement into a pseudo-C of N inits/task (paired
across the C roster), the formal inner bootstrap (R = 10000, task-stratified,
paired) is regenerated from the frozen stream
``Generator(PCG64(SeedSequence(INNER_SEED, spawn_key=(N, r))))`` and
``h1_verdict.evaluate_h1_verdict`` -- the same object the confirmation
analyzer calls -- decides pass/fail under the composite rule.

Every (N, r) row carries a canonical digest over its COMPLETE formal content
(verdict, reason, effect, q05, joint miss, support, audit, index digests);
the record's aggregate digest is the SHA-256 of those row digests
concatenated in (N, r) order, so no adjudication value can change without
changing the aggregate. ``validate_power_record`` recomputes the per-N
counts, the Clopper-Pearson bounds, the selection and every digest from the
rows and rejects any record that is not the formal 4 x 200 design bound to
the outcome design, cost map and C roster it names; ``validate_c_roster``
rebuilds the expected roster from the outcome design. ``replay`` recomputes
a digest-derived subset of replicates from the sources and writes a replay
artifact the seal requires.
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import pathlib
import socket
import time

import numpy as np

from exp.dispatch_surface.analysis import budget_outcome_design as bod
from exp.dispatch_surface.analysis.estimator_version import budget_mixture_digest
from exp.dispatch_surface.analysis.h1_verdict import FrozenDesign, evaluate_h1_verdict
from exp.dispatch_surface.phase0_roster import ANCHOR_ARM, F_MIN, M_MAX, FAMILY_S0, FAMILY_SV, FAMILY_THRESHOLD
from exp.dispatch_surface.run_precheck import FORMAL_TRIALS, NUM_TASKS

PROTOCOL = "dispatch_surface_rev2_power_record"
REPLAY_PROTOCOL = "dispatch_surface_rev2_power_replay"
N_CANDIDATES = (30, 40, 50, 60)
R_OUTER = 200
R_OUTER_SMOKE = 20
R_INNER = 10000
OUTER_SEED = 20260830
INNER_SEED = 20260829
LCB_ALPHA = 0.05
POWER_TARGET = 0.80
REPLAY_PER_N = 5
SELECTION_RULE = "smallest N with LCB>=target for N and all larger candidates"
ROW_KEYS = ("N", "r", "formal", "passed", "reason", "effect", "q05", "joint_miss", "left_support_ok",
            "half_effect_proxy_pass", "outer_index_sha256", "inner_index_sha256", "audit_inner0", "row_sha256")
RECORD_KEYS = ("protocol", "smoke", "suite", "estimator_version", "outcome_design_sha256", "c_roster_sha256",
               "budget_cost_map_sha256", "budget_interval", "roster", "constants", "assumption", "per_N", "selection",
               "verdict", "selected_N", "replicates", "aggregate_sha256", "wall_seconds", "half_effect_note")
PER_N_KEYS = ("passes", "n", "power_point", "lcb95", "half_effect_proxy_power", "reasons")
_SHA_LEN = 64


def clopper_pearson_lower(k: int, n: int, alpha: float = LCB_ALPHA) -> float:
    """One-sided lower confidence bound for a binomial proportion."""
    from scipy.stats import beta

    if n <= 0 or k < 0 or k > n:
        raise ValueError("invalid binomial counts")
    if k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))


def select_n(lcb_by_n: dict[int, float], candidates=None, target: float | None = None) -> dict:
    """Smallest N such that N and every larger candidate reach the target.

    ``candidates`` / ``target`` default to the module constants at CALL time
    (never bound at definition time) so the record, the validator and the
    selection always see the same frozen values."""
    cands = sorted(candidates if candidates is not None else N_CANDIDATES)
    target = POWER_TARGET if target is None else target
    ok = {n: lcb_by_n[n] >= target for n in cands}
    chosen = None
    for i, n in enumerate(cands):
        if all(ok[m] for m in cands[i:]):
            chosen = n
            break
    if chosen is None:
        return {"verdict": "underpowered_stop", "selected_N": None, "pass_by_N": ok}
    return {"verdict": "n_selected", "selected_N": chosen, "pass_by_N": ok}


def _canon(obj):
    """JSON round trip with the same float coercion the record is written with."""
    return json.loads(json.dumps(obj, sort_keys=True, default=float))


def _canonical_sha(obj) -> str:
    return hashlib.sha256(json.dumps(_canon(obj), sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _file_sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def outer_rng(N: int, r: int) -> np.random.Generator:
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(OUTER_SEED, spawn_key=(N, r))))


def inner_rng(N: int, r: int) -> np.random.Generator:
    """The frozen inner stream: PCG64 seeded by the SeedSequence itself (G2R1-B2)."""
    return np.random.Generator(np.random.PCG64(np.random.SeedSequence(INNER_SEED, spawn_key=(N, r))))


def pseudo_c(cells: dict, arms: list[str], grid: list, N: int, r: int) -> tuple[dict, list, str]:
    """Outer resample: per task, N cells with replacement (paired over arms)."""
    rng = outer_rng(N, r)
    by_task: dict[int, list] = {}
    for k in grid:
        by_task.setdefault(k[0], []).append(k)
    picks = []
    for t in sorted(by_task):
        lst = by_task[t]
        picks.append([list(lst[j]) for j in rng.integers(0, len(lst), N)])
    pseudo = {a: {} for a in arms}
    for t_pos, t in enumerate(sorted(by_task)):
        for j, orig in enumerate(picks[t_pos]):
            key = (t, j)
            for a in arms:
                pseudo[a][key] = cells[a][tuple(orig)]
    return pseudo, picks, _canonical_sha(picks)


def inner_index(N: int, r: int, *, r_inner: int = R_INNER) -> tuple[np.ndarray, str]:
    """Formal inner bootstrap index for the pseudo-C grid (task-stratified, paired).

    Draw order is frozen: for each replicate, for each task 0..9, ``N`` draws
    ``integers(0, N)`` from the stream of ``inner_rng(N, r)``."""
    rng = inner_rng(N, r)
    cells = [(t, j) for t in range(NUM_TASKS) for j in range(N)]
    cidx = {c: i for i, c in enumerate(cells)}
    rows = []
    for _ in range(r_inner):
        rep = []
        for t in range(NUM_TASKS):
            lst = [(t, j) for j in range(N)]
            for j in rng.integers(0, N, N):
                rep.append(cidx[lst[j]])
        rows.append(rep)
    idx = np.asarray(rows, dtype=np.int64)
    return idx, _canonical_sha(rows)


def row_digest(row: dict) -> str:
    """Canonical digest over the complete formal row (everything but the digest itself)."""
    body = {k: row[k] for k in ROW_KEYS if k != "row_sha256"}
    return _canonical_sha(body)


def aggregate_sha256(rows: list[dict]) -> str:
    ordered = sorted(rows, key=lambda x: (int(x["N"]), int(x["r"])))
    return hashlib.sha256("".join(str(x["row_sha256"]) for x in ordered).encode()).hexdigest()


def one_replicate(payload: dict) -> dict:
    cells = payload["cells"]
    arms = payload["arms"]
    N, r = payload["N"], payload["r"]
    pseudo, _picks, outer_sha = pseudo_c(cells, arms, payload["grid"], N, r)
    # Formal runs always use R_INNER; a smaller inner R is only reachable through
    # an explicit non-formal payload (unit tests) and is stamped on the result.
    r_inner = R_INNER
    if payload.get("nonformal_r_inner"):
        r_inner = int(payload["nonformal_r_inner"])
    idx, inner_sha = inner_index(N, r, r_inner=r_inner)
    des = FrozenDesign(family_a=FAMILY_SV, family_b=FAMILY_THRESHOLD, roster=payload["roster"],
                       B_L=payload["B_L"], B_H=payload["B_H"], R=r_inner)
    v = evaluate_h1_verdict(pseudo, des, idx, audit_replicates=[0])
    half_shift = v.q05 - (0.5 * v.effect if v.effect is not None else 0.0)
    row = {"N": N, "r": r, "formal": r_inner == R_INNER, "passed": v.passed, "reason": v.reason, "effect": v.effect, "q05": v.q05,
           "joint_miss": v.joint_miss, "left_support_ok": v.left_support_ok,
           "half_effect_proxy_pass": bool(v.left_support_ok and v.joint_miss <= des.max_joint_miss and half_shift > 0.0),
           "outer_index_sha256": outer_sha, "inner_index_sha256": inner_sha,
           "audit_inner0": v.audit["replicates"][0] if v.audit["replicates"] else None}
    row = _canon(row)
    row["row_sha256"] = row_digest(row)
    return row


def per_n_stats(rows: list[dict], candidates) -> dict[int, dict]:
    per_n = {}
    for N in candidates:
        sub = [x for x in rows if x["N"] == N]
        k = sum(1 for x in sub if x["passed"])
        kh = sum(1 for x in sub if x["half_effect_proxy_pass"])
        reasons: dict[str, int] = {}
        for x in sub:
            reasons[x["reason"]] = reasons.get(x["reason"], 0) + 1
        per_n[N] = {"passes": k, "n": len(sub), "power_point": k / len(sub), "lcb95": clopper_pearson_lower(k, len(sub)),
                    "half_effect_proxy_power": kh / len(sub), "reasons": reasons}
    return per_n


def frozen_constants(r_outer: int) -> dict:
    return {"N_CANDIDATES": list(N_CANDIDATES), "R_OUTER": r_outer, "R_INNER": R_INNER,
            "OUTER_SEED": OUTER_SEED, "INNER_SEED": INNER_SEED, "LCB_ALPHA": LCB_ALPHA,
            "POWER_TARGET": POWER_TARGET, "REPLAY_PER_N": REPLAY_PER_N, "rule": SELECTION_RULE}


def roster_from_doc(roster_doc: dict) -> dict[str, list[str]]:
    return {fam: [e["arm"] for e in roster_doc["arms"] if e["family"] == fam] for fam in (FAMILY_SV, FAMILY_THRESHOLD)}


# ------------------------------------------------------------------
# validators (shared by the power run, the replay and the seal)
# ------------------------------------------------------------------

def _threshold_pair_from_yaml(path: str) -> list:
    from openpi.cache.config import load_cache_config

    cfg = load_cache_config(path)
    cp1 = cfg.checkpoints.get("cp1")
    if cp1 is None or cp1.judge.type != "threshold":
        raise SystemExit(f"{path}: not a threshold arm")
    tiers = cp1.judge.warm_tiers or []
    return [float(cp1.judge.threshold), float(tiers[0]["threshold"]) if tiers else None]


def validate_c_roster(roster_doc: dict, design: dict, design_path, cost_map: dict, cost_map_path) -> list[dict]:
    """Rebuild the expected C roster from the outcome design and require the
    roster document to equal it arm by arm (family, reasons, active frequency,
    source, delta, digests) with every digest re-read from disk (G2R1-B3)."""
    design_sha = _file_sha(design_path)
    cm_sha = _file_sha(cost_map_path)
    if roster_doc.get("protocol") != bod.PROTOCOL + "_c_roster" or roster_doc.get("schema") != 1:
        raise SystemExit("not a C roster document")
    if roster_doc.get("outcome_design_sha256") != design_sha:
        raise SystemExit("c_roster does not bind this outcome design")
    if design.get("budget_cost_map_sha256") != cm_sha or roster_doc.get("budget_cost_map_sha256") != cm_sha:
        raise SystemExit("c_roster / outcome design do not bind this budget cost map")
    if design.get("estimator_version") != budget_mixture_digest() or roster_doc.get("estimator_version") != budget_mixture_digest():
        raise SystemExit("outcome design / c_roster estimator != budget_mixture_v1")
    for key in ("suite", "verdict", "budget_interval", "library_sha256"):
        if roster_doc.get(key) != design.get(key):
            raise SystemExit(f"c_roster {key} != outcome design")
    if roster_doc.get("F_MIN") != F_MIN or roster_doc.get("M_MAX") != M_MAX or design.get("F_MIN") != F_MIN or design.get("M_MAX") != M_MAX:
        raise SystemExit("c_roster / outcome design selector constants != frozen F_MIN / M_MAX")
    sel = design.get("c_roster_selection") or {}
    expected = []
    for fam in (FAMILY_SV, FAMILY_S0, FAMILY_THRESHOLD):
        block = sel.get(fam) or {}
        for a in block.get("arms") or []:
            expected.append((a, fam, list(block["reasons"][a]), block["active_freq"][a]))
    expected.append((ANCHOR_ARM, "anchor", ["fixed_anchor"], None))
    entries = roster_doc.get("arms") or []
    if len(entries) != len(expected):
        raise SystemExit(f"c_roster has {len(entries)} arms, the outcome design selects {len(expected)}")
    for e, (arm, fam, reasons, freq) in zip(entries, expected):
        if e.get("arm") != arm or e.get("family") != fam or list(e.get("reasons") or []) != reasons or e.get("active_freq") != freq:
            raise SystemExit(f"c_roster entry {e.get('arm')!r} != outcome design selection {arm!r}/{fam}")
        if fam != "anchor":
            if e.get("source") != cost_map["sources"].get(arm) or e.get("delta") != cost_map["deltas"].get(arm):
                raise SystemExit(f"{arm}: c_roster source / delta != budget cost map")
            if e.get("yaml_sha256") != (cost_map["sources"][arm].get("yaml_sha256")):
                raise SystemExit(f"{arm}: c_roster yaml digest != budget cost map source")
        if not isinstance(e.get("yaml_path"), str) or _file_sha(e["yaml_path"]) != e.get("yaml_sha256"):
            raise SystemExit(f"{arm}: c_roster yaml bytes != recorded digest")
        if e.get("artifact_path"):
            if _file_sha(e["artifact_path"]) != e.get("artifact_sha256"):
                raise SystemExit(f"{arm}: c_roster artifact bytes != recorded digest")
        elif e.get("artifact_sha256") is not None:
            raise SystemExit(f"{arm}: artifact digest without an artifact path")
        if fam == FAMILY_THRESHOLD:
            pair = e.get("threshold_pair")
            if pair is None or list(pair) != _threshold_pair_from_yaml(e["yaml_path"]):
                raise SystemExit(f"{arm}: c_roster threshold pair != yaml")
            src_pair = (cost_map["sources"][arm] or {}).get("threshold_pair")
            if src_pair is not None and list(src_pair) != list(pair):
                raise SystemExit(f"{arm}: c_roster threshold pair != cost map source")
        elif e.get("threshold_pair") is not None:
            raise SystemExit(f"{arm}: non-threshold arm carries a threshold pair")
    hyp = design.get("hypotheses") or {}
    roll = roster_doc.get("active_bitset_rollup_sha256") or {}
    if set(roll) != set(hyp) or any(roll[n] != hyp[n].get("active_bitset_rollup_sha256") for n in hyp):
        raise SystemExit("c_roster active-bitset rollups != outcome design")
    return entries


def _is_sha(x) -> bool:
    return isinstance(x, str) and len(x) == _SHA_LEN and all(c in "0123456789abcdef" for c in x)


def validate_power_record(record: dict, *, outcome_design_path, c_roster_path, budget_cost_map_path,
                          allow_smoke: bool = False) -> dict:
    """Mechanically re-derive everything a power record claims (G2R1-B3)."""
    if not isinstance(record, dict) or record.get("protocol") != PROTOCOL:
        raise SystemExit("not a power record")
    if set(record) != set(RECORD_KEYS):
        raise SystemExit(f"power record key set is not exact: extra={sorted(set(record) - set(RECORD_KEYS))} "
                         f"missing={sorted(set(RECORD_KEYS) - set(record))}")
    if record["smoke"] is not False:
        if not allow_smoke or record["smoke"] is not True:
            raise SystemExit("a smoke / non-formal power record is refused here")
    r_outer = R_OUTER_SMOKE if record["smoke"] else R_OUTER
    if record["constants"] != frozen_constants(r_outer):
        raise SystemExit("power record constants != the frozen constants")
    if record["estimator_version"] != budget_mixture_digest():
        raise SystemExit("power record estimator != budget_mixture_v1")
    design = json.loads(pathlib.Path(outcome_design_path).read_text())
    roster_doc = json.loads(pathlib.Path(c_roster_path).read_text())
    cost_map = json.loads(pathlib.Path(budget_cost_map_path).read_text())
    if record["outcome_design_sha256"] != _file_sha(outcome_design_path) or record["c_roster_sha256"] != _file_sha(c_roster_path) \
            or record["budget_cost_map_sha256"] != _file_sha(budget_cost_map_path):
        raise SystemExit("power record outcome design / c_roster / cost map digests drifted")
    if design.get("verdict") != "proceed_to_power":
        raise SystemExit("outcome design verdict is not proceed_to_power")
    validate_c_roster(roster_doc, design, outcome_design_path, cost_map, budget_cost_map_path)
    if record["suite"] != design["suite"] or record["budget_interval"] != design["budget_interval"]:
        raise SystemExit("power record suite / interval != outcome design")
    if record["roster"] != roster_from_doc(roster_doc):
        raise SystemExit("power record roster != c_roster")
    rows = record["replicates"]
    if not isinstance(rows, list) or len(rows) != len(N_CANDIDATES) * r_outer:
        raise SystemExit(f"power record must hold {len(N_CANDIDATES) * r_outer} replicates")
    keys = set()
    for row in rows:
        if not isinstance(row, dict) or tuple(sorted(row)) != tuple(sorted(ROW_KEYS)):
            raise SystemExit("power record row key set is not exact")
        if row["N"] not in N_CANDIDATES or not (0 <= int(row["r"]) < r_outer):
            raise SystemExit(f"row (N={row['N']}, r={row['r']}) outside the frozen design")
        if row["formal"] is not True:
            raise SystemExit(f"row (N={row['N']}, r={row['r']}) is not formal")
        if not isinstance(row["passed"], bool) or not isinstance(row["reason"], str):
            raise SystemExit("row verdict schema invalid")
        if not _is_sha(row["outer_index_sha256"]) or not _is_sha(row["inner_index_sha256"]):
            raise SystemExit("row index digests malformed")
        if row_digest(row) != row["row_sha256"]:
            raise SystemExit(f"row (N={row['N']}, r={row['r']}) digest does not match its content")
        keys.add((int(row["N"]), int(row["r"])))
    if keys != {(N, r) for N in N_CANDIDATES for r in range(r_outer)}:
        raise SystemExit("power record replicates are not exactly one per (N, r)")
    per_n = per_n_stats(rows, N_CANDIDATES)
    got_per_n = record["per_N"]
    if set(got_per_n) != {str(N) for N in N_CANDIDATES}:
        raise SystemExit("per_N keys != candidates")
    for N in N_CANDIDATES:
        g = got_per_n[str(N)]
        if set(g) != set(PER_N_KEYS):
            raise SystemExit(f"per_N[{N}] key set is not exact")
        if _canon(g) != _canon(per_n[N]):
            raise SystemExit(f"per_N[{N}] cannot be recomputed from the rows")
    sel = select_n({N: per_n[N]["lcb95"] for N in N_CANDIDATES})
    if _canon(record["selection"]) != _canon(sel) or record["verdict"] != sel["verdict"] or record["selected_N"] != sel["selected_N"]:
        raise SystemExit("power record selection cannot be recomputed from the rows")
    if sel["selected_N"] is not None and sel["selected_N"] not in N_CANDIDATES:
        raise SystemExit("selected N is not a frozen candidate")
    if record["aggregate_sha256"] != aggregate_sha256(rows):
        raise SystemExit("power record aggregate digest != rows")
    return {"per_N": per_n, "selection": sel, "aggregate_sha256": record["aggregate_sha256"]}


def replay_indices(record: dict) -> list[tuple[int, int]]:
    """Digest-derived replay subset: REPLAY_PER_N outer replicates per candidate N."""
    r_outer = int(record["constants"]["R_OUTER"])
    ss = np.random.SeedSequence(int.from_bytes(bytes.fromhex(record["aggregate_sha256"]), "big"))
    rng = np.random.Generator(np.random.PCG64(ss))
    out = []
    for N in N_CANDIDATES:
        picks = sorted(int(x) for x in rng.choice(r_outer, size=min(REPLAY_PER_N, r_outer), replace=False))
        out.extend((N, r) for r in picks)
    return out


def validate_power_replay(replay: dict, record: dict, record_path) -> None:
    """The seal's check that a replay artifact recomputed this record's rows."""
    if replay.get("protocol") != REPLAY_PROTOCOL:
        raise SystemExit("not a power replay artifact")
    if replay.get("power_record_sha256") != _file_sha(record_path) or replay.get("aggregate_sha256") != record["aggregate_sha256"]:
        raise SystemExit("power replay does not bind this power record")
    expected = replay_indices(record)
    got = [(int(x["N"]), int(x["r"])) for x in replay.get("replayed") or []]
    if got != expected:
        raise SystemExit("power replay subset != the digest-derived subset for this record")
    by_key = {(int(x["N"]), int(x["r"])): x["row_sha256"] for x in record["replicates"]}
    for x in replay["replayed"]:
        key = (int(x["N"]), int(x["r"]))
        if x.get("match") is not True or x.get("expected_row_sha256") != by_key[key] or x.get("replayed_row_sha256") != by_key[key]:
            raise SystemExit(f"power replay row {key} did not reproduce the record")
    if replay.get("passed") is not True or replay.get("constants") != record["constants"]:
        raise SystemExit("power replay did not pass under the record's constants")


# ------------------------------------------------------------------
# run / replay
# ------------------------------------------------------------------

def _load_inputs(args) -> dict:
    if args.trials != FORMAL_TRIALS:
        raise SystemExit(f"formal power MC is frozen at --trials {FORMAL_TRIALS}")
    design = json.loads(pathlib.Path(args.outcome_design).read_text())
    roster_doc = json.loads(pathlib.Path(args.c_roster).read_text())
    if design.get("verdict") != "proceed_to_power":
        raise SystemExit(f"outcome design verdict {design.get('verdict')!r}: power MC refused")
    if design.get("estimator_version") != budget_mixture_digest():
        raise SystemExit("outcome design estimator != budget_mixture_v1")
    cost_map = json.loads(pathlib.Path(args.budget_cost_map).read_text())
    validate_c_roster(roster_doc, design, args.outcome_design, cost_map, args.budget_cost_map)
    src = bod.load_sources(args, cost_map, args.trials)
    roster = roster_from_doc(roster_doc)
    arms = roster[FAMILY_SV] + roster[FAMILY_THRESHOLD]
    return {"design": design, "roster_doc": roster_doc, "cost_map": cost_map, "roster": roster, "arms": arms,
            "cells": {a: src["cells"][a] for a in arms}, "grid": sorted(src["grid"]),
            "B_L": design["budget_interval"]["B_L"], "B_H": design["budget_interval"]["B_H"]}


def _jobs(inp: dict, keys) -> list[dict]:
    return [{"cells": inp["cells"], "arms": inp["arms"], "grid": inp["grid"], "N": N, "r": r, "roster": inp["roster"],
             "B_L": inp["B_L"], "B_H": inp["B_H"]} for N, r in keys]


def _execute(jobs: list[dict], workers: int) -> list[dict]:
    if workers > 1:
        with cf.ProcessPoolExecutor(max_workers=workers) as ex:
            results = list(ex.map(one_replicate, jobs, chunksize=4))
    else:
        results = [one_replicate(j) for j in jobs]
    results.sort(key=lambda x: (x["N"], x["r"]))
    return results


def run(args) -> dict:
    t0 = time.time()
    inp = _load_inputs(args)
    r_outer = R_OUTER_SMOKE if args.smoke else R_OUTER
    results = _execute(_jobs(inp, [(N, r) for N in N_CANDIDATES for r in range(r_outer)]), args.workers)
    per_n = per_n_stats(results, N_CANDIDATES)
    sel = select_n({N: per_n[N]["lcb95"] for N in N_CANDIDATES})
    record = {
        "protocol": PROTOCOL, "smoke": bool(args.smoke), "suite": inp["design"]["suite"],
        "estimator_version": budget_mixture_digest(), "outcome_design_sha256": inp["roster_doc"]["outcome_design_sha256"],
        "c_roster_sha256": _file_sha(args.c_roster),
        "budget_cost_map_sha256": inp["design"]["budget_cost_map_sha256"], "budget_interval": inp["design"]["budget_interval"],
        "roster": inp["roster"], "constants": frozen_constants(r_outer),
        "assumption": ("effect-replication: the empirical joint (cost, SR) distribution of the official A' development cells, "
                       "conditional on the development-selected roster, approximates fresh C; the P pilot only checks "
                       "full-inference difficulty drift, not H1 effect transport"),
        "per_N": {str(N): per_n[N] for N in N_CANDIDATES}, "selection": sel, "verdict": sel["verdict"],
        "selected_N": sel["selected_N"], "replicates": results, "aggregate_sha256": aggregate_sha256(results),
        "wall_seconds": time.time() - t0, "half_effect_note": "proxy: q05 shifted by effect/2; non-gating",
    }
    record = _canon(record)
    p = pathlib.Path(args.out)
    p.write_text(json.dumps(record, indent=2, sort_keys=True))
    validate_power_record(json.loads(p.read_text()), outcome_design_path=args.outcome_design, c_roster_path=args.c_roster,
                          budget_cost_map_path=args.budget_cost_map, allow_smoke=bool(args.smoke))
    print(json.dumps({"verdict": sel["verdict"], "selected_N": sel["selected_N"],
                      "per_N": {str(N): {k: per_n[N][k] for k in ("passes", "n", "lcb95")} for N in N_CANDIDATES}}, indent=2))
    return record


def replay(args) -> dict:
    """Recompute the digest-derived subset of a formal record from the sources."""
    t0 = time.time()
    record = json.loads(pathlib.Path(args.power_record).read_text())
    validate_power_record(record, outcome_design_path=args.outcome_design, c_roster_path=args.c_roster,
                          budget_cost_map_path=args.budget_cost_map)
    inp = _load_inputs(args)
    keys = replay_indices(record)
    by_key = {(int(x["N"]), int(x["r"])): x["row_sha256"] for x in record["replicates"]}
    results = _execute(_jobs(inp, keys), args.workers)
    replayed = []
    for row in results:
        key = (int(row["N"]), int(row["r"]))
        replayed.append({"N": key[0], "r": key[1], "expected_row_sha256": by_key[key], "replayed_row_sha256": row["row_sha256"],
                         "match": by_key[key] == row["row_sha256"]})
    out = {"protocol": REPLAY_PROTOCOL, "power_record_sha256": _file_sha(args.power_record),
           "aggregate_sha256": record["aggregate_sha256"], "constants": record["constants"], "replayed": replayed,
           "passed": all(x["match"] for x in replayed), "host": socket.gethostname(), "wall_seconds": time.time() - t0}
    pathlib.Path(args.out).write_text(json.dumps(out, indent=2, sort_keys=True))
    if not out["passed"]:
        raise SystemExit("power replay FAILED: at least one replicate did not reproduce the record")
    print(json.dumps({"replayed": len(replayed), "passed": out["passed"]}, indent=2))
    return out


def _add_source_args(ap: argparse.ArgumentParser) -> None:
    ap.add_argument("--outcome-design", required=True)
    ap.add_argument("--c-roster", required=True)
    ap.add_argument("--budget-cost-map", required=True)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--phase0-arm-matrix", required=True)
    ap.add_argument("--phase0-launch-manifest", required=True)
    ap.add_argument("--phase0-journal", required=True)
    ap.add_argument("--phase0-per-step", required=True)
    ap.add_argument("--split-manifest", required=True)
    ap.add_argument("--tgrid-package-manifest", default="")
    ap.add_argument("--trials", type=int, default=FORMAL_TRIALS, help=f"frozen at {FORMAL_TRIALS}")
    ap.add_argument("--workers", type=int, default=1, help="process pool size (never changes the result)")
    ap.add_argument("--out", required=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    _add_source_args(r)
    r.add_argument("--smoke", action="store_true", help=f"R_OUTER={R_OUTER_SMOKE}; the record is marked non-formal")
    p = sub.add_parser("replay")
    _add_source_args(p)
    p.add_argument("--power-record", required=True)
    args = ap.parse_args()
    if args.cmd == "run":
        run(args)
    else:
        replay(args)


if __name__ == "__main__":
    main()
