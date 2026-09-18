"""Aggregate the x0-head x label-filtered-data experiment and apply the pre-registered decision rules.

Inputs
------
* the frozen cell matrix ``cells_manifest.json`` (+ the cell yamls next to it) written by ``x0_cells.py`` -- the list of
  *expected* cells per arm (core / explore / image) and the tasks skipped with a reason;
* one ``summary.json`` per (cell, split, sampler) under ``<root>/<cell_id>/<split>_<sampler>_<k>/`` written by
  ``eval_dp_steps_v2.py``::

    {"cell": {identity bound to the checkpoint: task_name, modality, variant, head, train_seed, budget_id, ...},
     "sampler": {"sampler", "k", "head", "timesteps", "nfe_per_call", "eps_mode", "protocol_id": "trailing_v1", ...},
     "split": "screen"|"test", "start_seed", "n_test", "episode_ids_expected": [...], "episodes": {id: score},
     "complete": true, "error": null, "manifest": {"checkpoint_sha256": ...}}

Validation (G2 R1-B8): a record is *valid* only if it is ``complete`` without error, its protocol id / grid / NFE / head /
eps_mode equal the frozen protocol, its episode ids are exactly the split's frozen id set (screen 90000+32, test 100000+n),
its scores are finite, its identity names an expected cell with the expected budget_id, and every record of the same cell
carries the same checkpoint hash. Invalid or unexpected records are listed, never used.

Decision rules (plan §4.5): the inference unit is the episode id; for every episode the scores of all paired cells are
averaged over the pre-registered training seeds first, then episodes are resampled as a block B times; intervals are
two-sided percentile intervals at 1 - 0.05/16 (Bonferroni over the family of 16 = 4 tasks x {S_x0, I, D1(eps,U),
D1(x0,U)}). The main anchor is DDIM-100 (trailing); D1(h, d) = Q_ddim100 - Q_ddim1.
    H1-data : S_x0 = D1(x0, M) - D1(x0, U)      supported if lower > 0.05; not supported if upper <= 0.05; else inconclusive
    H2-head : D1(eps, U) lower > 0.20 AND D1(x0, U) upper < 0.05 -> supported;
              D1(eps, U) upper <= 0.20 OR D1(x0, U) lower >= 0.05 -> not supported; else inconclusive
    I       : S_x0 - S_eps                      supported if lower > 0.05; not supported if upper <= 0.05; else inconclusive
Formal verdicts exist only for the **core lowdim** tasks with exactly the manifest's three training seeds, and only when
every required record (4 cells x 3 seeds x {six test samplers + screen ddim_100}) is valid and every screening
Q >= 0.5. Explore (``variant=full``), image and official-checkpoint records are summarised descriptively (Q ladders;
image U/M pairs also get descriptive, non-preregistered D1/S estimates) and never enter the formal family (G2 R1-B9).

usage: python -m exp.dp_nfe.analysis.aggregate_x0 --cells <cells dir with cells_manifest.json> \
           --root <results root> --out <decisions.json> [--boot 20000] [--seed 20260918] [--image-n-test 50]
"""

from __future__ import annotations

import argparse
import json
import math
import pathlib
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import yaml

from exp.dp_nfe.dp_sampler import make_timesteps
from exp.dp_nfe.x0_identity import CELL_KEY_FIELDS, cell_key, frozen_identity_diff, sha256_file

PROTOCOL_ID = "trailing_v1"
EPS_MODE = "recompute"
T_TRAIN = 100
HEADS = ("epsilon", "sample")
VARIANTS = ("U", "M")
ANCHOR = "ddim_100"
ONE_STEP = "ddim_1"
LADDER = ("ddpm_100", "ddim_100", "ddim_10", "ddim_4", "ddim_2", "ddim_1")
FAMILY_SIZE = 16
ALPHA = 0.05
DELTA = 0.05  # pre-registered substantive effect for S_x0 and I
H2_EPS_MIN = 0.20  # eps head must lose at least this on U
H2_X0_MAX = 0.05  # x0 head may lose at most this on U
SCREEN_MIN_Q = 0.5
SPLITS = {"screen": {"start_seed": 90000, "n": 32}, "test": {"start_seed": 100000, "n": 100}}
IMAGE_TEST_N = 50


class AggregationError(ValueError):
    """Raised for inputs the aggregator refuses to interpret (missing manifest / cell yaml, invalid n)."""
    pass


def wilson(successes: int, n: int, z: float = 1.959964) -> Tuple[float, float]:
    """Wilson score interval of a binomial proportion (``n`` must be positive)."""
    if n <= 0:
        raise AggregationError("wilson: n must be positive")
    p = successes / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def sampler_key(sampler: str, k: int) -> str:
    """``<sampler>_<k>`` key of an arm (``ddim_100``, ``ddpm_100``, ...)."""
    return f"{sampler}_{int(k)}"


def expected_grid(sampler: str, k: int) -> List[int]:
    """The frozen trailing grid of an arm (DDPM anchor: all T steps)."""
    if sampler == "ddpm":
        return list(range(T_TRAIN - 1, -1, -1))
    return list(make_timesteps(T_TRAIN, int(k)))


def expected_ids(split: str, n_test: int) -> List[str]:
    """The frozen episode ids of ``split``: ``start_seed .. start_seed + n_test - 1`` as strings."""
    return [str(SPLITS[split]["start_seed"] + i) for i in range(int(n_test))]


def load_results(root: pathlib.Path) -> List[dict]:
    """Every ``<root>/*/*/summary.json`` as a dict with ``_path``; unreadable files become ``{"_path", "_error"}``."""
    recs = []
    for f in sorted(root.glob("*/*/summary.json")):
        try:
            d = json.loads(f.read_text())
            if not isinstance(d, dict):
                raise ValueError("summary must be a JSON object")
        except Exception as e:  # noqa: BLE001
            d = {"_error": f"unreadable: {e}"}
        d["_path"] = str(f)
        recs.append(d)
    return recs


def load_cells(cells_dir: pathlib.Path) -> Tuple[dict, Dict[str, dict]]:
    """``(cells_manifest, {cell_id: cell yaml})`` from the frozen cell directory (``AggregationError`` when missing)."""
    man = cells_dir / "cells_manifest.json"
    if not man.is_file():
        raise AggregationError(f"{man} missing")
    manifest = json.loads(man.read_text())
    if manifest.get("train_seeds") != [42, 43, 44]:
        raise AggregationError("formal matrix requires exactly train_seeds [42, 43, 44]")
    listed = [c for arm in ("core", "explore", "image") for c in manifest["by_arm"][arm]]
    if len(listed) != len(set(listed)) or sorted(listed) != sorted(manifest["cells"]):
        raise AggregationError("manifest arms must partition the unique cell list")
    cells = {}
    for cid in manifest["cells"]:
        p = cells_dir / f"{cid}.yaml"
        if not p.is_file():
            raise AggregationError(f"cell yaml {p} missing")
        cells[cid] = yaml.safe_load(p.read_text())
        cells[cid]["_yaml_sha256"] = sha256_file(p)
    for cid in manifest["by_arm"]["core"]:
        c = cells[cid]["identity"]
        if (c["modality"] != "lowdim" or c["variant"] not in VARIANTS
                or c["task_name"] not in ("pusht", "blockpush", "kitchen", "square_mh")):
            raise AggregationError(f"non-preregistered core cell: {cid}")
    return manifest, cells


def check_record(r: dict, cells: Mapping[str, dict], image_n_test: int = IMAGE_TEST_N) -> Tuple[Optional[str], Optional[str]]:
    """Validate an untrusted summary without letting malformed types abort the ledger."""
    try:
        return _check_record(r, cells, image_n_test)
    except (ValueError, TypeError, KeyError, AttributeError, OverflowError) as e:
        c = r.get("cell") if isinstance(r, dict) else None
        cid = cell_key(c) if isinstance(c, dict) and all(k in c for k in CELL_KEY_FIELDS) else None
        return cid, f"malformed record: {e}"


def _check_record(r: dict, cells: Mapping[str, dict], image_n_test: int) -> Tuple[Optional[str], Optional[str]]:
    """Return ``(cell_id, None)`` for a valid record or ``(cell_id or None, reason)`` for an invalid one. ``cell_id`` is
    the cell key derived from the record's identity; a record whose key is not in ``cells`` is 'unexpected' unless it is
    an official checkpoint (``variant=official``), which is validated against the protocol but never against the matrix."""
    if "_error" in r:
        return None, r["_error"]
    c = r.get("cell") or {}
    s = r.get("sampler") or {}
    for key in CELL_KEY_FIELDS:
        if key not in c:
            return None, f"cell.{key} missing"
    cid = cell_key(c)
    official = c.get("variant") == "official"
    if not official and cid not in cells:
        return cid, "unexpected cell (not in cells_manifest)"
    if not official:
        cy = cells[cid]
        differences = frozen_identity_diff(c, cy)
        if differences:
            return cid, "frozen identity differs: " + "; ".join(differences)
    if r.get("complete") is not True:
        return cid, "not complete"
    if r.get("error"):
        return cid, f"error recorded: {str(r['error'])[:80]}"
    if s.get("protocol_id") != PROTOCOL_ID:
        return cid, f"protocol {s.get('protocol_id')!r} != {PROTOCOL_ID!r}"
    if s.get("sampler") not in ("ddim", "ddpm") or "k" not in s:
        return cid, "sampler/k missing"
    k = int(s["k"])
    if sampler_key(s["sampler"], k) not in LADDER:
        return cid, "sampler/k outside the frozen ladder"
    if s.get("T") != T_TRAIN or s.get("clip_sample") is not True:
        return cid, "T/clip_sample differs from the frozen protocol"
    if list(s.get("timesteps") or []) != expected_grid(s["sampler"], k):
        return cid, f"grid differs from the frozen trailing grid for {s['sampler']}-{k}"
    if int(s.get("nfe_per_call", -1)) != len(expected_grid(s["sampler"], k)):
        return cid, "nfe_per_call differs from the grid length"
    if s.get("head") != c.get("head") or c.get("head") not in HEADS:
        return cid, f"sampler head {s.get('head')} != cell head {c.get('head')}"
    if s.get("eps_mode") != EPS_MODE:
        return cid, f"eps_mode {s.get('eps_mode')} != {EPS_MODE}"
    split = r.get("split")
    if split not in SPLITS:
        return cid, f"split {split!r} is not screen|test"
    if split == "screen" and sampler_key(s["sampler"], k) != ANCHOR:
        return cid, "screening requires ddim_100"
    if s.get("sampling_seed") != 0:
        return cid, "sampling_seed differs from the frozen seed 0"
    n_expected = SPLITS[split]["n"] if not (split == "test" and c.get("modality") == "image") else image_n_test
    if int(r.get("n_test", -1)) != n_expected or int(r.get("start_seed", -1)) != SPLITS[split]["start_seed"]:
        return cid, f"{split}: n_test/start_seed {r.get('n_test')}/{r.get('start_seed')} != {n_expected}/{SPLITS[split]['start_seed']}"
    ids = expected_ids(split, n_expected)
    eps = r.get("episodes") or {}
    if sorted(eps) != sorted(ids) or list(r.get("episode_ids_expected") or []) != ids:
        return cid, f"{split}: episode id set differs from the frozen set"
    for e, v in eps.items():
        if v is None or not math.isfinite(float(v)):
            return cid, f"non-finite score for episode {e}"
        if not 0 <= float(v) <= 1:
            return cid, f"score outside [0, 1] for episode {e}"
    if not (r.get("manifest") or {}).get("checkpoint_sha256"):
        return cid, "checkpoint_sha256 missing"
    return cid, None


def validate(recs: Iterable[dict], cells: Mapping[str, dict], image_n_test: int = IMAGE_TEST_N) -> Tuple[dict, dict]:
    """Index valid records as ``table[cell_id][split][sampler_key] -> {episode: score}`` and return
    ``(table, report)`` where ``report = {"valid": n, "invalid": [{path, cell, reason}], "unexpected": [...],
    "official": {cell_id: {split: {sampler_key: scores}}}}``. Duplicates and checkpoint-hash conflicts within a cell
    invalidate the entire conflicting slot / cell, including any earlier accepted records."""
    table: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    official: Dict[str, Dict[str, Dict[str, Dict[str, float]]]] = {}
    ckpt: Dict[str, str] = {}
    poisoned_cells = set()
    poisoned_slots = set()
    identities = {}
    report = {"valid": 0, "invalid": [], "unexpected": []}
    for r in recs:
        cid, reason = check_record(r, cells, image_n_test)
        if reason is not None:
            (report["unexpected"] if reason.startswith("unexpected") else report["invalid"]).append(
                {"path": r.get("_path"), "cell": cid, "reason": reason})
            attributed = cid
            if attributed not in cells and r.get("_path"):
                attributed = pathlib.Path(r["_path"]).parent.parent.name
            if attributed in cells:
                s = r.get("sampler") or {}
                try:
                    poisoned_slots.add((attributed, r.get("split"), sampler_key(s["sampler"], s["k"])))
                except (KeyError, ValueError, TypeError):
                    poisoned_cells.add(attributed)
            continue
        s = r["sampler"]; sk = sampler_key(s["sampler"], s["k"]); split = r["split"]
        dest = official if r["cell"]["variant"] == "official" else table
        sha = r["manifest"]["checkpoint_sha256"]
        if cid in ckpt and ckpt[cid] != sha:
            report["invalid"].append({"path": r.get("_path"), "cell": cid, "reason": "checkpoint_sha256 differs between records of the same cell"})
            poisoned_cells.add(cid)
        if cid in identities and identities[cid] != r["cell"]:
            report["invalid"].append({"path": r.get("_path"), "cell": cid, "reason": "identity differs between records of the same cell"})
            poisoned_cells.add(cid)
        identities[cid] = r["cell"]
        ckpt[cid] = sha
        slot = dest.setdefault(cid, {}).setdefault(split, {})
        if sk in slot:
            report["invalid"].append({"path": r.get("_path"), "cell": cid, "reason": f"duplicate result {split} {sk}"})
            poisoned_slots.add((cid, split, sk))
            continue
        slot[sk] = {str(e): float(v) for e, v in r["episodes"].items()}
    for dest in (table, official):
        for cid in poisoned_cells:
            dest.pop(cid, None)
        for cid, split, sk in poisoned_slots:
            dest.get(cid, {}).get(split, {}).pop(sk, None)
    report["valid"] = sum(len(arms) for dest in (table, official) for splits in dest.values() for arms in splits.values())
    report["official"] = official
    return table, report


def cell_ids_for(task: str, modality: str, variant: str, head: str, seed: int, budget_id: str) -> str:
    """cell_id of one matrix cell (x0_cells convention)."""
    return cell_key({"task_name": task, "modality": modality, "variant": variant, "head": head, "train_seed": seed,
                     "budget_id": budget_id})


def required_records(task: str, modality: str, seeds: Sequence[int], budget_id: str) -> List[Tuple[str, str, str]]:
    """``(cell_id, split, sampler_key)`` triples a formal verdict needs."""
    req = []
    for v in VARIANTS:
        for h in HEADS:
            for seed in seeds:
                cid = cell_ids_for(task, modality, v, h, int(seed), budget_id)
                req += [(cid, "test", sk) for sk in LADDER] + [(cid, "screen", ANCHOR)]
    return req


def missing_records(table, req: Sequence[Tuple[str, str, str]]) -> List[str]:
    """``cell/split/sampler`` strings of the required records absent from ``table``."""
    return [f"{cid}/{split}/{sk}" for cid, split, sk in req if sk not in table.get(cid, {}).get(split, {})]


def _episode_matrix(table, task: str, modality: str, seeds: Sequence[int], budget_id: str, samplers: Sequence[str]
                    ) -> Tuple[List[str], Dict[Tuple[str, str, str], np.ndarray]]:
    """Common episode ids and, for every (variant, head, sampler), the seed-averaged score vector."""
    ids: Optional[set] = None
    per: Dict[Tuple[str, str, str], List[np.ndarray]] = {}
    ordered: List[str] = []
    for v in VARIANTS:
        for h in HEADS:
            for seed in seeds:
                cid = cell_ids_for(task, modality, v, h, int(seed), budget_id)
                for sk in samplers:
                    e = table.get(cid, {}).get("test", {}).get(sk)
                    if e is None:
                        raise AggregationError(f"missing {cid}/test/{sk}")
                    if ids is None:
                        ids = set(e); ordered = sorted(ids)
                    elif set(e) != ids:
                        raise AggregationError(f"episode ids differ for {cid} {sk} (pairing broken)")
                    per.setdefault((v, h, sk), []).append(np.array([e[i] for i in ordered], dtype=np.float64))
    mats = {k: np.mean(np.stack(vs, 0), 0) for k, vs in per.items()}  # average over training seeds first
    return ordered, mats


def screening_passed(table, task: str, modality: str, seeds: Sequence[int], budget_id: str) -> Tuple[bool, List[str]]:
    """Gate: every (variant, head, seed) cell has a screening DDIM-100 record with mean Q >= SCREEN_MIN_Q."""
    reasons = []
    for v in VARIANTS:
        for h in HEADS:
            for seed in seeds:
                cid = cell_ids_for(task, modality, v, h, int(seed), budget_id)
                scr = table.get(cid, {}).get("screen", {}).get(ANCHOR)
                if scr is None:
                    reasons.append(f"no screening for {cid}")
                    continue
                q = float(np.mean(list(scr.values())))
                if q < SCREEN_MIN_Q:
                    reasons.append(f"screening Q={q:.3f} < {SCREEN_MIN_Q} for {cid}")
    return (not reasons), reasons


def bootstrap_indices(n: int, boot: int, seed: int) -> np.ndarray:
    """``[boot, n]`` episode-index resamples (with replacement) from a fixed analysis seed; shared by every quantity."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, n, size=(boot, n))


def stats_from_mats(mats: Mapping[Tuple[str, str, str], np.ndarray], idx: np.ndarray, formal: bool = True) -> Dict[str, Dict[str, float]]:
    """Percentile intervals: the four formal quantities use Bonferroni; descriptive quantities use 95%."""
    def Q(v, h, sk, ix=None):
        x = mats[(v, h, sk)]
        return x.mean() if ix is None else x[ix].mean(-1)
    def d1(h, v, ix=None):
        return Q(v, h, ANCHOR, ix) - Q(v, h, ONE_STEP, ix)
    out = {}
    def rec(name, point, samples):
        q = ALPHA / FAMILY_SIZE if formal and name in ("S_x0", "I", "D1_epsilon_U", "D1_sample_U") else ALPHA
        lo, hi = np.quantile(samples, [q / 2, 1 - q / 2])
        out[name] = {"point": float(point), "lower": float(lo), "upper": float(hi), "level": 1 - q}
    for h in HEADS:
        for v in VARIANTS:
            rec(f"D1_{h}_{v}", d1(h, v), d1(h, v, idx))
            rec(f"Q_{h}_{v}_{ANCHOR}", Q(v, h, ANCHOR), Q(v, h, ANCHOR, idx))
            rec(f"Q_{h}_{v}_{ONE_STEP}", Q(v, h, ONE_STEP), Q(v, h, ONE_STEP, idx))
    s_x0 = d1("sample", "M") - d1("sample", "U"); s_x0_b = d1("sample", "M", idx) - d1("sample", "U", idx)
    s_eps = d1("epsilon", "M") - d1("epsilon", "U"); s_eps_b = d1("epsilon", "M", idx) - d1("epsilon", "U", idx)
    rec("S_x0", s_x0, s_x0_b); rec("S_eps", s_eps, s_eps_b); rec("I", s_x0 - s_eps, s_x0_b - s_eps_b)
    return out


def decide(st: Mapping[str, Mapping[str, float]]) -> Dict[str, str]:
    """Apply the pre-registered rules to the interval table of :func:`stats_from_mats`."""
    def rule(name, thr):
        lo, hi = st[name]["lower"], st[name]["upper"]
        if lo > thr:
            return "supported"
        if hi <= thr:
            return "not_supported"
        return "inconclusive"
    h1 = rule("S_x0", DELTA)
    i = rule("I", DELTA)
    e_lo, e_hi = st["D1_epsilon_U"]["lower"], st["D1_epsilon_U"]["upper"]
    x_lo, x_hi = st["D1_sample_U"]["lower"], st["D1_sample_U"]["upper"]
    if e_lo > H2_EPS_MIN and x_hi < H2_X0_MAX:
        h2 = "supported"
    elif e_hi <= H2_EPS_MIN or x_lo >= H2_X0_MAX:
        h2 = "not_supported"
    else:
        h2 = "inconclusive"
    equiv = abs(st["S_x0"]["lower"]) <= DELTA and abs(st["S_x0"]["upper"]) <= DELTA
    return {"H1_data": h1, "H2_head": h2, "I": i, "S_x0_equivalent_within_delta": "yes" if equiv else "no"}


def ladder(table_cell: Mapping[str, Mapping[str, Mapping[str, float]]]) -> Dict[str, dict]:
    """Descriptive Q ladder of one cell: ``{split: {sampler_key: {"mean", "n", "wilson": [lo, hi] (success-rate
    tasks only, i.e. all scores in {0, 1})}}}``."""
    out = {}
    for split, arms in table_cell.items():
        out[split] = {}
        for sk, eps in arms.items():
            vals = np.array(list(eps.values()), dtype=np.float64)
            rec = {"mean": float(vals.mean()), "n": int(vals.size)}
            if np.all(np.isin(vals, (0.0, 1.0))):
                rec["wilson"] = list(wilson(int(vals.sum()), int(vals.size)))
            out[split][sk] = rec
    return out


def core_ladders(table, task: str, modality: str, seeds: Sequence[int], budget_id: str) -> Dict[str, Dict[str, float]]:
    """Descriptive seed-averaged test Q per (head_variant, sampler) over whatever valid test records exist (a sampler is
    listed only when every seed has it)."""
    out = {}
    for v in VARIANTS:
        for h in HEADS:
            arms = {}
            for sk in LADDER:
                vals = []
                for seed in seeds:
                    e = table.get(cell_ids_for(task, modality, v, h, int(seed), budget_id), {}).get("test", {}).get(sk)
                    if e is None:
                        break
                    vals.append(float(np.mean(list(e.values()))))
                else:
                    arms[sk] = float(np.mean(vals))
            out[f"{h}_{v}"] = arms
    return out


def evaluate_task(table, task: str, modality: str, seeds: Sequence[int], budget_id: str, boot: int, seed: int) -> dict:
    """Formal evaluation of one core task: ``status`` complete|incomplete, the missing required records, the screening
    gate, the interval table and the verdict (``None`` unless every requirement holds)."""
    if modality != "lowdim" or list(seeds) != [42, 43, 44]:
        raise AggregationError("formal verdicts require lowdim and exactly seeds [42, 43, 44]")
    req = required_records(task, modality, seeds, budget_id)
    missing = missing_records(table, req)
    res = {"task": task, "modality": modality, "seeds": list(seeds), "budget_id": budget_id, "required": len(req),
           "missing": missing, "status": "complete" if not missing else "incomplete",
           "ladders": core_ladders(table, task, modality, seeds, budget_id)}
    if missing:
        res.update({"verdict": None, "reason": f"incomplete: {len(missing)} of {len(req)} required records missing"})
        return res
    ok, reasons = screening_passed(table, task, modality, seeds, budget_id)
    res.update({"screening_passed": ok, "screening_reasons": reasons})
    ids, mats = _episode_matrix(table, task, modality, seeds, budget_id, (ANCHOR, ONE_STEP))
    idx = bootstrap_indices(len(ids), boot, seed)
    st = stats_from_mats(mats, idx)
    res["n_episodes"] = len(ids)
    res["stats"] = st
    res["per_seed"] = {}
    for train_seed in seeds:
        _, single = _episode_matrix(table, task, modality, [train_seed], budget_id, (ANCHOR, ONE_STEP))
        res["per_seed"][str(train_seed)] = stats_from_mats(single, idx, formal=False)
    res["seed_point_ranges"] = {
        name: [min(s[name]["point"] for s in res["per_seed"].values()), max(s[name]["point"] for s in res["per_seed"].values())]
        for name in st
    }
    if not ok:
        res.update({"verdict": None, "reason": "screening gate failed; estimates reported, no formal verdict"})
    else:
        res["verdict"] = decide(st)
    return res


def descriptive_pair(table, task: str, modality: str, seed: int, budget_id: str, boot: int, aseed: int) -> Optional[dict]:
    """Non-preregistered D1/S estimates of a single-seed U/M pair (image confirmation cells); None when incomplete."""
    try:
        ids, mats = _episode_matrix(table, task, modality, [seed], budget_id, (ANCHOR, ONE_STEP))
    except AggregationError as e:
        return {"complete": False, "reason": str(e)}
    st = stats_from_mats(mats, bootstrap_indices(len(ids), boot, aseed), formal=False)
    return {"complete": True, "n_episodes": len(ids), "stats": st, "note": "descriptive; not part of the formal family"}


def aggregate(cells_dir: pathlib.Path, root: pathlib.Path, boot: int, seed: int, image_n_test: int = IMAGE_TEST_N) -> dict:
    """Full aggregation: validation report, formal core verdicts, descriptive explore/image/official summaries and the
    completeness ledger against the frozen matrix."""
    manifest, cells = load_cells(cells_dir)
    table, report = validate(load_results(root), cells, image_n_test)
    seeds = [int(s) for s in manifest["train_seeds"]]
    def budget_id(task, modality):
        budget = manifest.get("budgets_by_task", {}).get(modality, {}).get(task, manifest["budgets"][modality])
        return f"B{int(budget) // 1000}k"
    core_tasks = sorted({cells[c]["identity"]["task_name"] for c in manifest["by_arm"]["core"]})
    tasks = [evaluate_task(table, t, "lowdim", seeds, budget_id(t, "lowdim"), boot, seed) for t in core_tasks]
    for sk in manifest.get("skipped", []):
        if sk.get("arm") == "core":
            tasks.append({"task": sk["task"], "modality": "lowdim", "status": "skipped", "reason": sk["reason"], "verdict": None})
    descriptive = {"explore": {}, "image": {}, "image_pairs": {}, "official": {}}
    for cid in manifest["by_arm"]["explore"]:
        descriptive["explore"][cid] = ladder(table.get(cid, {})) or {"note": "no valid results"}
    for cid in manifest["by_arm"]["image"]:
        descriptive["image"][cid] = ladder(table.get(cid, {})) or {"note": "no valid results"}
    for t in sorted({cells[c]["identity"]["task_name"] for c in manifest["by_arm"]["image"]}):
        descriptive["image_pairs"][t] = descriptive_pair(table, t, "image", seeds[0], budget_id(t, "image"), boot, seed)
    for cid, splits in report["official"].items():
        descriptive["official"][cid] = ladder(splits)
    completeness = {}
    for arm, ids in manifest["by_arm"].items():
        exp_n = len(ids) * len(LADDER)
        got = sum(1 for cid in ids for sk in LADDER if sk in table.get(cid, {}).get("test", {}))
        completeness[arm] = {"expected_cells": len(ids), "expected_test_records": exp_n, "valid_test_records": got,
                             "pending": [f"{cid}/test/{sk}" for cid in ids for sk in LADDER if sk not in table.get(cid, {}).get("test", {})]}
    return {"protocol_id": PROTOCOL_ID, "family_size": FAMILY_SIZE, "alpha": ALPHA, "delta": DELTA, "boot": boot,
            "analysis_seed": seed, "cells_manifest": str(cells_dir / "cells_manifest.json"),
            "expected_cells": len(manifest["cells"]), "skipped": manifest.get("skipped", []),
            "records": {k: report[k] for k in ("valid", "invalid", "unexpected")},
            "tasks": tasks, "descriptive": descriptive, "completeness": completeness}


def main() -> None:
    """CLI: aggregate ``--cells`` + ``--root`` into ``--out`` and print one line per core task plus the record counts."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cells", required=True, help="directory holding cells_manifest.json and the cell yamls")
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--boot", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=20260918)
    ap.add_argument("--image-n-test", type=int, default=IMAGE_TEST_N)
    a = ap.parse_args()
    out = aggregate(pathlib.Path(a.cells), pathlib.Path(a.root), a.boot, a.seed, a.image_n_test)
    pathlib.Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    pathlib.Path(a.out).write_text(json.dumps(out, indent=1))
    for t in out["tasks"]:
        print(t["task"], t["modality"], t["status"], t.get("verdict"), t.get("reason", ""))
    print(f"records valid={out['records']['valid']} invalid={len(out['records']['invalid'])} "
          f"unexpected={len(out['records']['unexpected'])}")


if __name__ == "__main__":
    main()
