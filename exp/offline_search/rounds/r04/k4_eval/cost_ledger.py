"""R4 request/control cost ledger shared by collect and kpi.

Tables: {models: {MODEL: {full: {s1,s2,s3,k}, modes: {MODE: {s1,s2,s3,
miss_s1_extra}}, full_cost_ms: ...}}}. s3 is the full K-loop cost; costs share
one unit. Unknown cheap modes never silently receive a measured saving.
"""
from collections import Counter
import json
import math
from pathlib import Path

REPO = Path(__file__).resolve().parents[5]
DEFAULT_TABLE = REPO / "exp/offline_search/closed_loop/ops/cost_table.json"


def r4_enabled(meta, rows, starts):
    return bool(meta.get("cost_ledger") or meta.get("yaml_patch") or meta.get("client_overrides") or "replan_steps" in meta
                or meta.get("pure_inference") or any("vision" in r or "miss_k" in r for r in rows)
                or any("stage1_mode" in r or "miss_steps" in r for r in starts))


def request_length(meta):
    value = meta.get("client_overrides", {}).get("replan_steps", meta.get("replan_steps", 5))
    if type(value) is not int or value <= 0:
        raise ValueError("replan_steps must be a positive integer")
    return value


def _positive(value, label, allow_zero=False):
    value = float(value)
    if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
        raise ValueError(f"invalid {label}: {value}")
    return value


def stage_costs(model, table=None, suite=None):
    path = Path(table) if table else DEFAULT_TABLE
    if model == "pi05":
        full = {"s1": .152, "s2": .410, "s3": .438, "k": 10}
        source, full_ms = "owner rounded shares .152/.410/.438", 67.52
    elif model == "groot":
        old = REPO / "exp/libero_groot/config/rit/cost_groot_libero_measured.json"
        d = json.loads(old.read_text())
        full = {"s1": d["stage1_ms"], "s2": d["stage2_ms"], "s3": d["stage3_full_loop_ms"], "k": d["num_steps"]}
        source, full_ms = f"historical owner GR00T constants: {old}", sum(full[x] for x in ("s1", "s2", "s3"))
    else:
        raise ValueError(f"unknown model {model!r}")
    modes = {"full": dict(full)}
    if path.exists():
        data = json.loads(path.read_text())
        model_data = data.get("models", data).get(model)
        if model_data:
            suite = {"libero_spatial": "spatial", "libero_10": "l10", "sp": "spatial"}.get(suite, suite)
            if suite in model_data.get("by_suite", {}):
                model_data = model_data["by_suite"][suite]
            if "full" not in model_data:
                raise ValueError(f"{path}: {model} cost entry needs full={{s1,s2,s3,k}}")
            full = dict(model_data["full"])
            modes = {"full": dict(full), **model_data.get("modes", {})}
            full_ms = model_data.get("full_cost_ms", full_ms)
            source = str(path) + (f" ({suite})" if suite else "")
    for key in ("s1", "s2", "s3"):
        full[key] = _positive(full[key], key, True)
    full["k"] = _positive(full["k"], "full k")
    return full, modes, source, full_ms


def ledger(meta, rows, starts=(), comp=None, table=None):
    model = meta.get("model") or (starts[0].get("model") if starts else "pi05")
    legacy = not r4_enabled(meta, rows, starts)
    # Historical outcomes retain their original owner pricing even after a new
    # hardware/backend table is installed. --cost-table explicitly opts into repricing.
    chosen_table = table if table is not None else (REPO / "__historical_owner_constants__" if legacy else None)
    full, modes, source, full_ms = stage_costs(model, chosen_table, meta.get("suite_short") or meta.get("suite"))
    denominator = sum(full[x] for x in ("s1", "s2", "s3"))
    if denominator <= 0:
        raise ValueError("full stage cost must be positive")
    L = request_length(meta)
    default_mode = meta.get("stage1_mode", "full")
    default_k = ((meta.get("yaml_patch") or {}).get("miss") or {}).get("num_steps", full["k"])
    by_tag = {s.get("tag"): s for s in starts}
    modes_used, k_counts, l_counts = Counter(), Counter(), Counter()
    n = len(rows)
    nv = nm = controls = 0
    cost = 0.0
    measured_s1, measured_s23 = [], []
    inputs, missing_modes = {}, set()
    for row in rows:
        st = by_tag.get(row.get("tag"), starts[0] if len(starts) == 1 else {})
        mode = row.get("stage1_mode") or st.get("stage1_mode") or default_mode
        prices = {**full, **modes.get(mode, {})}
        if mode not in modes:
            missing_modes.add(mode)
        inputs[mode] = prices
        hit = row.get("hit", row.get("src") != "policy")
        vision = row.get("vision", row.get("vision_used", row.get("src") != "cache_blind"))
        if not hit and not vision:
            raise ValueError("a policy MISS requires vision")
        length = row.get("replan_steps", L)
        length = _positive(length, "L")
        l_counts[str(length)] += 1
        controls += length
        if vision:
            nv += 1
            modes_used[mode] += 1
            cost += prices["s1"]
            if row.get("s1_ms") is not None:
                measured_s1.append(float(row["s1_ms"]))
        if not hit:
            nm += 1
            k = row.get("miss_k")
            if k is None:
                k = st.get("miss_steps") or default_k
            k = _positive(k, "MISS K")
            k_counts[str(k)] += 1
            cost += prices.get("miss_s1_extra", 0) + prices["s2"] + prices["s3"] * k / prices["k"]
            if row.get("s23_ms") is not None:
                measured_s23.append(float(row["s23_ms"]))
    episodes = len(comp) if comp is not None else len({r.get("uid") for r in rows})
    # Controls are nominal requests*L; terminal requests may execute fewer than L.
    actual = None
    if comp and all(r.get("n_steps") is not None for r in comp.values()):
        actual = sum(int(r["n_steps"]) for r in comp.values())
    ir = cost / (n * denominator) if n else None
    scaled = cost / (controls / 5 * denominator) if controls else None
    if legacy and table is None and model == "pi05" and n:
        ir = .152 + .848 * (nm / n)
        scaled = ir * 5 / L
    measured = None
    if controls and full_ms and len(measured_s1) == nv and len(measured_s23) == nm:
        measured = (sum(measured_s1) + sum(measured_s23)) / (controls / 5 * full_ms)
    return {"model": model, "decisions": n, "vision_decisions": nv, "misses": nm,
            "v": nv / n if n else None, "m": nm / n if n else None,
            "k_per_miss": {"counts": dict(k_counts), "mean": sum(float(k)*v for k,v in k_counts.items()) / nm if nm else None},
            "l_per_request": {"counts": dict(l_counts), "mean": controls / n if n else None, "default": L},
            "stage1_modes": dict(modes_used), "stage_costs": inputs, "full_stage_costs": full,
            "cost_source": source, "full_cost": denominator, "full_cost_ms": full_ms,
            "legacy_owner_pricing": legacy and table is None,
            "total_cost": cost, "ir_per_request": ir, "ir_per_five_controls": scaled,
            "controls": controls, "controls_per_episode": controls / episodes if episodes else None,
            "controls_note": "nominal requested controls (requests * L); terminal chunk may be partial",
            "actual_controls": actual, "actual_controls_per_episode": actual / episodes if actual is not None and episodes else None,
            "s1_ms_total": sum(measured_s1), "s23_ms_total": sum(measured_s23),
            "s1_ms_n": len(measured_s1), "s23_ms_n": len(measured_s23), "ir_measured_per_five_controls": measured,
            "warnings": [f"unmeasured stage1 mode {x}: charged full reference costs" for x in sorted(missing_modes)]}
