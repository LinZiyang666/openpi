"""Small CPU-only helpers; all campaign outcome/cost parsing uses the cache loader."""
from collections import Counter, defaultdict
import json
from pathlib import Path

from exp.offline_search.rounds.r17.cache import analyze_cache as cache

LEVELS = ("pure", "k50", "k40", "k30", "k20", "k10", "koff")
C0 = {"DP": 5, "Fast-WAM": 5, "Cosmos": 16, "π0.5": 5, "GR00T": 5}
VLA_MODELS = ("π0.5", "GR00T")
MODEL_IDS = {v: k for k, v in cache.MODEL_IDS.items()}
LIBRARY_ROOT = Path("/data/r17_library")
CONTROL_ROOT = Path("/data/mw_finetune/eval")
CONTROL_NAMES = {
    "DP": ("hs5dp_ctl_142812",),
    "Fast-WAM": ("hs6fw_res_152931",),
    # load_lane follows the resume ancestor and preserves its cost evidence.
    "Cosmos": ("formal_cosmos_libero_official2000_20261010_100533_5f558380",
               "formal_cosmos_libero_official2000_20261010_130639_a8ca2043"),
    "π0.5": ("pl_a",), "GR00T": ("gl_a",),
}
REDUCED_CONTROL_NAMES = {
    "DP": ("hs5dp_ddim10s0",),
    "Fast-WAM": ("hs6fw_k1",),
    "Cosmos": ("hs2_cosmos_h100_k2incl_c16_t107_20261010_115120",),
    "π0.5": ("pl_k1a",), "GR00T": ("gl_k1a",),
}
LABELS = dict(zip(LEVELS, ("纯推理", "0.5 档", "0.4 档", "0.3 档", "0.2 档", "0.1 档", "关（只有停滞门）")))
SUITE_LABELS = {"libero_spatial": "空间任务", "libero_10": "长任务"}
MODEL_LABELS = {model:model for model in C0}


def model_label(value):
    return cache.MODEL_IDS.get(value.removesuffix('_libero'),value)


def pure_references(lanes):
    """Read standalone formal pure baselines through the existing lane loader."""
    references = {}
    sources = cache.Sources()
    for lane in map(Path,lanes):
        loaded = cache.load_lane(lane,cache.receipt_snapshot(lane),sources,cache.SIZES)
        for (model,_,group,suite),result in loaded.items():
            if group != 'control':continue
            data = result['data']
            for row in data.values():row['pure']=True
            merge(references.setdefault((model,suite),{}),data)
    return references


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def nested_set(table, cell, task, value):
    model, suite, size = cell
    table.setdefault(model, {}).setdefault(suite, {}).setdefault(str(size), {})[task] = value


def merge(target, incoming):
    for key, episode in incoming.items():
        if key not in target or episode["ts"] > target[key]["ts"]:
            target[key] = episode


def load_knob(lanes, pool):
    """Return observed levels and planned task levels, including empty cells."""
    cells, planned = {}, {}
    sources = cache.Sources()
    limit = 30 if pool == "B" else 50
    for lane in lanes:
        lane = Path(lane).resolve()
        loaded = cache.load_lane(lane, cache.receipt_snapshot(lane), sources, cache.SIZES)
        model = cache.MODEL_IDS[sources.obj(lane / "campaign.json")["model"]]
        base = "crg" if model == "Cosmos" else "chrg"
        for segment, _ in cache.resume_segments(lane, sources):
            path = segment / "FULL_EPISODE_PLAN.json"
            if not path.exists():
                path = segment / "EPISODE_PLAN.json"
            for task in sources.obj(path).get("tasks", []):
                if task["experiment"] not in cache.SUITES:
                    continue
                extra = task["extra"]
                group = extra.get("cache_group", "control")
                if group == "control":
                    continue
                prefix, _, level = group.rpartition("_")
                if prefix != base or level not in LEVELS:
                    raise ValueError(f"{lane}: unknown knob group {group}")
                key = cache.canonical(task)
                if extra.get("init_pool", "A") != pool or not 0 <= key[2] < limit:
                    raise ValueError(f"{lane}: expected pool {pool}, init 0..{limit-1}: {key}")
                size = extra["library_size"]
                if size not in cache.SIZES:
                    raise ValueError(f"{lane}: unsupported library size {size}")
                cell = model, key[0], size
                cells.setdefault(cell, {}).setdefault(level, {})
                planned.setdefault(cell, {}).setdefault(key[1], set()).add(level)
        for (model, size, group, suite), result in loaded.items():
            if group == "control":
                continue
            level = group.rpartition("_")[2]
            if level == "pure":
                for episode in result["data"].values():
                    episode["pure"] = True
            merge(cells[model, suite, size][level], result["data"])
    if not cells:
        raise ValueError("No knob tasks in the supplied lanes")
    return cells, planned


def teacher_reference(root, model, suite):
    path = Path(root) / f"{MODEL_IDS[model]}_libero_{suite}" / "bpool/READY.json"
    ready = json.loads(path.read_bytes())
    episodes = ready["metadata"]["episodes"]
    ref = {(suite, e["task_name"], e["orig_init_idx"]): dict(success=e["success"], pure=True)
           for e in episodes}
    names = sorted({k[1] for k in ref})
    expected = {(suite, task, i) for task in names for i in range(50)}
    if len(episodes) != 500 or len(ref) != 500 or len(names) != 10 or set(ref) != expected:
        raise ValueError(f"{path}: need exactly one teacher episode per 10 tasks × 50 inits")
    if any(type(e["success"]) is not bool for e in episodes):
        raise ValueError(f"{path}: teacher success must be boolean")
    # READY capture rows supply executed controls when a selected task is pure.
    # Do not substitute expected_controls (a horizon) for actual controls.
    counts, sequences = defaultdict(Counter), defaultdict(set)
    for row in ready.get("rows", []):
        key = suite, row["task_name"], row["orig_init_idx"]
        seq = row["decision_seq"]
        if key not in ref or seq in sequences[key] or row.get("executed_count") is None:
            continue
        sequences[key].add(seq)
        counts[key]["controls"] += row["executed_count"]
        counts[key]["policy"] += row.get("source", "policy") == "policy"
    for e in episodes:
        key = suite, e["task_name"], e["orig_init_idx"]
        if sequences[key] == set(range(e.get("n_decisions", 0))) and sequences[key]:
            ref[key]["cost"] = dict(counts[key])
    return ref


def stats(data, model):
    rows = list(data.values())
    n = len(rows)
    total = Counter()
    for row in rows:
        total.update(row.get("cost", {}))
    cost_n = sum("cost" in row for row in rows)
    controls = total["controls"]
    # Aggregate actual calls/controls, including shortened terminal chunks.
    ir = total["policy"] * C0[model] / controls if n and cost_n == n and controls > 0 else None
    # Pure inference is the owner's unit reference by convention.
    if n and all(row.get("pure") for row in rows):
        ir = 1.0
    successes = sum(row["success"] for row in rows)
    return dict(n=n, successes=successes, sr=successes/n if n else None, ir_proxy=ir,
                cost_n=cost_n, controls=controls, policy=total["policy"], library=total["library"])


def percent(x):
    return "暂无" if x is None else f"{100*x:.1f}%"


def number(x):
    return "暂无" if x is None else f"{x:.3f}"
