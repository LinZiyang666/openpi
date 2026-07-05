"""Stage 2c -- danger-step offline join (libero_spatial only, 0 GPU).

Joins the ``deviate_score >= 5`` oracle danger label (from the trajectory
deviation experiment) onto the Stage-0 gate_rows and asks whether cheap gate
signals (prev cp1_score, prev-is-MISS, cp1_score, step phase) predict a danger
step. Positive evidence -> N4 injection could target danger steps.

Scope + caveats (all in the plan risk register):
- **libero_spatial ONLY** (R1): l10 has no deviate_score / GT and cannot be done
  offline.
- **cross-config proxy** (R2): deviate_score is defined for the ``spatial16_w8_d4``
  cache bundle, the gate uses ``cp1_spatial_pool_16`` fh/ws configs -> a
  cross-config transfer assumption; the verdict is suggestive, not definitive.
- **trajectory divergence** (R3): deviate is an open-loop GT replay, gate_rows is
  a live gated rollout -> step alignment is exact early, approximate after
  divergence; AUC is also reported on the early-phase slice.

The join key is ``(task_id, orig_init_state_idx, step_idx = 5 * cycle)``.

Usage:
    python -m exp.gate_research.stage2c_danger_join \
        --deviate <deviate_score_*.json> --gt-dir <gt/> \
        --gate-rows <libero_spatial/gate_rows.jsonl> [--gate-yaml <yaml_id>] \
        --out <report.md>
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from exp.gate_research.analyze_n1_live import dedup_episodes, load_jsonl
from exp.gate_research.stage2_common import auc

DANGER_THRESHOLD = 5.0
REPLAN_STEPS = 5


# ------------------------------------------------------------------
# Oracle danger labels (pure)
# ------------------------------------------------------------------
def gt_index_from_attrs(records) -> dict:
    """Map episode name ``task_{TID}/episode_{EP}`` -> (task_id, orig_init_state_idx)
    from GT attribute records ``(task_id, episode_idx, orig_init_state_idx)``."""
    return {f"task_{tid}/episode_{eid}": (int(tid), int(orig))
            for tid, eid, orig in records}


def danger_labels(deviate_scores: dict, gt_index: dict, threshold: float = DANGER_THRESHOLD) -> dict:
    """{(task_id, orig_init, step_idx): danger_bool} from the per-cycle deviate
    scores (step_idx = REPLAN_STEPS * cycle). Episodes absent from ``gt_index``
    (no GT attrs) are skipped."""
    out = {}
    for ep_name, payload in deviate_scores.items():
        if ep_name not in gt_index:
            continue
        tid, orig = gt_index[ep_name]
        for t, score in enumerate(payload["deviate_score"]):
            out[(tid, orig, REPLAN_STEPS * t)] = bool(score >= threshold)
    return out


# ------------------------------------------------------------------
# Cheap gate signals (pure)
# ------------------------------------------------------------------
def gate_signals(gate_rows_path, gate_yaml_id: str) -> dict:
    """{(task_id, orig_init, step_idx): {signal: value}} for one gate config.

    Signals are oriented so that HIGHER = more danger-like (mirrors
    ``gate_structure_analysis`` 'neg' signals): ``neg_prev_score``,
    ``prev_is_MISS``, ``neg_cp1_score``, ``step_idx``. ``prev_*`` are ``None`` on
    an episode's first decision step.
    """
    episodes = dedup_episodes(load_jsonl(gate_rows_path), yaml_id=gate_yaml_id)
    out = {}
    for ep in episodes.values():
        prev_score = None
        prev_miss = None
        for r in ep:
            key = (r["task_id"], r["orig_init_state_idx"], r["step_idx"])
            cp1 = r.get("cp1_score")
            out[key] = {
                "neg_prev_score": (-prev_score) if prev_score is not None else None,
                "prev_is_MISS": (1.0 if prev_miss else 0.0) if prev_miss is not None else None,
                "neg_cp1_score": (-cp1) if cp1 is not None else None,
                "step_idx": float(r["step_idx"]),
            }
            prev_score = cp1
            prev_miss = r.get("hit_type") == "MISS"
    return out


# ------------------------------------------------------------------
# Join + AUC (pure)
# ------------------------------------------------------------------
SIGNAL_NAMES = ("neg_prev_score", "prev_is_MISS", "neg_cp1_score", "step_idx")


def join_auc(labels: dict, signals: dict, early_frac: float | None = None) -> dict:
    """Inner-join on ``(task_id, orig_init, step_idx)`` and compute per-signal AUC
    predicting danger. ``early_frac`` optionally restricts to the earliest
    fraction of each episode's steps (R3 alignment slice)."""
    keys = sorted(set(labels) & set(signals))
    if early_frac is not None:
        max_step = {}
        for tid, orig, step in keys:
            max_step[(tid, orig)] = max(max_step.get((tid, orig), 0), step)
        keys = [k for k in keys if k[2] <= early_frac * max_step[(k[0], k[1])]]
    n = len(keys)
    n_danger = sum(1 for k in keys if labels[k])
    aucs = {}
    for name in SIGNAL_NAMES:
        sc, lab = [], []
        for k in keys:
            v = signals[k].get(name)
            if v is not None:
                sc.append(v)
                lab.append(1 if labels[k] else 0)
        aucs[name] = auc(sc, lab) if sc else float("nan")
    return {"n_joined": n, "n_danger": n_danger,
            "danger_rate": (n_danger / n) if n else 0.0, "auc": aucs}


# ------------------------------------------------------------------
# GT attr reading (thin h5 wrapper) + report
# ------------------------------------------------------------------
def read_gt_index(gt_dir) -> dict:
    """Read ``(task_id, episode_idx, orig_init_state_idx)`` from every GT
    ``task_*/episode_*.h5`` under ``gt_dir`` (h5 attrs)."""
    import re

    import h5py

    records = []
    for h5path in sorted(Path(gt_dir).glob("task_*/episode_*.h5")):
        m = re.match(r"episode_(\d+)", h5path.stem)
        eid = int(m.group(1)) if m else None
        with h5py.File(h5path, "r") as f:
            records.append((int(f.attrs["task_id"]), eid, int(f.attrs["orig_init_state_idx"])))
    return gt_index_from_attrs(records)


def render_md(keybuilder: str, gate_yaml: str, full: dict, early: dict) -> str:
    def _auc_rows(res):
        return "\n".join(f"| {name} | {res['auc'][name]:.3f} |" for name in SIGNAL_NAMES)

    return "\n".join([
        "# Stage 2c — 危险步离线 join（libero_spatial）", "",
        f"oracle 危险 = deviate_score ≥ {DANGER_THRESHOLD}（keybuilder `{keybuilder}`）；"
        f"gate 信号取自 `{gate_yaml}`。**跨配置 proxy（R2）+ 轨迹发散（R3）**，结论定性 suggestive。", "",
        f"join {full['n_joined']} 步，危险 {full['n_danger']}（{100*full['danger_rate']:.1f}%）。", "",
        "**全程 AUC → danger（信号已定向为 higher=more danger）**", "",
        "| signal | AUC |", "|---|---|", _auc_rows(full), "",
        f"**早期相位切片（前 {int(100* (early.get('early_frac', 0.5)))}% 步，R3 对齐更可信）**："
        f" join {early['n_joined']} 步，危险率 {100*early['danger_rate']:.1f}%", "",
        "| signal | AUC |", "|---|---|", _auc_rows(early), "",
        "及格线：若某廉价信号 AUC 显著偏离 0.5 → N4 注入可做危险步靶向的证据；否则记录“危险步不可廉价预测”。",
    ])


def main():
    ap = argparse.ArgumentParser(description="Stage 2c danger-step offline join (libero_spatial)")
    ap.add_argument("--deviate", required=True, help="deviate_score_<keybuilder>.json")
    ap.add_argument("--gt-dir", required=True, help="trajectory_deviation GT dir (task_*/episode_*.h5)")
    ap.add_argument("--gate-rows", required=True, help="libero_spatial gate_rows.jsonl")
    ap.add_argument("--gate-yaml", default="", help="gate yaml_id for the signals (default: first)")
    ap.add_argument("--early-frac", type=float, default=0.5)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    deviate = json.loads(Path(a.deviate).read_text())
    gt_index = read_gt_index(a.gt_dir)
    labels = danger_labels(deviate, gt_index)

    gate_yaml = a.gate_yaml
    if not gate_yaml:
        for r in load_jsonl(a.gate_rows):
            if r.get("_kind") != "episode_summary" and r.get("yaml_id"):
                gate_yaml = r["yaml_id"]
                break
    signals = gate_signals(a.gate_rows, gate_yaml)

    full = join_auc(labels, signals)
    early = join_auc(labels, signals, early_frac=a.early_frac)
    early["early_frac"] = a.early_frac

    keybuilder = Path(a.deviate).stem.replace("deviate_score_", "")
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(render_md(keybuilder, gate_yaml, full, early))
    print(f"[stage2c] wrote {out} (join {full['n_joined']}, danger {full['n_danger']})")


if __name__ == "__main__":
    main()
