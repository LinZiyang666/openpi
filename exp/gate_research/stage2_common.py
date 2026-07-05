"""Shared offline-analysis foundation for gate roadmap Stage 2 (2a/2b/2c).

Reuses the verified pure functions in :mod:`exp.gate_research.analyze_n1_live`
for run loading, episode dedup, authoritative/reconstructed skip flags, and
paired McNemar, and adds the primitives Stage 2 needs that ``analyze_n1_live``
does not provide:

- :func:`auc` -- re-implemented Mann-Whitney rank AUC. ``gate_structure_analysis``
  has an equivalent, but it is a side-effectful monolithic script (importing it
  runs a full analysis reading ``sys.argv``), so we deliberately do NOT import it
  and keep a self-contained copy here (WA 3.1 minimal-change: we neither import
  nor edit that script).
- :func:`mcnemar_exact_p` -- two-sided exact binomial McNemar p over the
  discordant pair counts. The reused ``analyze_n1_live.mcnemar`` returns only a
  continuity-corrected chi2; per-task slices have tiny ``b+c`` where the exact
  binomial test is the correct tool.
- :func:`load_run_episodes` / :func:`load_stage0_episodes` -- per-episode records
  with aligned ``(searched, hit_type, start_t, cp1_score)`` sequences, success,
  and the canonical unit key.
- :func:`action_source_seq` / :func:`cache_run_lengths` -- the continuous
  cache-execution run-length primitive (H1), with WARM_START as its own class.

Pure offline post-processing: reads only recorded data files; no ``src`` or
inference-path dependency.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

from exp.gate_research.analyze_n1_live import (
    check_complete_decisions,
    dedup_episodes,
    episode_searched,
    journal_success,
    load_jsonl,
    parse_task_uid,
    reconstruct_searched,
)

# ------------------------------------------------------------------
# Action-source labels (per-step action provenance)
# ------------------------------------------------------------------
CACHE_FH = "CACHE_FH"  # searched FULL_HIT -> action replayed from cache
WARM_START = "WARM_START"  # searched WARM_START -> partial-denoise replay (own class)
NEW_INFER = "NEW_INFER"  # searched MISS or skip -> fresh full inference

# Re-export so callers/tests reach the whole primitive set from one module.
__all__ = [
    "CACHE_FH",
    "WARM_START",
    "NEW_INFER",
    "EpisodeRec",
    "auc",
    "mcnemar_exact_p",
    "reconstruct_searched",
    "load_run_episodes",
    "load_stage0_episodes",
    "action_source_seq",
    "cache_run_lengths",
]


@dataclass
class EpisodeRec:
    """One episode's decision sequence, aligned step-by-step.

    ``searched_seq`` is authoritative for client_controlled runs and
    ordinal-reconstructed for periodic runs; the verdict fields (``hit_type``,
    ``start_t``, ``cp1_score``) are the raw recorded values and are only
    meaningful on searched steps (a skipped step stamps placeholder MISS).
    """

    task_id: int
    subset_init_state_idx: int
    searched_seq: list
    hit_type_seq: list
    start_t_seq: list
    cp1_score_seq: list
    success: bool

    @property
    def unit(self) -> tuple[int, int]:
        return (self.task_id, self.subset_init_state_idx)


# ------------------------------------------------------------------
# Statistics
# ------------------------------------------------------------------
def auc(scores, labels) -> float:
    """Mann-Whitney rank AUC of ``scores`` predicting binary ``labels`` (truthy =
    positive). Ties receive averaged ranks. Returns ``nan`` if either class is
    empty (an undefined AUC, never silently 0.5)."""
    pairs = sorted(zip(scores, labels), key=lambda x: x[0])
    n = len(pairs)
    n_pos = sum(1 for _, lab in pairs if lab)
    n_neg = n - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    rank_sum = 0.0  # sum of averaged ranks over positives
    i = 0
    while i < n:
        j = i
        while j < n and pairs[j][0] == pairs[i][0]:
            j += 1
        avg_rank = (i + 1 + j) / 2.0  # 1-based ranks of positions i..j-1 are i+1..j
        for k in range(i, j):
            if pairs[k][1]:
                rank_sum += avg_rank
        i = j
    return (rank_sum - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)


def mcnemar_exact_p(b: int, c: int) -> float:
    """Two-sided exact binomial McNemar p over discordant counts ``b``, ``c``.

    Under H0 each of ``n = b + c`` discordant pairs is equally likely either way,
    so ``p = min(1, 2 * sum_{k=0}^{min(b,c)} C(n,k) * 0.5^n)``. Complements the
    reused continuity-corrected chi2, which is a poor approximation when ``b+c``
    is small (e.g. per-task slices).
    """
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) * (0.5 ** n)
    return min(1.0, 2.0 * tail)


# ------------------------------------------------------------------
# Per-episode loading
# ------------------------------------------------------------------
def _episode_rec(ep: list, searched_seq: list, success) -> EpisodeRec:
    tid, idx = parse_task_uid(ep[0]["task_uid"])
    return EpisodeRec(
        task_id=tid,
        subset_init_state_idx=idx,
        searched_seq=list(searched_seq),
        hit_type_seq=[r.get("hit_type") for r in ep],
        start_t_seq=[r.get("start_t") for r in ep],
        cp1_score_seq=[r.get("cp1_score") for r in ep],
        success=bool(success),
    )


def load_run_episodes(manifest_or_path) -> list[EpisodeRec]:
    """Per-episode records for one n1_live run.

    Accepts an already-parsed manifest dict OR a path to the manifest JSON. Uses
    the authoritative recorded ``searched`` (client_controlled) or the
    reconstructed periodic ordinal, and validates the decision-step grid with the
    manifest's trusted ``replan_steps`` before segmenting.
    """
    m = manifest_or_path if isinstance(manifest_or_path, dict) else json.loads(
        Path(manifest_or_path).read_text())
    episodes = dedup_episodes(load_jsonl(m["per_step_out_path"]), yaml_id=m["yaml_id"])
    check_complete_decisions(episodes, m["replan_steps"])
    succ = journal_success(load_jsonl(m["journal_path"]), yaml_id=m["yaml_id"])
    out = []
    for uid, ep in episodes.items():
        searched = episode_searched(ep, m["gate_type"], m)
        out.append(_episode_rec(ep, searched, succ[uid]))
    return out


def load_stage0_episodes(gate_rows_path, yaml_id, replan_steps: int = 5,
                         journal_path=None) -> list[EpisodeRec]:
    """Stage-0 baseline per-episode records (always_search -> every step searched).

    Success comes from the per-step ``success`` flag stamped on every row (same as
    ``verify_gate.py``), or from ``journal_path`` when supplied.
    """
    episodes = dedup_episodes(load_jsonl(gate_rows_path), yaml_id=yaml_id)
    check_complete_decisions(episodes, replan_steps)
    succ = journal_success(load_jsonl(journal_path), yaml_id=yaml_id) if journal_path else None
    out = []
    for uid, ep in episodes.items():
        s = succ[uid] if succ is not None else ep[0].get("success")
        out.append(_episode_rec(ep, [True] * len(ep), s))
    return out


# ------------------------------------------------------------------
# Cache-execution run-length primitive (H1)
# ------------------------------------------------------------------
def action_source_seq(hit_type_seq: list, searched_seq: list) -> list:
    """Per-step action provenance.

    A skipped step is a forced new inference (C10: skip != cheap); a searched step
    follows its verdict -- FULL_HIT replays from cache, WARM_START is a partial
    denoise replay (kept as its own class for H3), MISS is a fresh inference.
    """
    out = []
    for hit, searched in zip(hit_type_seq, searched_seq):
        if not searched:
            out.append(NEW_INFER)
        elif hit == "FULL_HIT":
            out.append(CACHE_FH)
        elif hit == "WARM_START":
            out.append(WARM_START)
        else:  # searched MISS (or placeholder None on an empty-search MISS)
            out.append(NEW_INFER)
    return out


def cache_run_lengths(source_seq: list, include_ws: bool) -> list:
    """Maximal consecutive cache-execution run lengths.

    ``include_ws`` controls whether WARM_START counts as cache execution (R8
    sensitivity): ``True`` folds WS into the run, ``False`` treats WS as a break.
    """
    cache_set = {CACHE_FH, WARM_START} if include_ws else {CACHE_FH}
    runs, cur = [], 0
    for s in source_seq:
        if s in cache_set:
            cur += 1
        else:
            if cur:
                runs.append(cur)
            cur = 0
    if cur:
        runs.append(cur)
    return runs
