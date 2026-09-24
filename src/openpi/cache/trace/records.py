"""Model-agnostic record helpers shared by the Pi0.5 and GR00T trace paths.

``search_trace_from_check`` flattens a ``CheckTrace`` (what the orchestrator
recorded about one checkpoint) into the ``SearchTrace`` the writer stores;
``judge_result_json`` / ``verdict_dict`` give one JSON spelling of a
``JudgeResult`` so both interceptors and the sidecar agree;
``action_error_proxies`` is the open-loop distance of every variant to the
full inference. Jax-free; depends on numpy and ``openpi.cache.trace.types``.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import numpy as np

from openpi.cache.trace.types import (
    ARM_FULL_HIT,
    ARM_WARM_EXEC,
    CheckTrace,
    SearchTrace,
    warm_arm_name,
)


def _enum_name(value: Any) -> Any:
    return getattr(value, "name", value)


def verdict_dict(result: Any) -> dict[str, Any]:
    """Plain-dict spelling of a ``JudgeResult`` (or ``None`` -> empty)."""
    if result is None:
        return {}
    return {
        "hit_type": _enum_name(getattr(result, "hit_type", None)),
        "winner_id": getattr(result, "winner_id", None),
        "start_t": getattr(result, "start_t", None),
        "composer_score": getattr(result, "composer_score", None),
        "hit_override": getattr(result, "hit_override", None),
        "factor_outputs": getattr(result, "factor_outputs", None),
        "router_outputs": getattr(result, "router_outputs", None),
    }


def judge_result_json(result: Any) -> Optional[str]:
    if result is None:
        return None
    return json.dumps(verdict_dict(result), ensure_ascii=False, sort_keys=True, default=str)


def _signals_json(signals: Any) -> Optional[str]:
    if signals is None:
        return None
    if hasattr(signals, "__dataclass_fields__"):
        payload = {k: getattr(signals, k) for k in signals.__dataclass_fields__}
    elif isinstance(signals, dict):
        payload = signals
    else:
        payload = {"repr": repr(signals)}
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def search_trace_from_check(ct: Optional[CheckTrace]) -> Optional[SearchTrace]:
    """Flatten a ``CheckTrace`` into the storable ``SearchTrace``."""
    if ct is None:
        return None
    feats = ct.twin_step_features
    winner_per_field = dict(getattr(feats, "winner_per_field", {}) or {})
    field_own_margin = dict(getattr(feats, "field_own_margin", {}) or {})
    fused_margin = getattr(feats, "fused_margin", None)
    n_results = getattr(feats, "n_results", None)
    return SearchTrace(
        checkpoint=ct.checkpoint,
        gate_real_should_search=ct.gate_real_should_search,
        gate_twin_should_search=bool(ct.gate_twin_should_search),
        real_topk_ids=None if ct.real_results is None else [r.id for r in ct.real_results],
        real_topk_scores=(
            None if ct.real_results is None else [float(r.score) for r in ct.real_results]
        ),
        real_verdict_json=judge_result_json(ct.real_judge_result),
        twin_topk_ids=[r.id for r in ct.twin_results],
        twin_topk_scores=[float(r.score) for r in ct.twin_results],
        twin_per_field=ct.twin_per_field,
        twin_chain_scores=ct.twin_chain_scores,
        twin_winner_per_field={k: float(v) for k, v in winner_per_field.items()},
        twin_field_own_margin={k: float(v) for k, v in field_own_margin.items()},
        twin_fused_margin=None if fused_margin is None else float(fused_margin),
        twin_n_results=None if n_results is None else int(n_results),
        twin_retrieval_signals_json=_signals_json(ct.twin_retrieval_signals),
        twin_proposed_verdict_json=judge_result_json(ct.twin_proposed_verdict) or "{}",
        twin_verdict_json=judge_result_json(ct.twin_verdict) or "{}",
        twin_validation_error=ct.twin_validation_error,
        twin_replay_target=ct.twin_replay_target,
        top1_entry_id=ct.top1_entry_id,
    )


def check_trace_json(ct: Optional[CheckTrace]) -> Optional[str]:
    """Compact JSON of a ``CheckTrace`` for the CP3 twin record."""
    if ct is None:
        return None
    st = search_trace_from_check(ct)
    payload = {
        "checkpoint": st.checkpoint,
        "gate_real_should_search": st.gate_real_should_search,
        "gate_twin_should_search": st.gate_twin_should_search,
        "twin_topk": [[i, s] for i, s in zip(st.twin_topk_ids, st.twin_topk_scores)],
        "twin_verdict": json.loads(st.twin_verdict_json),
        "twin_proposed_verdict": json.loads(st.twin_proposed_verdict_json),
        "twin_validation_error": st.twin_validation_error,
        "top1_entry_id": st.top1_entry_id,
    }
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)


def action_error_proxies(
    full: np.ndarray,
    full_hit: Optional[np.ndarray],
    warm: dict[int, Optional[np.ndarray]],
    warm_exec: Optional[np.ndarray],
) -> dict[str, float]:
    """Open-loop distances of every variant to the full inference (plan §7.2)."""
    out: dict[str, float] = {}

    def _add(name: str, chunk: Optional[np.ndarray]) -> None:
        if chunk is None:
            return
        diff = np.asarray(chunk, dtype=np.float64) - np.asarray(full, dtype=np.float64)
        out[f"l2_{name}_vs_full"] = float(np.sqrt(np.sum(diff * diff)))
        out[f"max_{name}_vs_full"] = float(np.max(np.abs(diff))) if diff.size else 0.0

    _add(ARM_FULL_HIT, full_hit)
    for idx, chunk in warm.items():
        _add(warm_arm_name(idx), chunk)
    _add(ARM_WARM_EXEC, warm_exec)
    return out
