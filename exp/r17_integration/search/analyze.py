"""Accepted episode quality and measured cost; missing ledgers stay unknown."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from exp.r17_integration.debug.reader_profiles import accepted_episodes, accepted_join, merge_decision_ledgers
from exp.r17_integration.debug.validate import digest


def _validate_ir_costs(ir_costs):
    """Owner-IR coefficient profile (P1 r17.ir_costs.v1); never inferred from wall time."""
    if ir_costs is None:
        return None
    if not isinstance(ir_costs, dict) or ir_costs.get("schema") != "r17.ir_costs.v1":
        raise ValueError("owner IR requires a bound r17.ir_costs.v1 coefficient profile")
    for name in ("vision", "miss"):
        value = ir_costs.get(name)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError("owner IR coefficients must be finite normalized stage costs")
    if not ir_costs.get("sampler_identity") or not ir_costs.get("source_identity"):
        raise ValueError("owner IR coefficients must bind the MISS sampler and a measurement source")
    return ir_costs


def summarize(expected, journal, decisions, *, task_groups, binding, full_baseline_ms=None,
              cost_rows=(), execution_rows=(), episode_manifests=(), ir_costs=None):
    """Quality plus two separate cost readings.

    * owner IR = (vision*V + miss(K)*M)/N from the actual stage ledger and a
      bound coefficient profile (the R16 price definition);
    * host timing: ``absolute_ms`` sums host-wall ``cost_ms`` (includes shared
      model-lock queueing); pricing sums each row's declared
      ``pricing_time_field`` (``service_ms`` = wall minus measured lock wait).
      Only rows with a measured queue/service split and no capture are priced.
    """
    journal = tuple(journal)
    ir_costs = _validate_ir_costs(ir_costs)
    if full_baseline_ms is not None and (full_baseline_ms["binding_digest"] != digest(binding)
                                        or not np.isfinite(full_baseline_ms["absolute_ms"])
                                        or full_baseline_ms["absolute_ms"] <= 0):
        raise ValueError("normalization baseline must match model and benchmark binding")
    expected = tuple(expected)
    if len(expected) != len(set(expected)):
        raise ValueError("duplicate expected task UID")
    accepted = list(accepted_episodes(journal).values())
    actual = {r["task_uid"] for r in accepted}
    if actual - set(expected):
        raise ValueError("unexpected accepted episodes")
    decisions = merge_decision_ledgers(decisions, cost_rows=cost_rows, execution_rows=execution_rows,
                                      episode_manifests=episode_manifests)
    joined = accepted_join(decisions, journal)
    tasks = defaultdict(list)
    for row in accepted:
        if row["task_name"] not in task_groups:
            raise ValueError("missing task difficulty assignment")
        tasks[row["task_name"]].append(row)
    per_task = {name: dict(completed=len(tasks[name]), successes=sum(r["success"] for r in tasks[name]),
                           sr=sum(r["success"] for r in tasks[name])/len(tasks[name]) if tasks[name] else None,
                           difficulty=group) for name, group in task_groups.items()}
    groups = defaultdict(list)
    for name, record in per_task.items():
        groups[task_groups[name]].append(record["sr"])
    difficulty = {group: dict(task_count=len(values), available_tasks=sum(v is not None for v in values),
                              mean_task_sr=float(np.mean([v for v in values if v is not None])) if any(v is not None for v in values) else None)
                  for group, values in groups.items()}
    required = ("stage_calls", "actual_nfe", "cost_ms", "executed_count", "cursor", "source")
    covered = all(sorted(r["decision_seq"] for r in joined if r["task_uid"] == ep["task_uid"]
                         and r["attempt"] == ep["attempt"]) == list(range(ep["n_decisions"])) for ep in accepted)
    complete_cost = bool(joined) and covered and all(all(k in r and r[k] is not None for k in required) for r in joined)
    cost = dict(status="available" if complete_cost else "unsupported",
                reason="" if complete_cost else "missing actual stage/NFE/cost/cursor ledger",
                absolute_ms=None, normalized_within_model=None, decisions=len(joined),
                pricing_absolute_ms=None, pricing_status="unsupported")
    if complete_cost:
        for row in joined:
            if row["cost_ms"] < 0 or row["actual_nfe"] < 0 or len(row["stage_calls"]) != 3 or min(row["stage_calls"]) < 0:
                raise ValueError("invalid cost ledger")
        total = sum(r["cost_ms"] for r in joined)
        cost.update(absolute_ms=total, stage_calls=np.sum([r["stage_calls"] for r in joined], axis=0).tolist(),
                    N=len(joined), V=sum(r["stage_calls"][0] for r in joined),
                    M=sum(r["stage_calls"][1] for r in joined), NFE=sum(r["actual_nfe"] for r in joined),
                    executed_controls=sum(r["executed_count"] for r in joined),
                    commit_rows=[dict(source=r["source"], cursor=r["cursor"], executed=r["executed_count"]) for r in joined],
                    extra_cost_ms={key: sum(r.get(key, 0.) for r in joined)
                                   for key in ("calibration_ms", "library_collection_ms", "shadow_ms")})
        captured = [r for r in joined if r.get("capture_enabled") is True]
        eligible = all(r.get("capture_enabled") is False and r.get("pricing_eligible") is True
                       and r.get("pricing_time_field") == "service_ms" and r.get("service_ms") is not None
                       for r in joined)
        pricing_total = sum(r["service_ms"] for r in joined) if eligible else None
        queue_total = (sum(r["queue_wait_ms"] for r in joined)
                       if all(r.get("queue_wait_ms") is not None for r in joined) else None)
        cost.update(pricing_status="excluded_capture" if captured else "available" if eligible else "unsupported",
                    pricing_absolute_ms=pricing_total, pricing_time_field="service_ms" if eligible else None,
                    queue_wait_total_ms=queue_total,
                    capture_decisions=len(captured), capture_overhead_ms=sum(r.get("capture_overhead_ms", 0.) for r in joined)
                    if not captured or all("capture_overhead_ms" in r for r in captured) else None,
                    capture_overhead_status="measured" if captured and all("capture_overhead_ms" in r for r in captured)
                    else "unknown" if captured else "not_applicable",
                    capture_overhead_scopes=sorted({r.get("capture_overhead_scope", "unspecified") for r in captured}),
                    capture_sink_ms=sum(r["capture_sink_ms"] for r in captured)
                    if captured and all("capture_sink_ms" in r for r in captured) else None,
                    server_infer_total_ms=sum(r["server_infer_ms"] for r in joined)
                    if all("server_infer_ms" in r for r in joined) else None,
                    timing_scopes=sorted({r.get("timing_scope", "unspecified") for r in joined}))
        if full_baseline_ms is not None:
            # Same pricing field on both sides: service time, never a wall ratio
            # that silently mixes in shared-lock queueing of a different load.
            baseline_price = full_baseline_ms.get("pricing_absolute_ms")
            baseline_eligible = (full_baseline_ms.get("capture_enabled") is False
                                 and full_baseline_ms.get("pricing_eligible") is True
                                 and full_baseline_ms.get("pricing_time_field") == "service_ms"
                                 and isinstance(baseline_price, (int, float)) and not isinstance(baseline_price, bool)
                                 and np.isfinite(baseline_price) and baseline_price > 0)
            if eligible and baseline_eligible:
                cost["normalized_within_model"] = pricing_total/baseline_price
            cost["normalization_status"] = "available" if eligible and baseline_eligible else "excluded_capture_or_unknown"
        cost.update(owner_ir=None, owner_ir_status="unsupported")
        if ir_costs is not None:
            identities = {r.get("sampler_identity") for r in joined}
            if identities != {ir_costs["sampler_identity"]}:
                raise ValueError("owner IR coefficients are bound to a different MISS sampler")
            from exp.r17_integration.common.ir import owner_ir
            cost.update(owner_ir=owner_ir(cost["N"], cost["V"], cost["M"], ir_costs), owner_ir_status="available",
                        ir_costs=dict(ir_costs), owner_ir_definition="(vision*V+miss(K)*M)/N")
    successes = sum(r["success"] for r in accepted)
    available_groups = [g["mean_task_sr"] for g in difficulty.values() if g["mean_task_sr"] is not None]
    return dict(schema="r17.analysis.v1", binding=binding, binding_digest=digest(binding),
                expected=len(expected), completed=len(accepted), successes=successes,
                sr=successes/len(accepted) if accepted else None,
                successes_over_expected=successes/len(expected) if expected else None,
                missing=sorted(set(expected)-actual), errors=sum(r.get("status") == "error" for r in journal),
                rejected=sum(r.get("accepted") is not True for r in journal), per_task=per_task,
                difficulty=difficulty, equal_group_mean=float(np.mean(available_groups)) if available_groups else None,
                cost=cost)


PAIR_FIELDS = ("campaign", "benchmark_protocol", "model_identity", "task_name", "orig_init_idx", "task_data_hash")


def paired_bootstrap(reference, candidate, *, seed=0, samples=1000):
    def index(rows):
        result = {}
        for row in accepted_episodes(rows).values():
            key = tuple(row[k] for k in PAIR_FIELDS)
            if key in result:
                raise ValueError("ambiguous paired episode identity")
            result[key] = bool(row["success"])
        return result
    a, b = index(reference), index(candidate)
    if set(a) != set(b) or not a:
        raise ValueError("paired cohorts must have identical semantic identities")
    if samples <= 0:
        raise ValueError("positive bootstrap sample count required")
    delta = np.array([int(b[k])-int(a[k]) for k in sorted(a)], np.float64)
    rng = np.random.default_rng(seed)
    draws = np.array([rng.choice(delta, len(delta), replace=True).mean() for _ in range(samples)])
    return dict(pairs=len(delta), delta_sr=float(delta.mean()), ci95=np.quantile(draws, [.025, .975]).tolist(),
                samples=samples, seed=seed, pair_identity_digest=digest(sorted(a)))


def offline_action_scores(predicted, teacher, *, layout, commit_horizon=None, sigma=None):
    """Layout-aware offline error only; never substitute it for closed-loop SR."""
    from exp.r17_integration.debug.validate import plain
    layout = plain(layout)
    active = layout.get("valid_action_indices") or list(range(layout["valid_action_dim"]))
    motion = layout["motion_channels"]
    grip = layout["gripper_channel"]
    h = commit_horizon or layout["replan_steps"]
    if not 1 <= h <= layout["horizon"]:
        raise ValueError("offline scoring horizon outside layout")
    predicted, teacher = np.asarray(predicted), np.asarray(teacher)
    if predicted.shape != teacher.shape or predicted.ndim != 3 or predicted.shape[1] != layout["horizon"]:
        raise ValueError("offline scoring action geometry mismatch")
    error = predicted[:, :h, active].astype(np.float64) - teacher[:, :h, active].astype(np.float64)
    if not np.isfinite(error).all():
        raise ValueError("nonfinite active scoring values")
    if sigma is not None:
        sigma = np.asarray(sigma, np.float64)
        if sigma.shape != (len(active),) or not np.isfinite(sigma).all() or np.any(sigma <= 0):
            raise ValueError("explicit active-channel sigma required")
        error = error / sigma
    motion_positions = [active.index(i) for i in motion]
    return dict(status="offline_action_error_only", decisions=len(error), horizon=h,
                mean_mse=float(np.mean(error**2)), motion_mse=float(np.mean(error[:, :, motion_positions]**2)),
                gripper_mse=float(np.mean(error[:, :, active.index(grip)]**2)) if grip is not None else None,
                per_decision_mse=np.mean(error**2, axis=(1, 2)).tolist())


def offline_sweep(variants, *, method_factory, library, episodes, teacher_actions, layout):
    from .replay import replay
    reports = []
    for variant in variants:
        method = method_factory(variant)
        results = replay(method, library, episodes)
        actions = np.stack([r["arrays"]["actions"] for r in results])
        reports.append(dict(variant=variant, scores=offline_action_scores(actions, teacher_actions, layout=layout),
                            ledger=[r["metadata"] for r in results]))
    return dict(status="offline_only", variants=reports, closed_loop_sr=None)
