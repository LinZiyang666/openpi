"""Horvitz-Thompson local excursions and unnormalized occupancy derivatives."""
from collections import Counter
import pandas as pd

from . import common as C


def estimate(entries, episodes, endpoint, mode, args, alpha=.05):
    """Original episodes include zeros for unreached entries, never duration weights."""
    frame = pd.DataFrame(entries)
    if mode == "first_entry" and len(frame):
        frame = frame.sort_values("decision_seq", kind="stable").drop_duplicates("episode_key", keep="first")
    reached = len(set(frame.episode_key)) if len(frame) else 0
    observed = []
    missing = 0
    reasons = Counter()
    for row in frame.to_dict("records"):
        value = C.number(row["outcomes"].get(endpoint))
        if value is None:
            missing += 1
            reasons[row["outcomes"].get("endpoint_reasons", {}).get(endpoint, "endpoint source unavailable")] += 1
        else:
            observed.append(dict(episode_key=row["episode_key"], numerator=row["score_weight"] * value))
    scores = pd.DataFrame(observed, columns=["episode_key", "numerator"]).groupby("episode_key").sum()
    population = episodes[["episode_key", "task_id", "init"]].merge(scores, how="left", on="episode_key")
    population["numerator"] = pd.to_numeric(population.numerator, errors="raise").fillna(0.)
    population["denominator"] = 1.
    result = dict(endpoint=endpoint, estimand=mode, original_episode_denominator=len(episodes), reached_episodes=reached,
                  unreached_share=1 - reached / len(episodes) if len(episodes) else None,
                  supported_assignment_denominator=len(frame), endpoint_unavailable=missing,
                  endpoint_unavailable_reasons=dict(reasons),
                  continuation="logged original future controller")
    if missing:
        result.update(status="unavailable", reason="supported assignments have missing endpoint; original population is not silently reduced")
        return result
    if not len(frame):
        result.update(status="unavailable", reason="no interior randomized support")
        return result
    result.update(status="available", population=C.cluster_interval(population, bootstraps=getattr(args, "bootstraps", 1000),
                   seed=getattr(args, "seed", 0), alpha=alpha))
    if mode == "first_entry":
        result["conditional_reached"] = C.cluster_interval(population[population.episode_key.isin(frame.episode_key)],
            bootstraps=getattr(args, "bootstraps", 1000), seed=getattr(args, "seed", 0), alpha=alpha)
    for branch in ("treated", "control"):
        weights = [row[branch + "_weight"] for row in frame.to_dict("records") if row[branch + "_weight"] > 0]
        result[branch + "_n"] = len(weights)
        result[branch + "_ESS"] = C.ess(weights)
        by_cluster = frame.groupby(["task_id", "init"])[branch + "_weight"].sum()
        result[branch + "_cluster_ESS"] = C.ess(by_cluster[by_cluster > 0])
    result["supported_task_init_clusters"] = len(frame[["task_id", "init"]].drop_duplicates())
    result["exploratory"] = result["supported_task_init_clusters"] < 30
    if not result["treated_n"] or not result["control_n"]:
        result.update(status="unavailable", reason="one randomized branch unobserved in this stratum")
        result.pop("population", None)
        result.pop("conditional_reached", None)
    return result
