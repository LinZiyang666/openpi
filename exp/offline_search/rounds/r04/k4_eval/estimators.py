"""Finite-population stratified SR/paired estimates. No rollout/controller imports."""
import math


def design_estimate(manifest, values):
    """values maps (task, init) to Y or paired D. Never silently reweight partial data."""
    selected = manifest["selected"]
    strata = manifest["data"].get("strata")
    out = {"manifest": manifest["path"], "selection_sha256": manifest["sha256"],
           "expected_pairs": len(selected), "observed_pairs": len(set(values) & set(selected)),
           "complete": set(selected) <= set(values),
           "note": "design uncertainty conditional on fixed outcomes; excludes stochastic rollout variance",
           "calibration": manifest["data"].get("calibration")}
    if not strata:
        return {**out, "estimate": None, "design_variance": None,
                "reason": "manifest has no stratum population sizes/inclusion probabilities"}
    population = sum(int(h["N"]) for h in strata)
    if population <= 0:
        raise ValueError("manifest population must be positive")
    ids = [h["stratum"] for h in strata]
    if len(set(ids)) != len(ids) or any(r.get("stratum") not in ids for r in selected.values()):
        raise ValueError("duplicate or unknown manifest stratum")
    estimate = variance = 0.0
    valid_var = True
    result = []
    for h in strata:
        N, n = int(h["N"]), int(h["n"])
        rows = [(p, r) for p, r in selected.items() if r.get("stratum") == h["stratum"]]
        if N < 1 or not 1 <= n <= N or len(rows) != n:
            raise ValueError(f"bad population/sample count in stratum {h['stratum']}")
        W, pi = N / population, n / N
        if "population_weight" in h and not math.isclose(float(h["population_weight"]), W):
            raise ValueError("manifest population weights do not match population counts")
        for p, r in rows:
            if not math.isclose(float(r["inclusion_probability"]), pi):
                raise ValueError(f"inclusion_probability != n/N for {p}")
            if "task" in h and p[0] != h["task"]:
                raise ValueError(f"wrong task in stratum for {p}")
        y = [float(values[p]) for p, _ in rows if p in values]
        mean = sum(y) / n if len(y) == n else None
        sample_var = sum((v - mean) ** 2 for v in y) / (n - 1) if mean is not None and n > 1 else None
        component = 0.0 if n == N else (W * W * (1 - pi) * sample_var / n if sample_var is not None else None)
        if mean is not None:
            # Horvitz-Thompson / stratified mean: sum_i y_i / pi_i / population.
            estimate += sum(float(values[p]) / float(r["inclusion_probability"]) for p, r in rows) / population
        if component is None:
            valid_var = False
        else:
            variance += component
        result.append({**h, "observed": len(y), "inclusion_probability": pi,
                       "mean": mean, "sample_variance": sample_var, "variance_component": component})
    estimate = estimate if out["complete"] else None
    variance = variance if out["complete"] and valid_var else None
    se = math.sqrt(variance) if variance is not None else None
    out.update(population=population, strata=result, estimate=estimate, design_variance=variance,
               design_se=se, design_normal95=[estimate - 1.959963984540054 * se, estimate + 1.959963984540054 * se]
               if estimate is not None and se is not None else None)
    if not out["complete"]:
        out["reason"] = "selected pairs missing; no full-pool estimate until all manifest pairs complete"
    elif not valid_var:
        out["reason"] = "non-census singleton stratum: design variance is not estimable"
    return out
