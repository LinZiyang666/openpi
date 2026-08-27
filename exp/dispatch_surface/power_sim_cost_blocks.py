"""Pre-registered power simulation fixing the cost-bench block count R.

Simulates the paired block bootstrap for the four cost gates at their frozen
surrogate effects (plan 4.6 / G1 R4), using the SAME quantile levels as the
formal adjudicator (imported from ``analyze_precheck`` — Gate 1 tests the
bootstrap p95, Gate 2 the p97.5; G2R2-B4):

    gate1_compute: true effect -10% vs threshold -5%   (p95,  compute-axis sigma)
    gate1_latency: true effect  -2% vs threshold  0%   (p95,  latency-axis sigma)
    gate2_compute: true effect   0% vs threshold +5%   (p97.5, compute-axis sigma)
    gate2_latency: true effect  -2% vs threshold  0%   (p97.5, latency-axis sigma)

For each R in {5, 10, 15}: draw n_sim experiments of R paired block
differences (true effect + N(0, sigma_axis)) and record the pass rate of the
per-gate bootstrap upper bound. chosen_r = the smallest R with ALL four
powers >= 0.80; if R=15 still fails, stop-loss B fires before any test data
exists (exit 3).

The record is tamper-evident: it carries a schema version, the sha256 of the
variance-source file, every frozen constant, and a canonical content digest.
``validate_power_record`` (shared by run_cost_bench and the analyzer)
recomputes the digest and re-derives chosen_r from per_r_power — a hand-
written ``{"chosen_r": 1}`` fails both checks.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from exp.dispatch_surface.analysis.analyze_precheck import GATE1_UPPER_Q, GATE2_UPPER_Q

POWER_RECORD_SCHEMA = 1
VARIANCE_SOURCE_SCHEMA = 1
R_CANDIDATES = (5, 10, 15)
POWER_TARGET = 0.80
STOP_LOSS_EXIT = 3
POWER_SEED = 20260827
POWER_N_SIM = 2000
POWER_N_BOOT = 500

# (name, true_effect, gate_threshold, quantile_level, sigma_axis)
GATES = (
    ("gate1_compute", -0.10, -0.05, GATE1_UPPER_Q, "compute"),
    ("gate1_latency", -0.02, 0.0, GATE1_UPPER_Q, "latency"),
    ("gate2_compute", 0.00, 0.05, GATE2_UPPER_Q, "compute"),
    ("gate2_latency", -0.02, 0.0, GATE2_UPPER_Q, "latency"),
)


def gate_power(r: int, true_effect: float, threshold: float, quantile: float,
               sigma: float, rng: np.random.Generator, n_sim: int, n_boot: int) -> float:
    passes = 0
    for _ in range(n_sim):
        diffs = true_effect + rng.normal(0.0, sigma, size=r)
        idx = rng.integers(0, r, size=(n_boot, r))
        boot_means = diffs[idx].mean(axis=1)
        if np.quantile(boot_means, quantile) <= threshold:
            passes += 1
    return passes / n_sim


def simulate(sigma_compute: float, sigma_latency: float, seed: int,
             n_sim: int, n_boot: int) -> dict:
    rng = np.random.default_rng(seed)
    sigmas = {"compute": sigma_compute, "latency": sigma_latency}
    per_r = {}
    for r in R_CANDIDATES:
        per_r[str(r)] = {
            name: gate_power(r, eff, thr, q, sigmas[axis], rng, n_sim, n_boot)
            for name, eff, thr, q, axis in GATES
        }
    record = {
        "schema_version": POWER_RECORD_SCHEMA,
        "chosen_r": derive_chosen_r(per_r),
        "per_r_power": per_r,
        "sigma_compute": sigma_compute,
        "sigma_latency": sigma_latency,
        "power_target": POWER_TARGET,
        "r_candidates": list(R_CANDIDATES),
        "gates": [{"name": n, "true_effect": e, "threshold": t, "quantile": q,
                   "sigma_axis": a} for n, e, t, q, a in GATES],
        "seed": seed,
        "n_sim": n_sim,
        "n_boot": n_boot,
    }
    record["record_digest"] = record_digest(record)
    return record


def load_variance_source(path: pathlib.Path) -> tuple[float, float]:
    """Load the E0 paired-block SDs from the content-bound authority file.

    The power CLI deliberately has no independent ``--sigma-*`` flags: if
    sigma can be supplied separately from the hashed source, a record can cite
    a high-variance E0 file while replaying an arbitrarily tiny variance and
    selecting R=5.  The source itself is therefore the sole value authority.
    """
    try:
        source = json.loads(path.read_text())
    except Exception as exc:  # noqa: BLE001 — malformed authority is a refusal
        raise SystemExit(f"variance source is not valid JSON: {path}: {exc}") from exc
    if source.get("schema_version") != VARIANCE_SOURCE_SCHEMA:
        raise SystemExit(
            f"variance source schema {source.get('schema_version')} unsupported; "
            f"expected {VARIANCE_SOURCE_SCHEMA}"
        )
    values = []
    for field in ("sigma_compute", "sigma_latency"):
        value = source.get(field)
        if isinstance(value, bool) or not isinstance(value, (int, float)) \
                or not np.isfinite(value) or value <= 0:
            raise SystemExit(f"variance source {field} must be finite and > 0, got {value!r}")
        values.append(float(value))
    return values[0], values[1]


def derive_chosen_r(per_r_power: dict) -> int | None:
    for r in R_CANDIDATES:
        powers = per_r_power.get(str(r), {})
        if powers and all(p >= POWER_TARGET for p in powers.values()):
            return r
    return None


def record_digest(record: dict) -> str:
    body = {k: v for k, v in record.items() if k != "record_digest"}
    return hashlib.sha256(
        json.dumps(body, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_power_record(record: dict, *, replay: bool = True) -> int:
    """Shared validation for run_cost_bench and the analyzer (G2R2-B4/G2R3-B2).

    A self-digest cannot authenticate a self-describing record — anyone can
    edit the powers and re-hash. Authenticity therefore rests on DETERMINISTIC
    REPLAY: the recorded (sigma, seed, n_sim, n_boot) are re-simulated through
    the shared ``simulate`` and the per-R powers must match exactly. On top of
    that: schema/key-domain/value-domain checks, frozen-constant comparison,
    chosen_r re-derivation, and a re-hash of the variance-source file.
    Returns the validated R; raises SystemExit otherwise.
    """
    if record.get("schema_version") != POWER_RECORD_SCHEMA:
        raise SystemExit(f"power record schema {record.get('schema_version')} unsupported")
    if record.get("record_digest") != record_digest(record):
        raise SystemExit("power record digest mismatch — record was edited by hand")
    expected_gates = [
        {"name": n, "true_effect": e, "threshold": t, "quantile": q, "sigma_axis": a}
        for n, e, t, q, a in GATES
    ]
    if record.get("gates") != expected_gates:
        raise SystemExit("power record gate constants differ from the frozen definitions")
    if record.get("power_target") != POWER_TARGET or \
            record.get("r_candidates") != list(R_CANDIDATES):
        raise SystemExit("power record target/candidates differ from the frozen definitions")
    frozen_simulation = {
        "seed": POWER_SEED,
        "n_sim": POWER_N_SIM,
        "n_boot": POWER_N_BOOT,
    }
    for field, expected in frozen_simulation.items():
        if record.get(field) != expected:
            raise SystemExit(
                f"power record {field}={record.get(field)!r} differs from frozen {expected}"
            )

    per_r = record.get("per_r_power", {})
    gate_names = [g[0] for g in GATES]
    if set(per_r) != {str(r) for r in R_CANDIDATES}:
        raise SystemExit(f"power record per_r_power keys {sorted(per_r)} != R candidates")
    for r_key, powers in per_r.items():
        if set(powers) != set(gate_names):
            raise SystemExit(f"power record R={r_key} gate keys {sorted(powers)} != four gates")
        for name, p in powers.items():
            if isinstance(p, bool) or not (
                isinstance(p, (int, float)) and np.isfinite(p) and 0.0 <= p <= 1.0
            ):
                raise SystemExit(f"power record R={r_key} gate {name} power {p!r} out of domain")

    derived = derive_chosen_r(per_r)
    if record.get("chosen_r") != derived:
        raise SystemExit(
            f"power record chosen_r={record.get('chosen_r')} but per_r_power derives "
            f"{derived} — refusing"
        )
    if derived is None:
        raise SystemExit(
            "power record shows no sufficient R (underpowered) — stop-loss B, "
            "the cost bench must not run"
        )

    if replay:
        for field in ("sigma_compute", "sigma_latency"):
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not np.isfinite(value) or value <= 0:
                raise SystemExit(
                    f"power record simulation parameter {field!r} is invalid: {value!r}"
                )
        src = record.get("variance_source")
        if not src or not pathlib.Path(src).is_file():
            raise SystemExit(f"power record variance_source missing on disk: {src!r}")
        src_path = pathlib.Path(src)
        if _file_sha256(src_path) != record.get("variance_source_sha256"):
            raise SystemExit("variance source file content drifted since the power freeze")
        sigma_compute, sigma_latency = load_variance_source(src_path)
        if record.get("sigma_compute") != sigma_compute or \
                record.get("sigma_latency") != sigma_latency:
            raise SystemExit(
                "power record sigma values do not equal the content-bound variance source"
            )
        replayed = simulate(
            sigma_compute=sigma_compute,
            sigma_latency=sigma_latency,
            seed=POWER_SEED,
            n_sim=POWER_N_SIM,
            n_boot=POWER_N_BOOT,
        )
        if replayed["per_r_power"] != per_r:
            raise SystemExit(
                "power record per_r_power does not reproduce under deterministic "
                "replay of its own parameters — forged or corrupted record"
            )
    return derived


def _file_sha256(path: pathlib.Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while chunk := f.read(1 << 22):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variance-source", required=True,
                    help="JSON E0 variance authority containing schema_version, "
                         "sigma_compute and sigma_latency; values and sha256 are frozen")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    src = pathlib.Path(args.variance_source)
    if not src.is_file():
        raise SystemExit(f"variance source file missing: {src}")
    sigma_compute, sigma_latency = load_variance_source(src)
    record = simulate(
        sigma_compute, sigma_latency, POWER_SEED, POWER_N_SIM, POWER_N_BOOT,
    )
    record["variance_source"] = str(src)
    record["variance_source_sha256"] = _file_sha256(src)
    record["record_digest"] = record_digest(record)
    pathlib.Path(args.out).write_text(json.dumps(record, indent=2))
    print(json.dumps({"chosen_r": record["chosen_r"],
                      "per_r_power": record["per_r_power"]}, indent=2))
    if record["chosen_r"] is None:
        raise SystemExit(STOP_LOSS_EXIT)


if __name__ == "__main__":
    main()
