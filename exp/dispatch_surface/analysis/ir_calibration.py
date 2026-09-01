"""Predicted versus realized inference ratio per arm (RIT-PL plan section 3.6).

The prediction never touches a fit: each arm's deployed artifact (its two cut
constants, or the (s, v) boundary table for an SV arm) is replayed row by row
with the shared ``surface_verdict`` over the frozen development rows of the
shadow table, priced with the cost authority and expressed in percent of the
all-MISS cost. The realized value is the rollout's ratio-of-sums cost from a
``sgrid_sweep summarize`` output. Every source is bound to ONE Rev 1 package
(suite, manifest SHA, library SHA), the summary to its matrix, and every arm
to its artifact SHA, so predictions from another library or suite can never
be compared to these rollouts.

Usage:
  uv run python -m exp.dispatch_surface.analysis.ir_calibration \
      --rev1-package-manifest <pkg>/MANIFEST.json --table <table.jsonl> \
      --source sgrid:<arm_matrix_sgrid.json>:<sgrid_summary.json> \
      --source sysgate:<arm_matrix_sgrid.json>:<sysgate_summary.json> \
      --out-json <out.json> --out-fig <out.png>
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pathlib

import numpy as np

from exp.dispatch_surface import rev1_package as pkgmod
from exp.dispatch_surface.analysis.analytic_cost import PINNED_START_T_WS, cost_model_digest, unit_cost
from exp.dispatch_surface.export_exploratory_surface import dev_mask_from_membership
from exp.dispatch_surface.fit_surface import _digest_obj, load_table
from exp.dispatch_surface.sgrid_sweep import LAYER_SGRID, PROTOCOL_SGRID, PROTOCOL_SYSGATE
from openpi.cache.components.surface_judge import load_surface_artifact, surface_verdict

ACCEPTED_PROTOCOLS = (PROTOCOL_SGRID, PROTOCOL_SYSGATE)
_COST = {"full": unit_cost("FULL_HIT", None), "warm": unit_cost("WARM_START", PINNED_START_T_WS),
         "miss": unit_cost("MISS", None)}
OUTPUT_KEYS = frozenset({"table_sha256", "dev_membership_sha256", "n_dev_rows", "c_miss", "sources"})
SOURCE_KEYS = frozenset({"matrix_sha256", "summary_sha256", "protocol", "gate_type", "gate_theta", "arms"})
ARM_KEYS = frozenset({"family", "estimator", "predicted_ir", "measured_ir", "gap", "artifact_sha256"})


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def predicted_ir_for_artifact(artifact, table, dev_mask: np.ndarray) -> float:
    """Row-by-row ``surface_verdict`` replay of a deployed artifact on the
    development rows, priced with the cost authority, in percent of all-MISS."""
    idx = np.where(dev_mask)[0]
    total = 0.0
    for i in idx:
        verdict = surface_verdict(float(table.s[i]), float(table.v[i]), artifact.v_bin_edges,
                                  artifact.s_min_full, artifact.s_min_warm,
                                  uses_disagreement=artifact.uses_disagreement)
        total += _COST[verdict]
    return 100.0 * total / (len(idx) * _COST["miss"])


def bind_source(tag: str, matrix_path: str, summary_path: str, *, manifest_sha: str, suite: str,
                library_sha256: str) -> dict:
    """Refuse any source that is not an sgrid-layer sweep of THIS package."""
    if not tag:
        raise SystemExit("--source tag must be non-empty")
    matrix = json.loads(pathlib.Path(matrix_path).read_text())
    summary = json.loads(pathlib.Path(summary_path).read_text())
    what = f"source {tag}"
    if matrix.get("protocol") not in ACCEPTED_PROTOCOLS or matrix.get("layer") != LAYER_SGRID:
        raise SystemExit(f"{what}: matrix protocol/layer {matrix.get('protocol')!r}/{matrix.get('layer')!r} is not an sgrid sweep")
    if matrix.get("suite") != suite:
        raise SystemExit(f"{what}: matrix suite {matrix.get('suite')!r} != package suite {suite!r}")
    if matrix.get("rev1_package_manifest_sha256") != manifest_sha:
        raise SystemExit(f"{what}: matrix was emitted against a different Rev 1 package")
    if matrix.get("library_sha256") != library_sha256:
        raise SystemExit(f"{what}: matrix library_sha256 != the package's frozen library")
    digest = cost_model_digest()
    if matrix.get("cost_model_digest") != digest or summary.get("cost_model_digest") != digest:
        raise SystemExit(f"{what}: matrix / summary cost model digest != the cost authority")
    matrix_sha = _sha(matrix_path)
    if summary.get("suite") != matrix.get("suite") or summary.get("protocol") != matrix.get("protocol"):
        raise SystemExit(f"{what}: summary suite/protocol differ from the matrix")
    if (summary.get("input_sha256") or {}).get("arm_matrix") != matrix_sha:
        raise SystemExit(f"{what}: summary does not bind this arm matrix")
    if summary.get("rev1_package_manifest_sha256") != manifest_sha:
        raise SystemExit(f"{what}: summary was produced against a different Rev 1 package")
    arms = summary.get("arms") or {}
    if not arms or not set(arms) <= set(matrix.get("arms") or {}):
        raise SystemExit(f"{what}: summary arms are not a non-empty subset of the matrix arms")
    for arm, a in arms.items():
        if arm not in (matrix.get("artifact_paths") or {}):
            raise SystemExit(f"{what}: arm {arm} has no surface artifact in the matrix")
        if a.get("family") != (matrix.get("families") or {}).get(arm):
            raise SystemExit(f"{what}: arm {arm} family differs between summary and matrix")
    return {"tag": tag, "matrix": matrix, "summary": summary, "matrix_sha256": matrix_sha,
            "summary_sha256": _sha(summary_path)}


def calibrate(args) -> dict:
    """Predicted-vs-realized IR table for every ``--source`` bound to the package.

    ``args`` carries ``rev1_package_manifest``, ``table`` and the list
    ``source`` of ``tag:arm_matrix.json:summary.json`` specs. Returns the
    output-schema dict (``OUTPUT_KEYS``); every binding failure is a ``SystemExit``."""
    manifest, pkg, manifest_sha = pkgmod.load_manifest(args.rev1_package_manifest)
    pkgmod.verify_package(args.rev1_package_manifest)
    suite = manifest["suite"]
    rev1_matrix = pkgmod.load_json_member(manifest, pkg, "matrix")
    library_sha = rev1_matrix["library_sha256"]
    fit_record = json.loads(pkgmod.verify_member(manifest, pkg, "fit.s0").read_text())
    table_sha = _sha(args.table)
    if (fit_record.get("input_digests") or {}).get("table") != table_sha:
        raise SystemExit("--table is not the shadow table the s-only fit record was built from")
    membership = fit_record.get("dev_membership")
    if not membership or _digest_obj(membership) != fit_record.get("dev_membership_sha256"):
        raise SystemExit("fit record development membership is missing or its digest drifted")
    table = load_table(args.table, ref_mode="fresh")
    dev_mask = dev_mask_from_membership(table, membership)

    seen: set[str] = set()
    sources = {}
    for spec in args.source:
        parts = spec.split(":", 2)
        if len(parts) != 3:
            raise SystemExit(f"--source {spec!r} must be tag:arm_matrix.json:summary.json")
        tag, matrix_path, summary_path = parts
        if tag in seen:
            raise SystemExit(f"duplicate --source tag {tag!r}")
        seen.add(tag)
        bound = bind_source(tag, matrix_path, summary_path, manifest_sha=manifest_sha, suite=suite,
                            library_sha256=library_sha)
        matrix, summary = bound["matrix"], bound["summary"]
        arms = {}
        for arm, a in summary["arms"].items():
            path = matrix["artifact_paths"][arm]
            if _sha(path) != matrix["artifact_sha256"][arm]:
                raise SystemExit(f"source {tag}: artifact of {arm} drifted since emit")
            art = load_surface_artifact(path)
            if art.retrieval_contract.get("library_sha256") != library_sha:
                raise SystemExit(f"source {tag}: artifact of {arm} was calibrated for another library")
            predicted = predicted_ir_for_artifact(art, table, dev_mask)
            measured = 100.0 * float(a["cost"]) / _COST["miss"]
            arms[arm] = {"family": a["family"], "estimator": (matrix.get("estimator") or {}).get(arm),
                         "predicted_ir": predicted, "measured_ir": measured, "gap": measured - predicted,
                         "artifact_sha256": matrix["artifact_sha256"][arm]}
        sources[tag] = {"matrix_sha256": bound["matrix_sha256"], "summary_sha256": bound["summary_sha256"],
                        "protocol": matrix["protocol"], "gate_type": matrix.get("gate_type"),
                        "gate_theta": matrix.get("gate_theta"), "arms": arms}
    return {"table_sha256": table_sha, "dev_membership_sha256": fit_record["dev_membership_sha256"],
            "n_dev_rows": int(dev_mask.sum()), "c_miss": _COST["miss"], "sources": sources}


def plot(result: dict, out_fig: str) -> None:
    """Scatter of realized against predicted IR per arm, gated sources hollow, y = x reference."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6.4, 6.0))
    lo, hi = 100.0, 0.0
    for tag, src in result["sources"].items():
        xs = [a["predicted_ir"] for a in src["arms"].values()]
        ys = [a["measured_ir"] for a in src["arms"].values()]
        gated = src.get("gate_type") == "score_hysteresis"
        ax.scatter(xs, ys, s=42, facecolors="white" if gated else None, edgecolors="k" if gated else None,
                   linewidths=1.2 if gated else 0.0, label=f"{tag} ({src.get('gate_type')})", zorder=4)
        for arm, a in src["arms"].items():
            ax.annotate(arm.replace("dsp_", ""), (a["predicted_ir"], a["measured_ir"]), xytext=(3, 3),
                        textcoords="offset points", fontsize=6)
        lo, hi = min(lo, *xs, *ys), max(hi, *xs, *ys)
    ax.plot([lo - 2, hi + 2], [lo - 2, hi + 2], ":", color="#666666", lw=1.1, label="y = x")
    ax.set_xlabel("predicted inference ratio on the development rows (%)")
    ax.set_ylabel("realized inference ratio in rollout (%)")
    ax.set_title("IR calibration: shadow-table prediction vs closed-loop measurement")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out_fig, dpi=170)
    plt.close(fig)


def main() -> None:
    """Command-line entry point: write the JSON table and the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--rev1-package-manifest", required=True)
    ap.add_argument("--table", required=True)
    ap.add_argument("--source", action="append", required=True, help="tag:arm_matrix.json:summary.json")
    ap.add_argument("--out-json", required=True)
    ap.add_argument("--out-fig", required=True)
    args = ap.parse_args()
    result = calibrate(args)
    pathlib.Path(args.out_json).write_text(json.dumps(result, indent=2, sort_keys=True))
    plot(result, args.out_fig)
    for tag, src in result["sources"].items():
        for arm, a in src["arms"].items():
            print(f"{tag:10s} {arm:20s} predicted {a['predicted_ir']:6.2f}  measured {a['measured_ir']:6.2f}  gap {a['gap']:+6.2f}")


if __name__ == "__main__":
    main()
