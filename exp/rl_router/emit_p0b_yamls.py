#!/usr/bin/env python3
"""X15 U2 — emit the P0-b collection and risk-router serving configs.

    python emit_p0b_yamls.py --base <arm yaml> --out-dir <dir> \
        --shares 0.25,0.40,0.55,0.70 [--shadow-path <jsonl>]

P0-b is the run that answers whether similarity carries usable discrimination
at all, and it has to produce three things at once:

* a **threshold sweep** — the TIER judge at several calibrated shares, which is
  the discriminating baseline the risk router must beat;
* a **blind mixture** at matched shares, the non-discriminating control;
* a **dump** carrying diagnostics and pre-normalisation query keys, without
  which the offline pipeline has no parity baseline (that omission is what made
  an earlier revision's replay fail with "zero comparable decisions").

Shadow-teacher collection is emitted only when ``--shadow-path`` is given, so a
measurement-only sweep never pays for a teacher forward it does not need.

Key dependency: the arm yaml schema validated by
``openpi.cache.config.validate_cache_config``.
"""

from __future__ import annotations

import argparse
import copy
import pathlib

import yaml


def _load(path: str | pathlib.Path) -> dict:
    return yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))


def _strip_routing(cfg: dict) -> dict:
    """Drop any sidecar routing inherited from the base yaml.

    P0-b derives from an X14 arm config, which may carry ``routing.hit_to``
    pointing at the student sidecar. Left in place it breaks both controls in
    different ways: the blind arm (``arms="tc"``, no student) is rejected
    outright by config validation, while the threshold recipe loads happily and
    then executes the STUDENT on every FULL_HIT — so the "cache replay
    baseline" would silently measure a different policy. Neither failure is
    visible in the yaml itself, which is why this is stripped rather than
    assumed absent.
    """
    cfg.pop("routing", None)
    return cfg


def build_threshold_yaml(base: dict, *, threshold: float, dump_dir: str) -> dict:
    """A TIER threshold judge at one calibrated operating point.

    The judge sees only the fused top-1 score, which is the whole point: it is
    the hand-tuned discriminator the learned risk model is measured against.
    """
    cfg = _strip_routing(copy.deepcopy(base))
    cp1 = cfg["checkpoints"]["cp1"]
    config_id = f"p0b_threshold_{threshold:.3f}"
    cp1["judge"] = {
        "type": "threshold",
        "threshold": float(threshold),
        # The real dump seam is ``judge.dump`` (which makes _build_judge wrap
        # the judge in a DumpingJudge). A bare ``dump_dir`` key means nothing to
        # a threshold judge and would silently collect nothing — leaving the
        # offline pipeline with no rows at all.
        "dump": {"path": f"{dump_dir}/{config_id}.jsonl", "config_id": config_id},
    }
    _require_topk(cp1)
    return cfg


def build_blind_yaml(
    base: dict, *, p: float, dump_dir: str, seed: int, weights_root: str = ""
) -> dict:
    """A constant-rate mixture: the non-discriminating control at share ``p``.

    Uses the X14 router in constant mode rather than a new component, so the
    control and the treatment differ in the decision rule alone.
    """
    cfg = _strip_routing(copy.deepcopy(base))
    cp1 = cfg["checkpoints"]["cp1"]
    config_id = f"p0b_blind_{p:.2f}"
    # A constant TEACHER SHARE, not a constant arm. ``constant_arm`` pins one
    # arm for every step (a "blind mixture" at p would degenerate to always
    # cache), and there is no ``constant_p`` field for the loader to read —
    # it would be dropped silently. The share is carried by frozen weights
    # whose sampled policy is Bernoulli(p), which is what makes this the
    # matched non-discriminating control.
    cp1["judge"] = {
        "type": "mlp_router",
        "arms": "tc",
        "weights_path": f"{weights_root}/constant_p{p:.2f}.pt",
        "mode": "sample",
        "temperature": 1.0,
        "seed": seed,
        "feature_fields": ["robot_state"],
        "hidden": 8,
        "dump_dir": dump_dir,
        "dump": {"path": f"{dump_dir}/{config_id}.jsonl", "config_id": config_id},
    }
    _require_topk(cp1)
    return cfg


def build_risk_router_yaml(
    base: dict,
    *,
    risk_model_path: str,
    tau: float,
    task_index: int,
    replan_steps: int,
    library_replan_steps: int,
    dwell: int = 1,
    dump_dir: str = "",
    shadow_path: str = "",
) -> dict:
    """The X15 serving config.

    ``library_replan_steps`` is spelled out rather than defaulted: a library
    entry's ``step_idx`` counts the LIBRARY episode's inference cycles, so
    assuming it matches this client's interval silently rescales every phase
    feature.
    """
    cfg = _strip_routing(copy.deepcopy(base))
    cp1 = cfg["checkpoints"]["cp1"]
    cp1["judge"] = {
        "type": "risk_router",
        "risk_model_path": risk_model_path,
        "tau": float(tau),
        "dwell": int(dwell),
        "task_index": int(task_index),
        "replan_steps": int(replan_steps),
        "library_replan_steps": int(library_replan_steps),
        "dump_dir": dump_dir,
    }
    _require_topk(cp1)
    if shadow_path:
        cfg["shadow_teacher"] = {"enabled": True, "path": shadow_path}
    return cfg


def _require_topk(cp1: dict, k: int = 5) -> None:
    """The top-k features cannot be built from a top-1 search."""
    ss = cp1.setdefault("search_strategy", {})
    if int(ss.get("top_k", 0)) < k:
        ss["top_k"] = k


def emit(
    base_path: str,
    out_dir: str,
    *,
    shares: list[float],
    thresholds: list[float],
    dump_root: str,
    seed: int = 0,
    shadow_path: str = "",
    weights_root: str = "",
    validate: bool = True,
) -> list[str]:
    """Write the sweep and return the paths, newest config last."""
    base = _load(base_path)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    weights_dir = pathlib.Path(weights_root) if weights_root else out / "weights"
    written: list[str] = []

    for threshold in thresholds:
        dump_dir = pathlib.Path(dump_root) / f"threshold_{threshold:.3f}"
        dump_dir.mkdir(parents=True, exist_ok=True)
        cfg = build_threshold_yaml(
            base,
            threshold=threshold,
            dump_dir=str(dump_dir),
        )
        path = out / f"p0b_threshold_{threshold:.3f}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        written.append(str(path))

    for p in shares:
        dump_dir = pathlib.Path(dump_root) / f"blind_{p:.2f}"
        dump_dir.mkdir(parents=True, exist_ok=True)
        # Build the artifact the recipe references, rather than trusting it to
        # exist. With no explicit root, keep the sweep self-contained under the
        # emitted config directory instead of pointing at filesystem root.
        build_constant_share_weights(
            p,
            str(weights_dir / f"constant_p{p:.2f}.pt"),
        )
        cfg = build_blind_yaml(
            base,
            p=p,
            dump_dir=str(dump_dir),
            seed=seed,
            weights_root=str(weights_dir),
        )
        path = out / f"p0b_blind_{p:.2f}.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        written.append(str(path))

    if shadow_path:
        cfg = _strip_routing(copy.deepcopy(base))
        cfg["shadow_teacher"] = {"enabled": True, "path": shadow_path}
        path = out / "phase_a_shadow.yaml"
        path.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        written.append(str(path))

    if validate:
        _validate_emitted(written)
    return written


def build_constant_share_weights(p: float, out_path: str, *, hidden: int = 8) -> str:
    """Write frozen router weights whose sampled policy is Bernoulli(p).

    The blind control must actually mix at share ``p``. Emitting a yaml that
    merely points at a weights file is not enough: a missing or mis-scaled file
    fails only when the server starts — after the campaign has been scheduled —
    or, worse, loads and mixes at some other rate.

    The trunk is zeroed so the logits are constant regardless of input, and the
    bias pair is set to ``log(p)``/``log(1-p)``: softmax then reproduces ``p``
    exactly, independent of features.
    """
    import math

    import torch

    from openpi.cache.components.mlp_router_judge import save_router_weights

    if not 0.0 < p < 1.0:
        raise ValueError(f"constant share must lie in (0, 1), got {p}")

    dims = {"robot_state": 32}
    in_dim = sum(dims.values())
    # Zero trunk => hidden is all-zero => logits are exactly the output bias.
    path = pathlib.Path(out_path)
    save_router_weights(
        path,
        W1=torch.zeros(hidden, in_dim),
        b1=torch.zeros(hidden),
        W2=torch.zeros(2, hidden),
        # ARM order for "tc" is (teacher, cache); the teacher share is p.
        b2=torch.tensor([math.log(p), math.log1p(-p)]),
        arms="tc",
        fields=("robot_state",),
        dims=dims,
        weights_version=f"constant_p{p:.2f}",
        mu=torch.zeros(in_dim),
        sigma=torch.ones(in_dim),
    )
    return str(path)


def _validate_emitted(paths: list[str]) -> None:
    """Load every emitted recipe through the real loader.

    Checking field names is not enough: a config can name only real fields and
    still be rejected (an inherited ``routing.hit_to`` with no student arm) or,
    worse, load fine and execute a different policy. The only honest check is
    the one the server itself performs.
    """
    from openpi.cache.config import load_cache_config

    for path in paths:
        try:
            cfg = load_cache_config(path)
        except Exception as exc:  # noqa: BLE001 - re-raised with the culprit
            raise ValueError(
                f"emitted recipe {path} does not load: {exc}. A recipe that the "
                "server cannot load (or loads into a different policy) invalidates "
                "the comparison it was emitted for."
            ) from exc
        _validate_referenced_weights(path, cfg)


def _validate_referenced_weights(path: str, cfg) -> None:
    """Check that a recipe's weights artifact exists and carries the right arms.

    ``load_cache_config`` validates structure, not the files a config points at.
    For the blind control that gap is the difference between "mixes at share p"
    and "fails at server start" — or an arm-set mismatch the judge would reject
    mid-episode.
    """
    judge = cfg.checkpoints["cp1"].judge
    weights_path = getattr(judge, "weights_path", None)
    if not weights_path:
        return
    artifact = pathlib.Path(weights_path)
    if not artifact.exists():
        raise ValueError(
            f"emitted recipe {path} references weights {weights_path} which do "
            "not exist; the server would only discover this at startup"
        )
    from openpi.cache.components.mlp_router_judge import RouterWeights

    try:
        weights = RouterWeights.load(str(artifact))
    except Exception as exc:  # noqa: BLE001 - identify the emitted recipe
        raise ValueError(
            f"emitted recipe {path} references invalid router weights "
            f"{weights_path}: {exc}"
        ) from exc
    if weights.arms != judge.arms:
        raise ValueError(
            f"emitted recipe {path} wants arms={judge.arms!r} but its weights "
            f"declare arms={weights.arms!r}; the judge rejects the mismatch at startup"
        )


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("--base", required=True, help="arm yaml to derive from")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--shares", default="0.25,0.40,0.55,0.70")
    ap.add_argument("--thresholds", default="0.90,0.93,0.96,0.98")
    ap.add_argument("--dump-root", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument(
        "--weights-root",
        default="",
        help="constant-share weights directory (default: <out-dir>/weights)",
    )
    ap.add_argument(
        "--shadow-path",
        default="",
        help="enable Phase-A shadow collection at this sidecar path",
    )
    args = ap.parse_args()

    written = emit(
        args.base,
        args.out_dir,
        shares=[float(x) for x in args.shares.split(",") if x],
        thresholds=[float(x) for x in args.thresholds.split(",") if x],
        dump_root=args.dump_root,
        seed=args.seed,
        shadow_path=args.shadow_path,
        weights_root=args.weights_root,
    )
    for path in written:
        print(path)


if __name__ == "__main__":
    main()
