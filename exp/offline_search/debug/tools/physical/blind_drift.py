"""E4 control-rate drift from followed demos, with certified physics only.

Without admitted backfill, robot-state drift is evaluated at recorded decision
checks against rs.npy. Intermediate controls are unavailable, never interpolated.
"""

from __future__ import annotations

import json
from pathlib import Path
import numpy as np

from .adapters import EnvironmentAdapter
from .common import (
    load_config,
    Episode,
    Unavailable,
    load_episodes,
    parser,
    report,
    safe_rows,
)
from .forensics import physical


class Library:
    def __init__(self, root=None, catalog=None, backfill_root=None):
        self.root = Path(root) if root else None
        self.catalog = {
            int(r["row"]): r
            for r in (
                catalog.to_dict("records")
                if hasattr(catalog, "to_dict")
                else catalog or []
            )
        }
        shas = {r["lib_sha"] for r in self.catalog.values() if r.get("lib_sha")}
        if len(shas) > 1:
            raise Unavailable("library catalog contains multiple fingerprints")
        self.lib_sha = next(iter(shas), None)
        self.rs = None
        self.scale = None
        if self.root and (self.root / "rs.npy").exists():
            self.rs = np.load(self.root / "rs.npy", mmap_mode="r", allow_pickle=False)
            success = np.load(
                self.root / "success.npy", mmap_mode="r", allow_pickle=False
            ).astype(bool)
            if success.any():
                scale = np.std(self.rs[success], axis=0)
                self.active = scale > np.finfo(float).eps * max(float(scale.max()), 1.0)
                self.scale = np.where(self.active, scale, 1.0)
        self.physics = {}
        self.rejected_backfills = []
        if backfill_root:
            for path in sorted(Path(backfill_root).rglob("episode.json")):
                meta = json.loads(path.read_text())
                cert = meta.get("backfill", {})
                if cert.get("admission") != "PASS" or meta.get("capture_errors"):
                    self.rejected_backfills.append(str(path))
                    continue
                if (
                    self.lib_sha is not None
                    and cert.get("lib_sha", cert.get("library_sha")) != self.lib_sha
                ):
                    self.rejected_backfills.append(
                        str(path) + ": missing/mismatched frozen library fingerprint"
                    )
                    continue
                blocks = []
                for block in sorted(path.parent.glob("controls_*.npz")):
                    with np.load(block, allow_pickle=False) as arrays:
                        blocks.append(
                            {
                                k: arrays[k]
                                for k in arrays.files
                                if not k.startswith("contact_")
                            }
                        )
                if not blocks:
                    continue
                controls = {
                    k: np.concatenate([b[k] for b in blocks]) for k in blocks[0]
                }
                ep = Episode(meta, controls)
                try:
                    ep.require("eef_pos", "obj_pos", "qpos", "decision_seq")
                except Unavailable as exc:
                    self.rejected_backfills.append(str(path) + ": " + str(exc))
                    continue
                rows = cert.get("library_rows", [])
                for seq, row in enumerate(rows):
                    hits = np.flatnonzero(controls["decision_seq"] == seq)
                    if len(hits):
                        if int(row) in self.physics:
                            raise Unavailable(
                                "duplicate admitted backfill for library row "
                                + str(row)
                            )
                        self.physics[int(row)] = (ep, int(hits[0]))

    def advance(self, rows, blocks):
        rows = np.asarray(rows, int).copy()
        for _ in range(blocks):
            nxt = []
            for row in rows:
                r = self.catalog.get(int(row))
                following = self.catalog.get(int(r.get("next", -1))) if r else None
                if (
                    following is None
                    or r["task_id"] != following["task_id"]
                    or r["episode"] != following["episode"]
                    or int(following["step"]) != int(r["step"]) + 1
                ):
                    nxt.append(-1)
                else:
                    nxt.append(int(following["row"]))
            rows = np.array(nxt, int)
        return rows


def kernel(decision, by_id, library, stride):
    diag = decision.get("diag", {}) or {}
    rows = diag.get("successor_cursor_rows", diag.get("follow_rows"))
    weights = decision.get("weights")
    anchor = by_id.get(decision.get("anchor_decision_id"), decision)
    if rows is None:
        rows = anchor.get("rows")
        weights = anchor.get("weights")
        if rows is None:
            raise Unavailable("missing followed anchor kernel")
        rows = library.advance(rows, int(decision.get("chunk_offset", 0)) // stride)
    if weights is None:
        weights = anchor.get("weights")
    rows, weights = np.asarray(rows, int), np.asarray(weights, float)
    if (
        rows.ndim != 1
        or weights.shape != rows.shape
        or not len(rows)
        or (rows < 0).any()
        or not np.isfinite(weights).all()
        or (weights < 0).any()
        or weights.sum() <= 0
    ):
        raise Unavailable("invalid or unsupported followed-demo kernel")
    return rows, weights / weights.sum()


def drift(episode, library, config=None):
    episode.require("decision_seq", "eef_pos", "qpos", "chunk_offset")
    adapter = EnvironmentAdapter(episode, config)
    by_id = {d.get("decision_id"): d for d in episode.decisions}
    stage = np.full(episode.n, "unknown", dtype="<U24")
    try:
        _, truth = physical(episode, config)
        stage = truth["stage"]
    except Unavailable:
        pass
    stride = int(
        episode.manifest.get("exec_steps", episode.meta.get("replan_steps", 5))
    )
    result = []
    for d in episode.decisions:
        if bool(d.get("vision", True)) or d.get("src") not in (
            "cache_tail",
            "follow",
            "cache",
        ):
            continue
        hits = np.flatnonzero(
            episode.controls["decision_seq"] == int(d["decision_seq"])
        )
        try:
            if not library.catalog:
                raise Unavailable(
                    "library row catalog unavailable; followed-demo support cannot be verified"
                )
            rows, weights = kernel(d, by_id, library, stride)
            if any(int(r) not in library.catalog for r in rows):
                raise Unavailable("followed row absent from library catalog")
            if any(
                int(library.catalog[int(r)]["task_id"]) != int(episode.meta["task_id"])
                for r in rows
            ):
                raise Unavailable("followed row task mismatch")
        except Unavailable as exc:
            result.append(
                dict(
                    episode.identity,
                    decision_id=d.get("decision_id"),
                    status="unavailable",
                    reason=str(exc),
                )
            )
            continue
        all_physics = all(int(r) in library.physics for r in rows)
        for t in hits:
            record = dict(
                episode.identity,
                decision_id=d.get("decision_id"),
                control_idx=int(t),
                truth_stage=str(stage[t]),
                blind_age_controls=int(d.get("blind_age_controls", 0))
                + int(episode.controls["chunk_offset"][t]),
                status="available",
                followed_rows=rows.tolist(),
                weights=weights.tolist(),
                object_relative_status="unavailable",
                robot_status="unavailable",
            )
            if all_physics:
                states, eefs, relative, names = [], [], [], []
                indices = adapter.settings.get(
                    "robot_qpos_indices", list(range(9)) if adapter.libero else None
                )
                if indices is None:
                    raise Unavailable("adapter must declare robot_qpos_indices")
                for row in rows:
                    demo, start = library.physics[int(row)]
                    offset = int(episode.controls["chunk_offset"][t])
                    point = start + offset
                    if point >= demo.n:
                        raise Unavailable("backfill control horizon has no support")
                    states.append(demo.controls["qpos"][point, indices])
                    eefs.append(demo.controls["eef_pos"][point])
                    da = EnvironmentAdapter(demo, config)
                    shared = sorted(set(adapter.names) & set(da.names))
                    names.append(shared)
                    relative.append(
                        np.array(
                            [
                                demo.controls["eef_pos"][point]
                                - demo.controls["obj_pos"][point, da.name_to_index[x]]
                                for x in shared
                            ]
                        )
                    )
                target = weights @ np.asarray(states)
                record.update(
                    robot_status="available",
                    robot_qpos_rms=float(
                        np.sqrt(
                            np.mean(
                                (episode.controls["qpos"][t, indices] - target) ** 2
                            )
                        )
                    ),
                    eef_drift=float(
                        np.linalg.norm(
                            episode.controls["eef_pos"][t] - weights @ np.asarray(eefs)
                        )
                    ),
                    reference="certified_control_rate_backfill",
                )
                if names[0] and all(x == names[0] for x in names):
                    actual = np.array(
                        [
                            episode.controls["eef_pos"][t]
                            - episode.controls["obj_pos"][t, adapter.name_to_index[x]]
                            for x in names[0]
                        ]
                    )
                    target_relative = np.einsum(
                        "k,kod->od", weights, np.asarray(relative)
                    )
                    record.update(
                        object_relative_status="available",
                        object_relative_rms=float(
                            np.sqrt(np.mean((actual - target_relative) ** 2))
                        ),
                        objects=names[0],
                    )
            elif (
                t == hits[0]
                and library.rs is not None
                and library.scale is not None
                and episode.reader_arm is not None
            ):
                try:
                    state = episode.reader_arm.decision_arrays(
                        ["state_norm"], [d["decision_id"]]
                    )["state_norm"][0]
                    width = min(len(state), library.rs.shape[1])
                    active = library.active[:width]
                    if not active.any():
                        raise Unavailable("no varying library robot-state coordinates")
                    delta = (
                        np.asarray(state[:width]) - weights @ library.rs[rows, :width]
                    ) / library.scale[:width]
                    if not np.isfinite(delta).all():
                        raise Unavailable("normalized robot state unavailable")
                    record.update(
                        robot_status="available",
                        robot_state_sigma_rms=float(
                            np.sqrt(np.mean(delta[active] ** 2))
                        ),
                        reference="rs_at_predecision_check",
                        evidence_control_idx=int(t) - 1,
                    )
                except (KeyError, Unavailable) as exc:
                    record["reason"] = str(exc)
            if record["robot_status"] == "unavailable":
                record.update(
                    status="unavailable",
                    reason=record.get(
                        "reason",
                        "no certified control-rate backfill; robot fallback only at decision checks",
                    ),
                )
            result.append(record)
    return result


def main(argv=None):
    ap = parser("blind_drift")
    ap.add_argument(
        "--library-root", type=Path, help="immutable store with rs.npy and success.npy"
    )
    ap.add_argument(
        "--backfill-root",
        type=Path,
        help="S2 client-format library replays; backfill.admission must be PASS",
    )
    ap.add_argument(
        "--catalog-file",
        type=Path,
        help="read-only shared rows.parquet; permits a scratch catalog without modifying the capture",
    )
    args = ap.parse_args(argv)
    episodes = load_episodes(args.run_root, args.arms, args.p3v2, args.limit)
    config = load_config(args, episodes)
    libraries = {}
    rows = []
    for ep in episodes:
        arm = ep.meta["arm"]
        if arm not in libraries:
            if args.catalog_file:
                import pandas as pd

                catalog = pd.read_parquet(args.catalog_file)
            else:
                catalog = ep.reader_arm.catalog() if ep.reader_arm is not None else None
            libraries[arm] = Library(args.library_root, catalog, args.backfill_root)
        rows.extend(safe_rows(drift, ep, libraries[arm], config))
    report(
        args.out,
        "blind_drift",
        {"blind_controls": rows},
        {
            "accepted_episodes": len(episodes),
            "certified_backfill_rows": {
                arm: len(lib.physics) for arm, lib in libraries.items()
            },
            "rejected_backfills": {
                arm: lib.rejected_backfills for arm, lib in libraries.items()
            },
            "between_check_excursions": "available only with certified control-rate backfill",
        },
        episodes=episodes,
    )


if __name__ == "__main__":
    main()
