"""Round-2 vs round-1 comparison + bucket attribution for the ws2 search.

Three subcommands, all reading the finalizer's round-1-shaped per-cell files
(plan §3-W6):

- ``compare``: the two estimands, kept in separate tables (plan D1).
  Full-matrix table — per cid present in both rounds, the (cid, task, idx)
  paired delta of the JOINT intervention (library growth + text-IVF); never
  split into factors. Matched-control table — ONLY the manifest's ws2c cells,
  three-point decomposition ws1 -> ws2c (library effect) -> ws2 (bucket
  margin), each leg paired-tested. A decomposition request for a cell without
  a ws2c pairing is a hard error.
- ``buckets``: joins per-step evidence (prompt header + ``__hit_meta__`` rows)
  with accepted journal rows and ``bucket_variants.json``; reports, per
  (cid, task, idx), the eval prompt vs the winner's bucket variant and the
  aggregate match rate per task. Retries: rows the scheduler fenced
  (``accepted: False``) are dropped first, then only the block from the last
  surviving header row counts — a fenced attempt can ARRIVE after the accepted
  one, so position alone would pick the stale evidence.
- ``reproduce``: ws2e idx 0-7 against the same ws2 cells/idx — the same-seed
  reproduction matrix (success->success etc.).

Paired testing reuses the round-1 analyzer's ``signflip_p`` verbatim
(seed default 12345, consumed in a frozen cell order).
"""

from __future__ import annotations

import argparse
import json
import pathlib
import random
from typing import Any

from exp.robocasa365.analyze_ws_search_stats import load_journals, macro_sr, signflip_p

# The per-step header sentinel, mirrored from ``ws2_episode_runner`` rather
# than imported: that module pulls the whole conductor/runner stack in, which
# an offline analysis tool must not need. Both sides are pinned by
# ``test_ws2_evidence_runner``.
HEADER_STEP_IDX = -1


def require_full_matrix(
    arms: dict[str, dict], grids: dict[str, list], *, cids: list[str],
    episodes: int, tasks: int, label: str,
) -> None:
    """Refuse to report an estimand the data does not support.

    Silently intersecting arms is the dangerous default: a missing task, an
    undrained cell or a grid that drifted between rounds would change the
    inference population — which cells the PickPlace revival readout covers,
    which cells the factor split is computed on — while still printing a table
    headed as the full result.
    """
    problems = []
    for arm, cells in arms.items():
        missing = sorted(set(cids) - set(cells))
        extra = sorted(set(cells) - set(cids))
        if missing:
            problems.append(f"{arm}: {len(missing)}/{len(cids)} cells missing (first {missing[:3]})")
        if extra:
            # A superset changes the population just as much as a subset: a
            # probe directory or a superseded cell would silently enter the
            # frozen 132-cell estimand.
            problems.append(
                f"{arm}: {len(extra)} cell(s) outside the frozen matrix (first {extra[:3]})")
    expected = tasks * episodes
    for arm, grid in grids.items():
        if len(grid) != expected:
            observed_tasks = sorted({t for t, _ in grid})
            problems.append(
                f"{arm}: grid is {len(grid)} episodes over {len(observed_tasks)} tasks, "
                f"expected {expected} ({tasks} x {episodes})"
            )
    reference = next(iter(grids.values()))
    for arm, grid in grids.items():
        if sorted(grid) != sorted(reference):
            problems.append(f"{arm}: (task, idx) grid differs from the other arm(s)")
    if problems:
        raise SystemExit(
            f"{label}: refusing to report an incomplete matrix as a formal result:\n  - "
            + "\n  - ".join(problems)
            + "\nRe-run the missing cells, or pass --allow-partial for an explicitly "
            "non-formal mid-run read."
        )


def _paired(a: dict, b: dict, keys: list, rng: random.Random, resamples: int) -> tuple[float, float]:
    """Paired (task, idx) delta a-b and sign-flip p over the shared grid."""
    diffs = [int(a[k]) - int(b[k]) for k in keys]
    weight = 1.0 / len(keys)
    return signflip_p(diffs, weight, rng, resamples)


# ------------------------------------------------------------------
# compare
# ------------------------------------------------------------------


def cmd_compare(args: argparse.Namespace) -> None:
    ws1, keys1 = load_journals(pathlib.Path(args.ws1_dir), args.episodes, "ws1")
    ws2, keys2 = load_journals(pathlib.Path(args.ws2_dir), args.episodes, "ws2")
    keys = sorted(set(keys1) & set(keys2))
    if not keys:
        raise SystemExit("no shared (task, idx) grid between the two rounds")
    index_cids = sorted(json.loads(pathlib.Path(args.index).read_text()))
    if not args.allow_partial:
        require_full_matrix(
            {"ws1": ws1, "ws2": ws2}, {"ws1": keys1, "ws2": keys2},
            cids=index_cids, episodes=args.episodes, tasks=args.tasks, label="full-matrix",
        )
    tasks = sorted({t for t, _ in keys})
    idxs = sorted({i for _, i in keys})
    rng = random.Random(args.seed)

    # Formal mode reports exactly the frozen matrix (the gate above proved
    # both arms equal it); a partial read falls back to what is shared.
    shared = (sorted(index_cids) if not args.allow_partial
              else sorted(set(ws1) & set(ws2)))
    heading = "full-matrix JOINT effect" if not args.allow_partial else (
        "PARTIAL (NOT A FORMAL RESULT) joint effect")
    print(f"# {heading} (library growth + text-IVF together; "
          f"{len(shared)} cells, {len(keys)} paired episodes/cell)")
    print("cid, macro_ws1, macro_ws2, delta, p_paired")
    rows = []
    for cid in shared:
        m1 = macro_sr(ws1[cid], tasks, idxs)
        m2 = macro_sr(ws2[cid], tasks, idxs)
        _, p = _paired(ws2[cid], ws1[cid], keys, rng, args.resamples)
        rows.append((cid, m1, m2, m2 - m1, p))
        print(f"{cid}, {m1:.3f}, {m2:.3f}, {m2 - m1:+.3f}, {p:.4f}")

    print("\n# per-task joint effect (mean SR over shared cells; PickPlace revival readout)")
    for task in tasks:
        t_idx = [(task, i) for i in idxs]
        s1 = sum(ws1[c][k] for c in shared for k in t_idx) / (len(shared) * len(t_idx))
        s2 = sum(ws2[c][k] for c in shared for k in t_idx) / (len(shared) * len(t_idx))
        print(f"{task}, {s1:.3f}, {s2:.3f}, {s2 - s1:+.3f}")

    if args.csv:
        out = pathlib.Path(args.csv)
        out.write_text("cid,macro_ws1,macro_ws2,delta,p_paired\n" + "".join(
            f"{c},{m1:.4f},{m2:.4f},{d:+.4f},{p:.5f}\n" for c, m1, m2, d, p in rows))

    if not args.manifest:
        return
    manifest = json.loads(pathlib.Path(args.manifest).read_text())
    matched = list(manifest["segments"]["ws2c"]["cells"])
    ws2c, keysc = load_journals(pathlib.Path(args.ws2c_dir or args.ws2_dir), args.episodes, "ws2c")
    keys_m = sorted(set(keys) & set(keysc))
    missing = [c for c in matched if c not in ws2c or c not in ws1 or c not in ws2]
    if missing:
        raise SystemExit(
            f"matched-control decomposition requires all three arms for every manifest "
            f"cell; missing pairing for {missing}. Cells outside the manifest are NOT "
            "eligible for decomposition (plan D1)."
        )
    if not args.allow_partial:
        # Project each arm onto the manifest cells BEFORE the exact-set check:
        # the decomposition's population is those 12 cells, and the arms
        # legitimately hold more (ws1/ws2 are full matrices).
        projected = {
            "ws1": {c: ws1[c] for c in matched if c in ws1},
            "ws2c": {c: ws2c[c] for c in matched if c in ws2c},
            "ws2": {c: ws2[c] for c in matched if c in ws2},
        }
        require_full_matrix(
            projected,
            {"ws1": keys1, "ws2c": keysc, "ws2": keys2},
            cids=matched, episodes=args.episodes, tasks=args.tasks, label="matched-control",
        )
    # Every macro in this table is read over the three-arm SHARED grid, not the
    # main grid: a control arm missing one (task, idx) would otherwise KeyError,
    # and mixing grids between the legs would make the two deltas incomparable.
    tasks_m = sorted({t for t, _ in keys_m})
    idxs_m = sorted({i for _, i in keys_m})
    print(f"\n# matched-control decomposition (ONLY the {len(matched)} manifest cells, "
          f"{len(keys_m)} shared episodes/cell)")
    print("cid, macro_ws1, macro_ws2c, macro_ws2, lib_effect, p_lib, bucket_margin, p_bucket")
    for cid in sorted(matched):
        m1 = macro_sr(ws1[cid], tasks_m, idxs_m)
        mc = macro_sr(ws2c[cid], tasks_m, idxs_m)
        m2 = macro_sr(ws2[cid], tasks_m, idxs_m)
        _, p_lib = _paired(ws2c[cid], ws1[cid], keys_m, rng, args.resamples)
        _, p_bkt = _paired(ws2[cid], ws2c[cid], keys_m, rng, args.resamples)
        print(f"{cid}, {m1:.3f}, {mc:.3f}, {m2:.3f}, {mc - m1:+.3f}, {p_lib:.4f}, "
              f"{m2 - mc:+.3f}, {p_bkt:.4f}")


# ------------------------------------------------------------------
# buckets
# ------------------------------------------------------------------


def attempt_rows(rows: list[dict[str, Any]], accepted: dict[str, Any]) -> list[dict[str, Any]]:
    """The rows of exactly the attempt the journal accepted for this episode.

    The journal — not the evidence stream — decides which run counts, so the
    selection is a key match on ``(attempt, run_id)`` rather than a guess from
    position. Position cannot substitute: the driver keeps fenced results'
    rows (stamped ``accepted: False``) and appends them WHEN THEY ARRIVE, so a
    slow attempt can land after its replacement; and ``run_id`` is what
    separates a pre-crash attempt from its re-run, since dispatch generations
    restart at 1 in a fresh process. A key the rows cannot satisfy yields an
    EMPTY result, never a fallback: reporting another execution's evidence
    under this episode's outcome is the failure this function exists to
    prevent. Legacy rows carrying neither stamp still match, because the
    corresponding key is then absent from the accepted record too.
    """
    want_attempt = accepted.get("attempt")
    want_run = accepted.get("run_id")
    live = [row for row in rows if row.get("accepted") is not False]
    keyed = [
        row for row in live
        if (want_attempt is None or row.get("attempt") == want_attempt)
        and (want_run is None or row.get("run_id") == want_run)
    ]
    if not keyed:
        # NEVER fall back to whatever else is in the file. The journal named an
        # attempt/run; rows from a different one are a DIFFERENT execution --
        # after a crash-resume the file can still hold the pre-crash run's
        # prompt and winners, and using them would attribute one run's
        # retrieval evidence to another run's outcome.
        return []
    last_header = None
    for i, row in enumerate(keyed):
        if row.get("step_idx") == HEADER_STEP_IDX:
            last_header = i
    return keyed if last_header is None else keyed[last_header:]


def join_buckets(
    per_step_rows: list[dict[str, Any]],
    accepted: dict[str, dict[str, Any]],
    variants: dict[str, Any],
) -> list[dict[str, Any]]:
    """Per accepted episode: eval prompt vs the winner bucket's variant.

    ``accepted`` maps task_uid -> the accepted journal record, whose
    ``attempt``/``run_id`` select which evidence block is the real one.
    """
    to_bucket = variants["trajectory_to_bucket"]
    bucket_meta = {
        b["bucket_index"]: {
            "prompt": b["representative"].get("prompt"),
            "status": b["representative"].get("status"),
            "object_class": b["representative"].get("object_class"),
            "ambiguous": bool(b.get("ambiguous")),
        }
        for b in variants["buckets"]
    }
    by_uid: dict[str, list[dict[str, Any]]] = {}
    for row in per_step_rows:
        by_uid.setdefault(row["task_uid"], []).append(row)
    out = []
    # Every accepted episode is reported, including the ones with no usable
    # evidence: a silently shorter table hides its own denominator, so a
    # coverage gap would read as a clean result.
    for uid in sorted(accepted):
        raw = by_uid.get(uid, [])
        rows = attempt_rows(raw, accepted[uid]) if raw else []
        gap = None
        if not raw:
            gap = "missing_evidence"
        elif not rows:
            gap = "run_mismatch"
        if gap is not None:
            out.append({
                "task_uid": uid,
                "attempt": accepted[uid].get("attempt"),
                "run_id": accepted[uid].get("run_id"),
                "eval_prompt": None,
                "top_bucket": None,
                "bucket_variant_prompt": None,
                "bucket_variant_status": gap,
                "bucket_ambiguous": False,
                "bucket_object_class": None,
                "n_searches": 0,
                "n_unmapped_winners": 0,
                "matched": None,
            })
            continue
        header = next((r for r in rows if r.get("step_idx") == HEADER_STEP_IDX), None)
        winner_buckets: dict[int, int] = {}
        unmapped = 0
        for row in rows:
            wid = row.get("winner_id")
            if row.get("step_idx") == HEADER_STEP_IDX or not wid:
                continue
            trajectory = wid.rsplit(":", 1)[0]
            bucket = to_bucket.get(trajectory)
            if bucket is None:
                unmapped += 1
                continue
            winner_buckets[bucket] = winner_buckets.get(bucket, 0) + 1
        top = max(winner_buckets, key=winner_buckets.get) if winner_buckets else None
        eval_prompt = header.get("prompt") if header else None
        meta = bucket_meta.get(top, {}) if top is not None else {}
        variant_prompt = meta.get("prompt")
        if header is None:
            # Decision rows without the episode header: no eval-side truth to
            # compare against, so this is a gap, not a verdict.
            out.append({
                "task_uid": uid,
                "attempt": accepted[uid].get("attempt"),
                "run_id": accepted[uid].get("run_id"),
                "eval_prompt": None,
                "top_bucket": top,
                "bucket_variant_prompt": variant_prompt,
                "bucket_variant_status": "missing_header",
                "bucket_ambiguous": bool(meta.get("ambiguous")),
                "bucket_object_class": meta.get("object_class"),
                "n_searches": sum(winner_buckets.values()),
                "n_unmapped_winners": unmapped,
                "matched": None,
            })
            continue
        # An unresolved representative (replay failed) or an ambiguous bucket
        # (members from >1 task) cannot support a match verdict; both are
        # carried into the result rather than collapsing to a bare None.
        variant_status = meta.get("status") if top is not None else "no_winner"
        out.append({
            "task_uid": uid,
            "attempt": accepted[uid].get("attempt"),
            "run_id": accepted[uid].get("run_id"),
            "eval_prompt": eval_prompt,
            "top_bucket": top,
            "bucket_variant_prompt": variant_prompt,
            "bucket_variant_status": variant_status,
            "bucket_ambiguous": bool(meta.get("ambiguous")),
            "bucket_object_class": meta.get("object_class"),
            "n_searches": sum(winner_buckets.values()),
            "n_unmapped_winners": unmapped,
            "matched": (
                None if eval_prompt is None or variant_prompt is None
                or variant_status != "resolved" or meta.get("ambiguous")
                else eval_prompt == variant_prompt
            ),
        })
    return out


def cmd_buckets(args: argparse.Namespace) -> None:
    data_dir = pathlib.Path(args.data_dir)
    variants = json.loads(pathlib.Path(args.bucket_variants).read_text())
    results = []
    n_accepted = 0
    for journal_path in sorted(data_dir.glob(f"journal_{args.run_prefix}-*.jsonl")):
        run_id = journal_path.name[len("journal_"):-len(".jsonl")]
        accepted: dict[str, dict[str, Any]] = {}
        for line in journal_path.read_text().splitlines():
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("accepted") and rec.get("status") in ("done", "failed"):
                # Last accepted terminal record wins, matching summarize_journal.
                accepted[rec["task_uid"]] = rec
        n_accepted += len(accepted)
        per_step_path = data_dir / f"per_step_{run_id}.jsonl"
        # A missing per-step file is itself a finding: the cell's accepted
        # episodes are still reported, as missing_evidence.
        rows = (
            [json.loads(line) for line in per_step_path.read_text().splitlines() if line.strip()]
            if per_step_path.exists() else []
        )
        results.extend(join_buckets(rows, accepted, variants))
    judged = [r for r in results if r["matched"] is not None]
    matched = sum(1 for r in judged if r["matched"])
    gaps = {
        name: sum(1 for r in results if r["bucket_variant_status"] == name)
        for name in ("missing_evidence", "run_mismatch", "missing_header",
                     "unresolved", "no_winner")
    }
    ambiguous = sum(1 for r in results if r["bucket_ambiguous"] and r["matched"] is None)
    print(f"[buckets] accepted episodes={n_accepted} joined={len(results)} "
          f"judged={len(judged)} variant-match={matched}/{len(judged)}")
    print("[buckets] unjudged breakdown: "
          + " ".join(f"{k}={v}" for k, v in gaps.items())
          + f" ambiguous={ambiguous}")
    if args.out:
        pathlib.Path(args.out).write_text("".join(json.dumps(r) + "\n" for r in results))


# ------------------------------------------------------------------
# reproduce
# ------------------------------------------------------------------


def cmd_reproduce(args: argparse.Namespace) -> None:
    ws2, _ = load_journals(pathlib.Path(args.ws2_dir), args.episodes, "ws2")
    ws2e, _ = load_journals(pathlib.Path(args.ws2e_dir), args.ws2e_episodes, "ws2e")
    overlap_idx = set(range(args.episodes))
    ss = sf = fs = ff = 0
    for cid in sorted(set(ws2) & set(ws2e)):
        for (task, idx), first in ws2[cid].items():
            if idx not in overlap_idx or (task, idx) not in ws2e[cid]:
                continue
            second = ws2e[cid][(task, idx)]
            ss += first and second
            sf += first and not second
            fs += (not first) and second
            ff += (not first) and (not second)
    total = ss + sf + fs + ff
    print(f"[reproduce] shared episodes={total}  s->s={ss} s->f={sf} f->s={fs} f->f={ff}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("compare", help="joint full-matrix + matched-control tables")
    c.add_argument("--ws1-dir", required=True)
    c.add_argument("--ws2-dir", required=True)
    c.add_argument("--ws2c-dir", default="", help="default: --ws2-dir")
    c.add_argument("--manifest", default="", help="enables the matched-control table")
    c.add_argument("--episodes", type=int, default=8)
    c.add_argument("--tasks", type=int, default=13, help="tasks the frozen grid must cover")
    c.add_argument("--index", default="exp/robocasa365/config/ws_search/groot_tp/index.json",
                   help="the frozen cid matrix a formal result must cover in full")
    c.add_argument("--allow-partial", action="store_true",
                   help="mid-run read: skip the completeness gates and mark the output "
                   "NOT A FORMAL RESULT")
    c.add_argument("--resamples", type=int, default=20000)
    c.add_argument("--seed", type=int, default=12345)
    c.add_argument("--csv", default="")
    c.set_defaults(func=cmd_compare)

    b = sub.add_parser("buckets", help="attribution join: prompts vs winner buckets")
    b.add_argument("--data-dir", required=True, help="finalized ws2 data dir (journals + per_step)")
    b.add_argument("--bucket-variants", required=True)
    b.add_argument("--run-prefix", default="ws2")
    b.add_argument("--out", default="", help="write the per-episode join as jsonl")
    b.set_defaults(func=cmd_buckets)

    r = sub.add_parser("reproduce", help="same-seed ws2e idx0-7 vs ws2 matrix")
    r.add_argument("--ws2-dir", required=True)
    r.add_argument("--ws2e-dir", required=True)
    r.add_argument("--episodes", type=int, default=8)
    r.add_argument("--ws2e-episodes", type=int, default=32)
    r.set_defaults(func=cmd_reproduce)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
