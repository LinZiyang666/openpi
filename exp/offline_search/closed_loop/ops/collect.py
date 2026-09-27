"""Pull an arm's timan107 client files (journal / per_step / driver.log, sha-checked) and summarize the arm.

    .venv/bin/python -m exp.offline_search.closed_loop.ops.collect --run-root <run-root> <arm> [<arm> ...] [--no-pull]

Completion judge (reference_conductor_journal_semantics): an episode is complete iff its journal row is
accepted AND status in {done, failed} AND carries no error (a normal task failure is a complete episode); the
success rate is success / complete over unique task_uids. Also reported: per-task success, retries, client-side
verdict mix (every decision must be FULL_HIT), server-side plugin logs (per-decision method / search / infer latency,
native-shadow agreement, executed == selected), throughput. Writes <run-root>/runs/<arm>/summary.json and merges
<run-root>/summary.json.

Mixed HIT/MISS arms (arms.json row with "judge" / "--os-judge" in plugin_args) get an extra "mixed" block: realized
hit rate h, MISS count, the inference ratio IR = 0.152 + 0.848 * (1 - h) (pi0.5 CUDA-graph three-stage formula) and
the measured one from the logged stage-1 (s1_ms) and stage-2/3 (s23_ms, MISS only) times, the judge reason mix, the
controller's tau (median over the second half of each server's decisions) and the check that the client verdict mix
equals the server log's hit / miss counts (replaces the all-FULL_HIT check, which is reported but expected False).
Pure-cache arms: output unchanged.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import pathlib
import subprocess
import sys
import tarfile
import time

import numpy as np

from exp.offline_search.closed_loop.ops.remote.run_gtp_subset import load_manifest, uid_pair, check_manifest
from exp.offline_search.rounds.r04.k4_eval.cost_ledger import ledger, r4_enabled
from exp.offline_search.rounds.r04.k4_eval.estimators import design_estimate

ISL = "/scratch/zixuans8/openpi_trace/os_cl"
IR_PI05 = (0.152, 0.848)


def is_mixed(meta: dict) -> bool:
    return meta.get("judge") is not None or "--os-judge" in (meta.get("plugin_args") or [])


def mixed_block(meta: dict, dd: list, decs: list, starts: list, comp: dict, ps_acc: list, hit_mix: dict) -> dict:
    """Mixed-arm statistics from the server decision rows dd (complete uids) and the client per_step rows."""
    n = len(dd)
    nh = sum(1 for r in dd if r.get("hit") is True)
    nm = sum(1 for r in dd if r.get("hit") is False)
    h = nh / n if n else None
    s1 = [r["s1_ms"] for r in dd if r.get("s1_ms") is not None]
    s23 = [r["s23_ms"] for r in dd if r.get("s23_ms") is not None]
    m1 = float(np.mean(s1)) if s1 else None
    m23 = float(np.mean(s23)) if s23 else None
    ir_meas = None
    if n and m1 is not None and m23 is not None:
        ir_meas = (n * m1 + nm * m23) / (n * (m1 + m23))          # (N_req*s1 + N_miss*(s2+s3)) / (N_req*sum s)
    jm: dict = {}
    for r in dd:
        j = r.get("judge")
        jm[str(j)] = jm.get(str(j), 0) + 1
    # controller tau: per server (tag), median over the second half of its decisions (deployable value)
    taus: dict = {}
    for r in sorted(dd, key=lambda r: r["ts"]):
        t = r.get("tau")
        if isinstance(t, (int, float)):
            taus.setdefault(r["tag"], []).append(float(t))
    tau_stats = {tag: {"n": len(v), "median_second_half": float(np.median(v[len(v) // 2:])), "last": v[-1]}
                 for tag, v in taus.items()}
    per_ep: dict = {}
    for r in dd:
        d = per_ep.setdefault(r["uid"], [0, 0])
        d[0] += 1
        d[1] += int(r.get("hit") is False)
    eps_with_miss = sum(1 for v in per_ep.values() if v[1] > 0)
    fail_uids = {u for u, r in comp.items() if not r.get("success")}
    miss_in_fail = sum(1 for r in dd if r.get("hit") is False and r.get("uid") in fail_uids)
    client_full = hit_mix.get("FULL_HIT", 0)
    client_miss = hit_mix.get("MISS", 0)
    out = {"judge": (starts[0].get("judge") if starts else None) or meta.get("judge"),
           "decisions": n, "hits": nh, "misses": nm, "h": round(h, 5) if h is not None else None,
           "miss_frac": round(nm / n, 5) if n else None,
           "ir_pi05_formula": round(IR_PI05[0] + IR_PI05[1] * (nm / n), 5) if n else None,
           "ir_measured": round(ir_meas, 5) if ir_meas is not None else None,
           "s1_ms": {"n": len(s1), "mean": m1, "p50": pct(s1, 50)},
           "s23_ms": {"n": len(s23), "mean": m23, "p50": pct(s23, 50), "p95": pct(s23, 95)},
           "judge_mix": jm, "tau": tau_stats,
           "episodes_with_miss": eps_with_miss, "episodes": len(per_ep),
           "miss_per_episode": round(nm / len(per_ep), 3) if per_ep else None,
           "miss_in_failed_episodes": miss_in_fail,
           "miss_in_failed_frac": round(miss_in_fail / nm, 4) if nm else None,
           "client_verdict_mix": hit_mix,
           "verdict_mix_matches_server": bool(ps_acc) and client_full == nh and client_miss == nm and
                                         set(hit_mix) <= {"FULL_HIT", "MISS"}}
    if meta.get("model") == "groot":
        out["ir_note"] = "ir_pi05_formula is the pi0.5 three-stage formula; for GR00T use ir_measured (s1 / s23 ms)"
    return out


def sh(cmd: list[str], timeout=600) -> str:
    return subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=timeout).stdout


def pull(run: pathlib.Path, arm: str) -> None:
    tag = run.name
    tar_remote = f"/tmp/oscl_{tag}_{arm}.tar"
    out = sh(["tether", "exec", "timan107", "--", "bash", "-c",
              f"cd {ISL}/runs/{tag} && tar -cf {tar_remote} {arm} && sha256sum {tar_remote}"])
    rsha = out.strip().split()[0]
    local = pathlib.Path(f"/tmp/oscl_local_{tag}_{arm}.tar")
    if local.exists():
        local.unlink()
    sh(["tether", "pull", f"timan107:{tar_remote}", str(local)])
    lsha = hashlib.sha256(local.read_bytes()).hexdigest()
    if lsha != rsha:
        raise SystemExit(f"{arm}: SHA MISMATCH remote {rsha} local {lsha}")
    dst = run / "runs" / arm / "client"
    dst.mkdir(parents=True, exist_ok=True)
    with tarfile.open(local) as tf:
        for m in tf.getmembers():
            if not m.isfile():
                continue
            p = pathlib.PurePosixPath(m.name)
            rel = pathlib.Path(*p.parts[1:])
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            with tf.extractfile(m) as src, open(dst / rel, "wb") as f:
                f.write(src.read())
    local.unlink()


def _jsonl(path):
    out = []
    if not path.exists():
        return out
    for line in open(path):
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return out


def pct(a, q):
    return float(np.percentile(a, q)) if len(a) else None


def arm_manifest(run, arm, meta, manifest=None):
    selection = run / "runs" / arm / "selection.json"
    if manifest is None and selection.exists():
        manifest = json.loads(selection.read_text()).get("manifest")
    elif manifest is None:
        manifest = meta.get("manifest")
    if not manifest:
        return None
    result = load_manifest(manifest)
    check_manifest(result, meta.get("model"), meta.get("suite_short") or meta.get("suite"))
    return result


def summarize(run: pathlib.Path, arm: str, *, manifest=None, include_ledger=False, cost_table=None, write=True) -> dict:
    arms = {r["arm"]: r for r in json.loads((run / "arms.json").read_text())}
    meta = arms.get(arm, {})
    design = arm_manifest(run, arm, meta, manifest)
    cd = run / "runs" / arm / "client"
    jr = _jsonl(cd / "journal.jsonl")
    comp = {}
    errors = 0
    for r in jr:
        if r.get("error"):
            errors += 1
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            comp[r["task_uid"]] = r
    if design:
        comp = {u: r for u, r in comp.items() if uid_pair(u) in design["selected"]}
    succ = sum(1 for r in comp.values() if r.get("success"))
    per_task = {}
    for uid, r in comp.items():
        t = int(uid.split(":")[2])
        d = per_task.setdefault(t, [0, 0])
        d[0] += 1
        d[1] += int(bool(r.get("success")))
    ps = [r for r in _jsonl(cd / "per_step.jsonl") if r.get("_kind") is None and "step_idx" in r]
    ps_acc = [r for r in ps if r.get("task_uid") in comp and r.get("accepted", True)]
    hit_mix = {}
    for r in ps_acc:
        hit_mix[r.get("hit_type")] = hit_mix.get(r.get("hit_type"), 0) + 1
    dur = [r.get("duration_s") for r in comp.values() if r.get("duration_s") is not None]
    ts = [r.get("ts") for r in comp.values() if r.get("ts") is not None]
    s = {"arm": arm, "model": meta.get("model"), "suite": meta.get("suite"), "mode": meta.get("mode"),
         "method": meta.get("method"), "kwargs": meta.get("kwargs"),
         "journal_rows": len(jr), "error_rows": errors, "complete": len(comp), "success": succ,
         "sr": round(succ / len(comp), 4) if comp else None,
         "per_task": {str(k): {"n": v[0], "success": v[1]} for k, v in sorted(per_task.items())},
         "retried": sum(1 for r in comp.values() if int(r.get("attempt", 1)) > 1),
         "client_decisions": len(ps_acc), "decisions_per_episode": round(len(ps_acc) / len(comp), 2) if comp else None,
         "client_hit_mix": hit_mix,
         "all_full_hit": bool(ps_acc) and set(hit_mix) == {"FULL_HIT"},
         "episode_duration_s": {"p50": pct(dur, 50), "p95": pct(dur, 95)},
         "wall_min_first_to_last_episode": round((max(ts) - min(ts)) / 60, 2) if len(ts) > 1 else None}
    # server-side plugin logs
    decs, epis, starts = [], [], []
    for f in sorted(glob.glob(str(run / "runs" / arm / "server_*" / "decisions_*.jsonl"))):
        for r in _jsonl(pathlib.Path(f)):
            ev = r.get("ev")
            if ev == "dec":
                decs.append(r)
            elif ev == "episode":
                epis.append(r)
            elif ev == "startup":
                starts.append(r)
    if decs:
        comp_uids = set(comp)
        dd = [r for r in decs if r.get("uid") in comp_uids] or decs
        modern = r4_enabled(meta, decs, starts) or include_ledger or cost_table is not None or design is not None
        if is_mixed(meta) or modern:
            # the accepted attempt only (a retried uid's earlier attempts are logged too)
            att = {u: int(r.get("attempt", 1) or 1) for u, r in comp.items()}
            dd = [r for r in decs if r.get("uid") in comp_uids and int(r.get("attempt", 1) or 1) == att[r["uid"]]] or dd
        if modern:
            dd = list({(r["uid"], r["step"]): r for r in decs if r.get("uid") in comp_uids
                       and int(r.get("attempt", 1) or 1) == att[r["uid"]]}.values())
        q = [r["q_us"] for r in dd if r.get("q_us") is not None]
        su = [r["search_us"] for r in dd if r.get("search_us") is not None]
        im = [r["infer_ms"] for r in dd if r.get("infer_ms") is not None]
        nat = [r for r in dd if r.get("agree") is not None]
        tsd = [r["ts"] for r in dd]
        s["server"] = {
            "decisions_logged": len(dd), "decisions_all_logged": len(decs), "servers": len(starts),
            "method": starts[0].get("method") if starts else None,
            "fit_s": [st.get("fit_s") for st in starts], "bytes_per_entry": starts[0].get("bytes_per_entry") if starts else None,
            "method_query_us": {"p50": pct(q, 50), "p95": pct(q, 95), "p99": pct(q, 99)},
            "plugin_search_us": {"p50": pct(su, 50), "p95": pct(su, 95)},
            "server_infer_ms": {"p50": pct(im, 50), "p95": pct(im, 95), "p99": pct(im, 99)},
            "native_shadow_agree": round(sum(1 for r in nat if r["agree"]) / len(nat), 5) if nat else None,
            "exec_ok": round(sum(1 for r in dd if r.get("exec_ok")) / len(dd), 5) if dd else None,
            "synth_frac": round(sum(1 for r in dd if r.get("synth")) / len(dd), 5) if dd else None,
            "lib_mix": {k: sum(1 for r in dd if r.get("lib") == k) for k in sorted({r.get("lib") for r in dd}, key=str)},
            "not_ok": sum(1 for r in decs if not r.get("ok", True)),
            "decisions_per_s": round(len(tsd) / (max(tsd) - min(tsd)), 3) if len(tsd) > 1 and max(tsd) > min(tsd) else None,
            "server_episode_rows": len(epis),
            "server_success_agrees_journal": (sum(1 for e in epis if e.get("uid") in comp and e.get("reason") ==
                                                  "episode_end" and bool(e.get("success")) ==
                                                  bool(comp[e["uid"]].get("success"))),
                                              sum(1 for e in epis if e.get("uid") in comp and e.get("reason") ==
                                                  "episode_end")),
        }
        if is_mixed(meta):
            s["mixed"] = mixed_block(meta, dd, decs, starts, comp, ps_acc, hit_mix)
        if modern:
            s["cost_ledger"] = ledger(meta, dd, starts, comp, cost_table)
            if "mixed" in s and r4_enabled(meta, dd, starts):
                s["mixed"]["ir_pi05_formula"] = (round(s["cost_ledger"]["ir_per_five_controls"], 5)
                                                   if dd and meta.get("model") == "pi05" else None)
                s["mixed"]["ir_measured"] = s["cost_ledger"]["ir_measured_per_five_controls"]
    if design:
        s["weighted"] = design_estimate(design, {uid_pair(u): int(bool(r.get("success"))) for u, r in comp.items()})
        s["weighted"]["sr"] = s["weighted"]["estimate"]
    s["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
    if not write:
        return s
    (run / "runs" / arm / "summary.json").write_text(json.dumps(s, indent=1))
    allp = run / "summary.json"
    allv = json.loads(allp.read_text()) if allp.exists() else {}
    allv[arm] = s
    allp.write_text(json.dumps(allv, indent=1))
    return s


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", required=True)
    ap.add_argument("--no-pull", action="store_true")
    ap.add_argument("--manifest", help="exact pairs, with strata/inclusion probabilities for weighted SR")
    ap.add_argument("--ledger", action="store_true", help="add the R4 ledger even for old logs")
    ap.add_argument("--cost-table", help="stage cost JSON (default closed_loop/ops/cost_table.json)")
    ap.add_argument("--no-write", action="store_true", help="read-only local summary; requires --no-pull")
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args(argv)
    if a.no_write and not a.no_pull:
        ap.error("--no-write requires --no-pull")
    run = pathlib.Path(a.run_root)
    rc = 0
    for arm in a.arms:
        if not a.no_pull:
            try:
                pull(run, arm)
            except Exception as e:  # noqa: BLE001
                print(f"{arm}: pull failed: {e}", file=sys.stderr)
                rc = 1
                continue
        s = summarize(run, arm, manifest=a.manifest, include_ledger=a.ledger, cost_table=a.cost_table, write=not a.no_write)
        sv = s.get("server", {})
        print(f"{arm}: complete={s['complete']} success={s['success']} SR={s['sr']} full_hit={s['all_full_hit']} "
              f"dec/ep={s['decisions_per_episode']} infer_ms_p50={sv.get('server_infer_ms', {}).get('p50')} "
              f"q_us_p50={sv.get('method_query_us', {}).get('p50')} native_agree={sv.get('native_shadow_agree')} "
              f"exec_ok={sv.get('exec_ok')}")
        mx = s.get("mixed")
        if mx:
            print(f"{arm}: MIXED h={mx['h']} misses={mx['misses']}/{mx['decisions']} IR_formula={mx['ir_pi05_formula']} "
                  f"IR_measured={mx['ir_measured']} s1_ms={mx['s1_ms']['mean']} s23_ms={mx['s23_ms']['mean']} "
                  f"verdict_mix_matches_server={mx['verdict_mix_matches_server']} judge_mix={mx['judge_mix']}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
