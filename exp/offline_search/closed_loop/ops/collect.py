"""Pull an arm's timan107 client files (journal / per_step / driver.log, sha-checked) and summarize the arm.

    .venv/bin/python -m exp.offline_search.closed_loop.ops.collect --run-root <run-root> <arm> [<arm> ...] [--no-pull]

Completion judge (reference_conductor_journal_semantics): an episode is complete iff its journal row is
accepted AND status in {done, failed} AND carries no error (a normal task failure is a complete episode); the
success rate is success / complete over unique task_uids. Also reported: per-task success, retries, client-side
verdict mix (every decision must be FULL_HIT), server-side plugin logs (per-decision method / search / infer latency,
native-shadow agreement, executed == selected), throughput. Writes <run-root>/runs/<arm>/summary.json and merges
<run-root>/summary.json.
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

ISL = "/scratch/zixuans8/openpi_trace/os_cl"


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


def summarize(run: pathlib.Path, arm: str) -> dict:
    arms = {r["arm"]: r for r in json.loads((run / "arms.json").read_text())}
    meta = arms.get(arm, {})
    cd = run / "runs" / arm / "client"
    jr = _jsonl(cd / "journal.jsonl")
    comp = {}
    errors = 0
    for r in jr:
        if r.get("error"):
            errors += 1
        if r.get("accepted") and r.get("status") in ("done", "failed") and not r.get("error"):
            comp[r["task_uid"]] = r
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
            "exec_ok": round(sum(1 for r in dd if r.get("exec_ok")) / len(dd), 5),
            "synth_frac": round(sum(1 for r in dd if r.get("synth")) / len(dd), 5),
            "lib_mix": {k: sum(1 for r in dd if r.get("lib") == k) for k in sorted({r.get("lib") for r in dd})},
            "not_ok": sum(1 for r in decs if not r.get("ok", True)),
            "decisions_per_s": round(len(tsd) / (max(tsd) - min(tsd)), 3) if len(tsd) > 1 else None,
            "server_episode_rows": len(epis),
            "server_success_agrees_journal": (sum(1 for e in epis if e.get("uid") in comp and e.get("reason") ==
                                                  "episode_end" and bool(e.get("success")) ==
                                                  bool(comp[e["uid"]].get("success"))),
                                              sum(1 for e in epis if e.get("uid") in comp and e.get("reason") ==
                                                  "episode_end")),
        }
    s["collected_at"] = time.strftime("%Y-%m-%dT%H:%M:%S%z")
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
    ap.add_argument("arms", nargs="+")
    a = ap.parse_args(argv)
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
        s = summarize(run, arm)
        sv = s.get("server", {})
        print(f"{arm}: complete={s['complete']} success={s['success']} SR={s['sr']} full_hit={s['all_full_hit']} "
              f"dec/ep={s['decisions_per_episode']} infer_ms_p50={sv.get('server_infer_ms', {}).get('p50')} "
              f"q_us_p50={sv.get('method_query_us', {}).get('p50')} native_agree={sv.get('native_shadow_agree')} "
              f"exec_ok={sv.get('exec_ok')}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
