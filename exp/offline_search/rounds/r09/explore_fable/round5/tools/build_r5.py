"""Round 5 (fable): build the look-cost batch ``r09_fable_r5`` (LIBERO-10 50-demo, inits 20-29; NOT launched here).

Arms (same-batch controls are the r3c stacks as-is, pointing at their frozen r3c artifacts):
  pi0.5 : esc (control) | esc_wall | esc_weasy | esc_wpace | esc_fg
  GR00T : np (control)  | np_fg | np_fv
Sub-commands: spec, prefit, check, selftest, score (owner IR with wrist looks priced at .0646).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve()
REPO = next(q for q in HERE.parents if (q / "exp" / "offline_search").is_dir() and (q / "src").is_dir())
R5 = Path(os.environ.get("R9F_R5_ROOT", "/home/weiland/trace_runs/os_closed_loop/r09_fable_r5"))
R3C = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r3c")
STORE = "/home/weiland/trace_runs/offline_search_store"
MANIFEST_SRC = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r2/manifests/eval100_inits20_29.json")
M5 = "exp.offline_search.rounds.r09.explore_fable.round5.tools.methods"
R07_FITS = Path("/home/weiland/trace_runs/os_closed_loop/r07_main/fits")
CONTROL = {"pi05": "r9f3c_pi05_l10_50_np_corr05_esc", "groot": "r9f3c_groot_l10_50_np_corr05"}
CLASS = {"pi05": "LookCostEsc", "groot": "LookCostGroot"}
CPUS = os.environ.get("R9F_CPUS", "22-37,66-81")
EVAL_INITS = range(20, 30)
PRICE = {"pi05": dict(look=0.152, wrist=0.0646, call=0.848), "groot": dict(look=0.148, wrist=0.0646, call=0.852)}
CAMERA_ARGS = ["--os-request-cameras", "--os-tokens", "off"]
# variant name -> look-cost kwargs (nothing task-indexed)
VARIANTS = {
    "pi05": {"wall": dict(wrist_gate="all"), "weasy": dict(wrist_gate="easy"), "wpace": dict(wrist_gate="pace", pace_lag=1),
             "fg": dict(follow_blocks=1, follow_stage_gate=True)},
    "groot": {"fg": dict(follow_blocks=1, follow_stage_gate=True), "fv": dict(follow_blocks=1, follow_stage_gate=False)},
}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _control_row(model):
    rows = {a["arm"]: a for a in json.loads((R3C / "arms.json").read_text())}
    return rows[CONTROL[model]]


def build(run_root=R5):
    fits = Path(run_root) / "fits"
    arms, prefit, prov = [], {}, {}
    for model in ("pi05", "groot"):
        src = _control_row(model)
        stack_kwargs = json.loads(json.dumps(src["kwargs"]))
        if stack_kwargs.get("max_calls") != 0 or stack_kwargs.get("force_trigger_at") != []:
            raise ValueError("control stack must be the production r3c stack (max_calls 0, no forced triggers)")
        base_args = [a for a in src["plugin_args"] if a != "--os-fit-artifact" and not a.endswith(".pkl")]
        pre = f"r9f5_{model}_l10_50_" + ("esc" if model == "pi05" else "np")
        # same-batch control: the r3c stack as-is, its frozen artifact
        arms.append(dict(name=pre, model=model, suite="l10", mode="plugin", full_model=True, method=src["method"],
                         kwargs=stack_kwargs, cost_ledger=True, client_overrides=dict(src["client_overrides"]),
                         plugin_args=base_args + ["--os-fit-artifact", src["plugin_args"][-1]]))
        prov[pre] = dict(control_source=src["arm"], artifact=src["plugin_args"][-1], artifact_sha256=sha(src["plugin_args"][-1]))
        for tag, lc in VARIANTS[model].items():
            name = f"{pre}_{tag}"
            kw = {**stack_kwargs, "wrist_gate": "off", "pace_lag": 1, "follow_blocks": 0, "follow_stage_gate": True,
                  "stage_fit": "", "wrist_fit": "", **lc}
            if model == "pi05":
                kw["stage_fit"] = str(R07_FITS / "stages_pi05_l10_50.pkl")
                if kw["wrist_gate"] != "off":
                    kw["wrist_fit"] = str(R07_FITS / "wrist_pi05_l10_50.pkl")
            method = f"{M5}:{CLASS[model]}"
            prefit[name] = dict(method=method, kwargs=kw, cell=f"{model}_l10_cache", model=model, wrist=kw["wrist_gate"] != "off")
            args = base_args + (CAMERA_ARGS if kw["wrist_gate"] != "off" else []) + ["--os-fit-artifact", str(fits / f"{name}.pkl")]
            arms.append(dict(name=name, model=model, suite="l10", mode="plugin", full_model=True, method=method, kwargs=kw,
                             cost_ledger=True, client_overrides=dict(src["client_overrides"]), plugin_args=args))
    names = [a["name"] for a in arms]
    if len(set(names)) != len(names):
        raise ValueError("duplicate arm names")
    return arms, prefit, prov


def write_spec(run_root=R5):
    run_root = Path(run_root)
    arms, prefit, prov = build(run_root)
    for d in ("fits", "prefit_logs", "manifests", "selftest"):
        (run_root / d).mkdir(parents=True, exist_ok=True)
    (run_root / "arms_in.json").write_text(json.dumps(arms, indent=1))
    (run_root / "r5_kwargs.json").write_text(json.dumps(prefit, indent=1))
    (run_root / "provenance.json").write_text(json.dumps(prov, indent=1))
    manifest = json.loads(MANIFEST_SRC.read_text())
    if {(int(t), int(i)) for t, i in manifest} != {(t, i) for t in range(10) for i in EVAL_INITS}:
        raise ValueError("manifest is not tasks 0-9 x inits 20-29")
    shutil.copyfile(MANIFEST_SRC, run_root / "manifests" / "eval100_inits20_29.json")
    print(f"{len(arms)} arms, {len(prefit)} prefits -> {run_root}")
    return arms, prefit


def _env(threads="2"):
    return dict(os.environ, PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS=threads, OPENBLAS_NUM_THREADS=threads,
                MKL_NUM_THREADS=threads)


def _run(cmd, log, threads="2"):
    cmd = ["taskset", "-c", CPUS] + cmd
    with open(log, "w") as f:
        f.write("# " + " ".join(cmd) + "\n")
        f.flush()
        p = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, env=_env(threads), cwd=str(REPO))
    return p.returncode


def prefit(run_root=R5, workers=6, reg_name="r5_kwargs.json"):
    run_root = Path(run_root)
    reg = json.loads((run_root / reg_name).read_text())

    def one(item):
        name, spec = item
        art = run_root / "fits" / f"{name}.pkl"
        if art.exists():
            return name, "exists"
        cmd = [str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.plugin", "--os-method", spec["method"],
               "--os-kwargs", json.dumps(spec["kwargs"]), "--os-cell", spec["cell"], "--os-root", STORE,
               "--os-log-dir", str(run_root / "prefit_logs"), "--os-fit-artifact", str(art)]
        return name, f"rc={_run(cmd, run_root / 'prefit_logs' / f'{name}.log')}"

    with ThreadPoolExecutor(min(workers, len(reg))) as ex:
        for name, status in ex.map(one, reg.items()):
            print(f"{name}: {status}")
            if status not in ("exists", "rc=0"):
                raise SystemExit(f"prefit failed: {name} ({status})")


def check(run_root=R5, reg_name="r5_kwargs.json"):
    """Artifact-level assertions for every variant (r3c asserts + look-cost asserts)."""
    run_root = Path(run_root)
    sys.path.insert(0, str(REPO))
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import fingerprint
    from exp.offline_search.rounds.r08.abl.judge import TriggerCommitJudge, TriggerGrootCommitJudge
    reg = json.loads((run_root / reg_name).read_text())
    bad = []
    for name, spec in reg.items():
        with open(run_root / "fits" / f"{name}.pkl", "rb") as f:
            blob = pickle.load(f)
        obj = blob["method"]
        with open(spec["kwargs"]["onlynp_fit"], "rb") as f:
            src = pickle.load(f)["method"]
        fam = TriggerGrootCommitJudge if spec["model"] == "groot" else TriggerCommitJudge
        kw = spec["kwargs"]
        c = dict(judge_burst_unchanged=int(obj.burst) == int(src.burst) == 0, gm_burst=int(obj.gm_burst) == 2, max_calls=int(obj.gm_max_calls) == 0,
                 single_family=isinstance(obj, fam) and isinstance(src, fam), base_class=type(obj.base).__name__ == "CorrectedCacheJW",
                 blend_half=float(obj.base.blend) == 0.5, wrist_pass_clear=obj.base.wrist_pass is False,
                 same_library=bool(__import__("numpy").array_equal(obj.base.act, src.base.act)),
                 stage_fp=obj.lc_stages.retrieval_fingerprint == fingerprint(obj.base), ref_blocks=obj.lc_reference_blocks == 2 == obj.lc_stages.reference_blocks,
                 wrist=(obj.lc_wrist is not None) == (kw["wrist_gate"] != "off"), gate=obj.lc_wrist_gate == kw["wrist_gate"],
                 follow=(obj.lc_follow is not None) == bool(kw["follow_blocks"]), cell=blob.get("cell") == spec["cell"],
                 camera_full=obj.next_camera_mode == "full" and obj._lc_camera_mode == "full",
                 wrist_width=(obj.lc_wrist is None) or (obj.lc_wrist.B1T.shape[0] + obj.lc_state_width == 72),
                 esc=(hasattr(obj, "lag_threshold") and obj.lag_threshold == 12 and obj.deadline == 80) if spec["model"] == "pi05" else not hasattr(obj, "lag_threshold"))
        ok = all(c.values())
        print(f"{name:36s} {type(obj).__name__:14s} gate={obj.lc_wrist_gate:5s} follow={obj.lc_follow_blocks} {'OK' if ok else 'FAIL ' + str([k for k, v in c.items() if not v])}")
        if not ok:
            bad.append(name)
    if bad:
        raise SystemExit(f"artifact checks failed: {bad}")
    print(f"{len(reg)} variant artifacts pass all checks")


def selftest(run_root=R5, workers=6, episodes=4, reg_name="r5_kwargs.json"):
    """CPU plugin selftests with production kwargs (no per-request camera path in the selftest harness: wrist arms run
    with full looks there) and, for every arm, a forced empty-grasp variant exercising the recovery burst."""
    run_root = Path(run_root)
    reg = json.loads((run_root / reg_name).read_text())
    jobs = []
    for name, spec in reg.items():
        yaml = run_root / "config" / f"{name}.yaml"
        if not yaml.exists():
            raise SystemExit(f"emit the arms first: {yaml} missing")
        base = [str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.selftest", "--cell", spec["cell"], "--yaml", str(yaml),
                "--method", spec["method"], "--root", STORE, "--blind", "--policy-tail", "--policy-tail-blocks", "1", "--judge", "guard_only",
                "--episodes", str(episodes), "--no-shadow"]
        jobs.append((f"{name}_prod", base + ["--kwargs", json.dumps(spec["kwargs"]), "--fit-artifact", str(run_root / "fits" / f"{name}.pkl"),
                                              "--out", str(run_root / "selftest" / f"{name}_prod")]))
        forced = dict(spec["kwargs"], max_calls=2, force_trigger_at=[2, 6])
        jobs.append((f"{name}_forced", base + ["--kwargs", json.dumps(forced), "--out", str(run_root / "selftest" / f"{name}_forced")]))

    def one(job):
        tag, cmd = job
        rc = _run(cmd, run_root / "selftest" / f"{tag}.log", threads="1")
        out = (run_root / "selftest" / f"{tag}.log").read_text()
        passed = '"PASS": true' in out and "P2 does not support" not in out and rc == 0
        miss = out.split('"miss": ')[1].split(",")[0] if '"miss": ' in out else "?"
        return tag, passed, miss

    with ThreadPoolExecutor(min(workers, len(jobs))) as ex:
        results = list(ex.map(one, jobs))
    for tag, passed, miss in results:
        print(f"{tag:44s} {'PASS' if passed else 'FAIL'} miss={miss}")
    if not all(p for _, p, _ in results):
        raise SystemExit("selftest failures")
    print(f"{len(results)} selftests pass")


def arm_outcomes(run_root, arm, inits=EVAL_INITS):
    """SR and owner IR (full look .152/.148, wrist look .0646, call .848/.852) from journal + server decision logs.
    Records whose init is outside ``inits`` are dropped at parse time (never aggregated)."""
    run = Path(run_root) / "runs" / arm
    model = "groot" if "groot" in arm else "pi05"
    journal = run / "client" / "journal.jsonl"
    if not journal.exists():
        return None
    ok = {}
    for line in journal.open():
        r = json.loads(line)
        if not r.get("accepted") or r.get("status") not in ("done", "failed") or r.get("error"):
            continue
        t, i = int(r["task_uid"].split(":")[-2]), int(r["task_uid"].split(":")[-1])
        if i in inits:
            ok[(t, i)] = bool(r.get("success"))
    n = full = wrist = call = follow = 0
    for f in sorted(run.glob("server_*/decisions_*.jsonl")):
        for line in f.open():
            r = json.loads(line)
            if r.get("ev") != "dec" or int(r["uid"].split(":")[-1]) not in inits:
                continue
            n += 1
            if r.get("vision"):
                if r.get("camera_mode") == "wrist_only":
                    wrist += 1
                else:
                    full += 1
            if not r.get("hit", True):
                call += 1
            if float((r.get("extras") or {}).get("os_sf_source", 0) or 0) > 0:
                follow += 1
    if not ok:
        return None
    p = PRICE[model]
    v, w, c = (full / n, wrist / n, call / n) if n else (float("nan"),) * 3
    return dict(arm=arm, n=len(ok), sr=sum(ok.values()) / len(ok), n_dec=n, look=v, wrist=w, call=c, follow=follow / n if n else float("nan"),
                ir=p["look"] * v + p["wrist"] * w + p["call"] * c)


def score(run_root=R5, inits=EVAL_INITS):
    run_root = Path(run_root)
    arms = [a["name"] for a in json.loads((run_root / "arms_in.json").read_text())]
    rows = {a: arm_outcomes(run_root, a, inits) for a in arms}
    for a, v in rows.items():
        print(f"{a:36s} " + ("(missing)" if v is None else f"n={v['n']:3d} SR {v['sr']:.3f} IR {v['ir']:.3f} (full {v['look']:.3f} wrist {v['wrist']:.3f} call {v['call']:.3f} follow {v['follow']:.3f})"))
    (run_root / "score.json").write_text(json.dumps(dict(table=rows, inits=[inits.start, inits.stop - 1]), indent=1))
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("action", choices=["spec", "prefit", "check", "selftest", "score"])
    ap.add_argument("--run-root", default=str(R5))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--episodes", type=int, default=4)
    a = ap.parse_args(argv)
    if a.action == "spec":
        write_spec(a.run_root)
    elif a.action == "prefit":
        prefit(a.run_root, a.workers)
    elif a.action == "check":
        check(a.run_root)
    elif a.action == "score":
        score(a.run_root)
    else:
        selftest(a.run_root, a.workers, a.episodes)


if __name__ == "__main__":
    main()
