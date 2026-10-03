"""Integration: drive each v2 stack through the real judge chain (CPU plugin selftest on recorded store keys) with the
guard path AND forced empty-grasp triggers in the same episodes. Skipped when the frozen artifacts are absent."""
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest

R3 = Path("/home/weiland/trace_runs/os_closed_loop/r09_fable_r3")
STORE = "/home/weiland/trace_runs/offline_search_store"
ONLYNP = {"pi05": "/home/weiland/trace_runs/os_closed_loop/r08_abl/fits/r8abl_onlynp_p_l10_50.pkl",
          "groot": "/home/weiland/trace_runs/os_closed_loop/r08_abl/fits/r8abl_onlynp_g_l10_50.pkl"}
CORR = {"pi05": "/home/weiland/trace_runs/os_closed_loop/r09_fable_r2/fits/r9f2_pi05_l10_50_corr05pt.pkl",
        "groot": "/home/weiland/trace_runs/os_closed_loop/r09_fable_r2/fits/r9f2_groot_l10_50_corr05pt.pkl"}
CASES = [("pi05", "NpGraspStack2", dict(empty_aperture=0.001, closed_sign=1.0)),
         ("pi05", "NpGraspEsc2", dict(lag_threshold=12, deadline=80, empty_aperture=0.001, closed_sign=1.0)),
         ("groot", "NpGraspStackGroot2", dict(empty_aperture=0.0009, closed_sign=-1.0))]


@pytest.mark.parametrize("model,cls,extra", CASES)
def test_stack_selftest_with_guard_and_forced_triggers(model, cls, extra, tmp_path):
    yaml = R3 / "config" / ("r9f3_pi05_l10_50_np_corr05_gm.yaml" if model == "pi05" else "r9f3_groot_l10_50_np_corr05.yaml")
    if not (yaml.exists() and Path(ONLYNP[model]).exists() and Path(CORR[model]).exists() and Path(STORE).exists()):
        pytest.skip("frozen artifacts / store not available on this machine")
    kw = dict(onlynp_fit=ONLYNP[model], corrected_fit=CORR[model], hold_decisions=2, burst=2, max_calls=2, force_trigger_at=[2, 6], **extra)
    cmd = [sys.executable, "-m", "exp.offline_search.closed_loop.selftest", "--cell", f"{model}_l10_cache", "--yaml", str(yaml),
           "--method", f"exp.offline_search.rounds.r09.explore_fable.round3.tools.methods:{cls}", "--kwargs", json.dumps(kw),
           "--root", STORE, "--blind", "--policy-tail", "--policy-tail-blocks", "1", "--judge", "guard_only", "--episodes", "3",
           "--no-shadow", "--out", str(tmp_path / "st")]
    env = dict(os.environ, PYTHONPATH=".:src", CUDA_VISIBLE_DEVICES="", OMP_NUM_THREADS="1")
    repo = next(q for q in Path(__file__).resolve().parents if (q / "exp" / "offline_search").is_dir() and (q / "src").is_dir())
    p = subprocess.run(cmd, capture_output=True, text=True, timeout=1500, env=env, cwd=str(repo))
    out = p.stdout + p.stderr
    assert '"PASS": true' in out, out[-3000:]
    assert "P2 does not support" not in out
    n_miss = int(out.split('"miss": ')[1].split(",")[0])
    assert n_miss >= 2, out[-2000:]          # forced triggers (and any guard miss) produced policy calls through the real chain
