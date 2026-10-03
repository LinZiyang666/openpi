"""R9Recipe unit tests: copied logic equals the originals, fitted state equals the reference artifacts, escalation and
budget semantics, self-containment of the serving pickle."""
from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.closed_loop.blind import LookReason
from exp.offline_search.harness import api
from exp.offline_search.rounds.r06.ideation_Q1.method_c.common import FitUnpickler
from exp.offline_search.rounds.r09.recipe import recipe as R

HERE = Path(__file__).resolve().parents[1]
ARMS_IN = HERE / "arms_in.json"
CELLS50 = ("pi05_l10_50", "groot_l10_50", "pi05_spatial_50", "groot_spatial_50")
STACK_CELLS = CELLS50 + ("pi05_l10_500", "groot_l10_500")
PACE_CELLS = ("pi05_spatial_500_look", "pi05_l10_500_look")
STORE = "/home/weiland/trace_runs/offline_search_store"


def _arms():
    if not ARMS_IN.exists():
        pytest.skip("arms not built")
    return {r["name"]: r for r in json.loads(ARMS_IN.read_text())}


def _fit(kw, cell):
    m = R.R9Recipe(**kw)
    m.fit(None, SimpleNamespace(cell=cell))
    return m


def test_predict_and_head_loader_equal_fable():
    from exp.offline_search.rounds.r09.explore_fable.round2.tools import methods as F
    head = str(HERE.parents[0] / "explore_fable/round2/out/corrector/head_pi05_l10_50_motion_pertask.npz")
    h1, m1 = R.load_head(head)
    h2, m2 = F.load_head(head)
    assert m1 == m2 and sorted(h1) == sorted(h2)
    rng = np.random.default_rng(0)
    for k in h1:
        for f in h1[k]:
            np.testing.assert_array_equal(h1[k][f], h2[k][f])
        X = rng.normal(size=(3, h1[k]["mean"].shape[-1])).astype(np.float32)
        np.testing.assert_array_equal(R._predict(h1[k], X), F._predict(h2[k], X))


def _same(a, b, path, depth=0):
    """Recursive structural equality (arrays exact, NaN == NaN, plain objects compared by their state)."""
    if depth > 12:
        return True
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b), err_msg=path)
        assert np.asarray(a).dtype == np.asarray(b).dtype, path
        return True
    if isinstance(a, dict):
        assert isinstance(b, dict) and sorted(a, key=str) == sorted(b, key=str), path
        return all(_same(a[k], b[k], f"{path}.{k}", depth + 1) for k in a)
    if isinstance(a, (list, tuple)):
        assert type(a) is type(b) and len(a) == len(b), path
        return all(_same(x, y, f"{path}[{i}]", depth + 1) for i, (x, y) in enumerate(zip(a, b)))
    if isinstance(a, float) and isinstance(b, float) and a != a and b != b:
        return True
    if hasattr(a, "__dict__") and not isinstance(a, type):
        assert type(a) is type(b), path
        return _same(vars(a), vars(b), path, depth + 1)
    assert a == b, path
    return True


@pytest.mark.parametrize("cell", STACK_CELLS)
def test_corrected_base_equals_fable_artifact(cell):
    from exp.offline_search.rounds.r09.recipe.tools.build_eq import recipe_kwargs, ref_row
    kw = recipe_kwargs(cell)
    ref = ref_row(cell)
    with open(ref["kwargs"]["corrected_fit"], "rb") as f:
        fable = FitUnpickler(f).load()["method"]
    mine = R.RecipeCorrectedBase.from_artifacts(kw["base_fit"], kw["head_path"], kw["blend"], ref["cell"])
    a, b = dict(vars(mine)), dict(vars(fable))
    a.pop("prof"), b.pop("prof")
    assert _same(a, b, "base")
    assert type(fable).__name__ == "CorrectedCache" and mine.blend == 0.5


@pytest.mark.parametrize("cell", STACK_CELLS)
def test_stack_fit_matches_reference_stack(cell):
    from exp.offline_search.rounds.r09.recipe.tools.build_eq import fit_arg, recipe_kwargs, ref_row
    ref = ref_row(cell)
    m = _fit(recipe_kwargs(cell), ref["cell"])
    with open(fit_arg(ref), "rb") as f:
        stack = pickle.load(f)["method"]
    j = m.inner
    for k in ("noprog_n", "prog_eps", "disabled_guards", "progress_guard", "burst", "stuck_guard", "policy_tail_gate", "monitor"):
        assert getattr(j, k) == getattr(stack, k), k
    assert type(j).__name__ in ("TriggerCommitJudge", "TriggerGrootCommitJudge")
    assert isinstance(j.base, R.RecipeCorrectedBase) and j.base.blend == stack.base.blend == 0.5
    np.testing.assert_array_equal(j.base.lib_step, stack.base.lib_step)
    np.testing.assert_array_equal(np.asarray(j.base.act), np.asarray(stack.base.act))
    assert m.escalation == (cell == "pi05_l10_50") == hasattr(stack, "_esc_step")
    if m.escalation:
        assert (m.lag_threshold, m.deadline) == (stack.lag_threshold, stack.deadline) == (12, 80)
    api.check_method_attrs(m)


def test_cache_mode_serves_the_a_artifact():
    from exp.offline_search.rounds.r09.recipe.tools.build_eq import fit_arg, recipe_kwargs, ref_row
    ref = ref_row("groot_spatial_500")
    m = _fit(recipe_kwargs("groot_spatial_500"), ref["cell"])
    with open(fit_arg(ref), "rb") as f:
        a = pickle.load(f)["method"]
    assert type(m.inner) is type(a) and type(a).__name__ == "BlindAWM"
    assert m.inner.lib == a.lib and m.inner.kref == a.kref == 8
    np.testing.assert_array_equal(np.asarray(m.inner.act), np.asarray(a.act))
    assert isinstance(m.policy_tail_step(SimpleNamespace()), LookReason)
    api.check_method_attrs(m)
    with pytest.raises(ValueError):
        R.R9Recipe(mode="cache", base_fit="x", escalation=True)


class _Judge:
    def __init__(self):
        self.base = SimpleNamespace(lib_step=np.arange(200))
        self._s = {"flag": []}
        self._noprog_span = 0
        self.script = {}

    def reset(self, episode):
        self._s = {"flag": []}

    def query(self, q):
        top1, reason = self.script.get(int(q.step), (int(q.step), 0.0))
        self._s["flag"].append(int(reason != 0))
        return api.Result(np.array([top1, 1]), np.zeros(2), 0.0, action=np.zeros((50, 32), np.float32), library="current",
                          extras=dict(os_force_miss=float(reason != 0), os_reason=float(reason), os_flags=0.0))

    def blind_step(self, bq):
        return LookReason(8, "noprog_span") if self._noprog_span > 0 else "BLIND"


def _stub(**kw):
    m = R.R9Recipe(mode="stack", **kw)
    m.inner = _Judge()
    return m


def _q(step, ep="x:eval:0:0", hits=None):
    hits = list(hits) if hits is not None else [1] * step
    return SimpleNamespace(step=step, episode=SimpleNamespace(uid=ep), hist_hit=np.array(hits[:step]),
                           hist_has_vision=np.ones(step, bool))


def test_escalation_persistent_after_lag_until_episode_change():
    m = _stub(escalation=True, lag_threshold=12, deadline=80)
    m.reset(SimpleNamespace(uid="x:eval:0:0"))
    m.inner.script = {20: (10, 0.0), 22: (8, 0.0), 24: (24, 4.0), 26: (26, 0.0)}   # lag 10, 14 -> escalate at 22
    out = {s: m.query(_q(s)).extras for s in (20, 22, 24, 26)}
    assert out[20]["os_force_miss"] == 0.0 and out[20]["r9o_lag"] == 10.0
    for s in (22, 24, 26):
        assert out[s]["os_force_miss"] == 1.0 and out[s]["os_reason"] == R.ESC_REASON and out[s]["r9o_esc_step"] == 22.0
    assert m.inner._s["flag"][-1] == 1
    out = m.query(_q(30, ep="x:eval:0:1")).extras                                     # new episode: not escalated
    assert out["os_force_miss"] == 0.0 and out["r9o_esc_step"] == -1.0


def test_escalation_deadline():
    m = _stub(escalation=True, lag_threshold=12, deadline=80)
    m.reset(SimpleNamespace(uid="x:eval:0:0"))
    m.inner.script = {82: (60, 0.0)}
    assert m.query(_q(82)).extras["os_force_miss"] == 0.0


def test_budget_counts_committed_calls_and_lifts_veto():
    m = _stub(call_budget=2)
    m.reset(SimpleNamespace(uid="x:eval:0:0"))
    m.inner.script = {6: (6, 4.0)}
    hits = [1, 0, 1, 1, 0, 1]                          # two committed calls before step 6
    ex = m.query(_q(6, hits=hits)).extras
    assert ex["os_force_miss"] == 0.0 and ex["r9o5_gate"] == 2.0 and ex["r9o5_gated_reason"] == 4.0
    m.inner._noprog_span = 3
    assert m.blind_step(_q(7, hits=hits + [1])) == "BLIND" and m.inner._noprog_span == 3
    m2 = _stub(call_budget=3)
    m2.reset(SimpleNamespace(uid="x:eval:0:0"))
    m2.inner.script = {6: (6, 4.0)}
    assert m2.query(_q(6, hits=hits)).extras["os_force_miss"] == 1.0
    m2.inner._noprog_span = 3
    assert isinstance(m2.blind_step(_q(7, hits=hits + [1])), LookReason)


def test_budget_off_leaves_verdicts_and_veto_alone():
    m = _stub()
    m.reset(SimpleNamespace(uid="x:eval:0:0"))
    m.inner.script = {6: (6, 4.0)}
    ex = m.query(_q(6, hits=[0] * 6)).extras
    assert ex["os_force_miss"] == 1.0 and "r9o5_gate" not in ex and "r9o_lag" not in ex
    m.inner._noprog_span = 2
    assert isinstance(m.blind_step(_q(7, hits=[0] * 7)), LookReason)


def test_constructor_validation():
    for bad in (dict(mode="x"), dict(lag_threshold=0), dict(deadline=-1), dict(call_budget=1.5), dict(call_budget=-1),
                dict(mode="pace_wrist", base_fit="a", wrist_fit="b", stage_fit="c"),                 # no follow kwargs
                dict(mode="pace_wrist", base_fit="a", wrist_fit="b", stage_fit="c", follow_kwargs={"x": 1}, escalation=True),
                dict(mode="stack", wrist_fit="b"), dict(mode="pace_wrist", base_fit="a", wrist_fit="b", stage_fit="c",
                                                         follow_kwargs={"x": 1}, pace_lag=-1)):
        with pytest.raises(ValueError):
            R.R9Recipe(**bad)


def test_source_imports_nothing_exploratory():
    src = (HERE / "recipe.py").read_text().splitlines()
    imports = [l for l in src if l.startswith(("import ", "from "))]
    assert imports and not any("explore_" in l for l in imports)


def test_serving_pickle_loads_without_exploratory_modules():
    arms = _arms()
    code = r'''
import sys, pickle, importlib.abc
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, name, path, target=None):
        if "explore_" in name:
            raise ImportError("blocked exploratory module " + name)
        return None
sys.meta_path.insert(0, Block())
from exp.offline_search.harness import api
for p in sys.argv[1:]:
    blob = pickle.load(open(p, "rb"))
    api.check_method_attrs(blob["method"])
    assert type(blob["method"]).__name__ == "R9Recipe"
print("OK", len(sys.argv) - 1, sorted(m for m in sys.modules if "explore_" in m))
'''
    paths = [a["plugin_args"][a["plugin_args"].index("--os-fit-artifact") + 1] for a in arms.values()]
    out = subprocess.run([sys.executable, "-c", code, *paths], capture_output=True, text=True,
                         cwd="/home/weiland/projects/openpi", env={"PYTHONPATH": ".:src", "PATH": "/usr/bin:/bin",
                                                                 "CUDA_VISIBLE_DEVICES": "", "HOME": "/home/weiland"})
    assert out.returncode == 0, out.stderr[-2000:]
    assert out.stdout.strip().startswith(f"OK {len(paths)} []")


def _artifact(name):
    arms = _arms()
    a = arms[name]
    with open(a["plugin_args"][a["plugin_args"].index("--os-fit-artifact") + 1], "rb") as f:
        return pickle.load(f)


def test_cell_defaults():
    arms = _arms()
    modes = {n.removeprefix("r9eq_"): a["kwargs"]["mode"] for n, a in arms.items()}
    assert modes == {"pi05_l10_50": "stack", "groot_l10_50": "stack", "pi05_spatial_50": "stack", "groot_spatial_50": "stack",
                     "pi05_l10_500": "stack", "groot_l10_500": "stack", "pi05_spatial_500": "cache",
                     "groot_spatial_500": "cache", "pi05_l10_500_look": "pace_wrist", "pi05_spatial_500_look": "pace_wrist"}
    esc = {n.removeprefix("r9eq_") for n, a in arms.items() if a["kwargs"].get("escalation")}
    assert esc == {"pi05_l10_50"}
    for n, a in arms.items():
        cams = "--os-request-cameras" in a["plugin_args"]
        assert cams == (a["kwargs"]["mode"] == "pace_wrist"), n
        if cams:
            i = a["plugin_args"].index("--os-tokens")
            assert a["plugin_args"][i + 1] == "off"
        assert a["kwargs"].get("call_budget") is None


@pytest.mark.parametrize("cell", PACE_CELLS)
def test_pace_wrist_state_equals_fable_artifact(cell):
    from exp.offline_search.rounds.r09.recipe.tools.build_eq import fit_arg, ref_row
    ref = ref_row(cell)
    with open(fit_arg(ref), "rb") as f:
        fable = pickle.load(f)["method"]
    mine = _artifact(f"r9eq_{cell}")["method"].inner
    assert type(fable).__name__ == "PaceWrist" and type(mine) is R.RecipePaceWrist
    assert type(mine.base).__name__ == type(fable.base).__name__ == "StageFollow"
    a, b = dict(vars(mine)), dict(vars(fable))
    for d in (a, b):
        d.pop("prof", None)
    assert _same(a, b, "pace_wrist")


def _episode(cell, k=0):
    from exp.offline_search.harness import store
    qc = store.QueryCell(STORE, cell)
    A = api.QueryArrays(qc)
    eps = [e for e in qc.episodes if e["num_steps"] >= 8]
    e = eps[k]
    return A, e, api.EpisodeView(e["uid"], e["task"], e["task_id"], e["init"], qc.episodes.index(e), 7)


def _qv(A, e, ev, step):
    start = int(e["start"])
    return api.QueryView(A, start + step, start, step, int(e["task_id"]), ev, None)


@pytest.mark.parametrize("cell,store_cell", [("pi05_spatial_500_look", "pi05_spatial_cache"), ("pi05_l10_500_look", "pi05_l10_cache")])
def test_pace_wrist_decisions_equal_fable_on_recorded_queries(cell, store_cell):
    """Full and wrist-only looks (the camera path the CPU selftest cannot drive) give identical results and plans."""
    from exp.offline_search.rounds.r09.recipe.tools.build_eq import fit_arg, ref_row
    if not Path(STORE).exists():
        pytest.skip("store not available")
    with open(fit_arg(ref_row(cell)), "rb") as f:
        fable = pickle.load(f)["method"]
    recipe = _artifact(f"r9eq_{cell}")["method"]
    n_wrist = 0
    for k in range(6):
        A, e, ev = _episode(store_cell, k)
        a, b = fable, recipe
        a.reset(ev)
        b.reset(ev)
        for step in (0, 2, 4):
            ra, rb = a.query(_qv(A, e, ev, step)), b.query(_qv(A, e, ev, step))
            np.testing.assert_array_equal(ra.topk, rb.topk)
            np.testing.assert_array_equal(ra.scores, rb.scores)
            np.testing.assert_array_equal(ra.action, rb.action)
            assert _same(dict(ra.extras), dict(rb.extras), "extras") and a.next_camera_mode == b.next_camera_mode
            mode = a.next_camera_mode
            n_wrist += mode == "wrist_only"
            a.set_camera_mode(mode)
            b.set_camera_mode(mode)
    assert n_wrist >= 1                     # the wrist-only retrieval path was exercised


def test_escalation_cap_then_back_to_stack():
    m = _stub(escalation=True, lag_threshold=12, deadline=80, escalation_max_calls=2)
    m.reset(SimpleNamespace(uid="x:eval:0:0"))
    m.inner.script = {20: (8, 0.0), 22: (8, 0.0), 24: (8, 4.0), 26: (8, 0.0)}    # escalate at 20 (lag 12)
    out = {s: m.query(_q(s)).extras for s in (20, 22, 24, 26)}
    assert [out[s]["os_reason"] for s in (20, 22)] == [R.ESC_REASON, R.ESC_REASON]
    assert out[24]["os_force_miss"] == 1.0 and out[24]["os_reason"] == 4.0          # cap spent: the stack's own verdict
    assert out[26]["os_force_miss"] == 0.0 and out[26]["r9o_escalated"] == 1.0 and out[26]["r9o_esc_calls"] == 2.0
    out = m.query(_q(30, ep="x:eval:0:1")).extras                                    # new episode: counter reset
    assert out["os_force_miss"] == 0.0 and out["r9o_esc_calls"] == 0.0


def test_escalation_cap_validation_and_old_pickles():
    with pytest.raises(ValueError):
        R.R9Recipe(escalation=False, escalation_max_calls=12)
    with pytest.raises(ValueError):
        R.R9Recipe(escalation=True, escalation_max_calls=0)
    m = _stub(escalation=True)
    del m.__dict__["escalation_max_calls"]                       # artifact pickled before the option existed
    m.reset(SimpleNamespace(uid="x:eval:0:0"))
    m.inner.script = {s: (0, 0.0) for s in range(20, 40, 2)}
    assert all(m.query(_q(s)).extras["os_reason"] == R.ESC_REASON for s in range(20, 40, 2))   # still persistent


def test_forced_escalation_debug_kwarg():
    with pytest.raises(ValueError):
        R.R9Recipe(force_escalation_at=[2])
    m = _stub(escalation=True, lag_threshold=12, deadline=80, force_escalation_at=[4])
    m.reset(SimpleNamespace(uid="x:eval:0:0"))
    out = {s: m.query(_q(s)).extras for s in (2, 4, 6)}                  # stub lag is 0 at every step
    assert out[2]["os_force_miss"] == 0.0 and out[4]["os_reason"] == R.ESC_REASON and out[6]["os_reason"] == R.ESC_REASON
