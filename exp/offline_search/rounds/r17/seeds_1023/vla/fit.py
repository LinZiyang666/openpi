"""New LIBERO subset seeds (owner 10-23), pi0.5 / GR00T: subsets, R10 fits, Q3 + periods, success-filtered knob fits.

CPU only; nothing is launched. Per seed S and cell {model}_{suite}_{N} (N = 50/100/200), under ROOT/s<S>/
(ROOT = /home/weiland/projects/openpi_ext/r17_seed_split, a link to /nvme/weiland/openpi/r17_seed_split: the H100
syncs mirror only /home and /data paths, so every path written into a fit or a definition goes through ROOT):
  subsets/<cell>.{npy,json}  r10/data.make_subsets with SEED = S: default_rng(S + task).permutation of the task's 50
                             parent episodes, first N/10, nested (never written into r10/subsets)
  cs/<cell>_cr.pkl           the subset's R10 fit (r10/recipe fit_recipe, arithmetic unchanged) in the cache study's
                             _cr form: R10Recipe.inner = the fitted corrected cache. fit_recipe builds the removed R8
                             judge; old_gate() allows that inside the fitting process only and the judge is dropped
                             before anything is written (servers keep refusing it)
  q3/<cell>/, PERIODS.json, Q3_FITS.json   q3_port_1020/fit.fit_cell on that base (periods from the subset's rows)
  CELLS.json                 kept episodes = the subset's successes (filter_failed/audit_counts.summarize), empty tasks
  fits/<cell>_<level>.pkl    success-filtered knob fits as q3_port_1020's (R15 TaskDescentKnob over Q3StallInner, all
                             ten tasks at one level, caps from PERIODS.json); empty tasks are refused until a
                             no-retrieval fallback is defined (c10 still queries the table and permits policy tails)

  prepare [--seeds S ...] [--workers 8]   all of the above with the knob-off fits (build.py levels adds level fits)
Run single-threaded BLAS (OMP/OPENBLAS/MKL_NUM_THREADS=1, as the original R10 build) for bit-identical fits.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
import pickle
import time
from types import SimpleNamespace as NS
from unittest.mock import patch

import numpy as np

from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020 import fit as q3
from exp.offline_search.rounds.r17.filter_failed_knob.q3_port_1020.fit import read, dump, LEVELS

OUT = Path('/nvme/weiland/openpi/r17_seed_split')
ROOT = Path('/home/weiland/projects/openpi_ext/r17_seed_split')  # -> OUT
SEEDS = (20261003, 20261004)
SIZES = (50, 100, 200)
MODELS = ('pi05', 'groot')
KNOB_SPEC = 'exp.offline_search.rounds.r15.descent.recipe:TaskDescentKnob'


def cells(model=None):
    return [f'{m}_{s}_{n}' for m in MODELS if model in (None, m) for s in ('l10', 'spatial') for n in SIZES]


def seed_dir(seed):
    return ROOT/f's{seed}'


def subset_json(d, cell):
    return Path(d)/'subsets'/f'{cell}.json'


def cs_path(d, cell):
    return Path(d)/'cs'/f'{cell}_cr.pkl'


def fit_path(d, cell, level):
    return Path(d)/'fits'/f'{cell}_{level}.pkl'


def alias():
    OUT.mkdir(parents=True, exist_ok=True)
    if not ROOT.is_symlink():
        ROOT.symlink_to(OUT, target_is_directory=True)


def write(path, blob):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp'); tmp.write_bytes(pickle.dumps(blob, protocol=4)); tmp.replace(path)


# --------------------------------------------------------------------------------------------- subsets
def subsets(seed, d):
    """r10/data.make_subsets with this seed, written to d/subsets/ (sizes 50/100/200)."""
    from exp.offline_search.rounds.r10 import data
    with patch.object(data, 'SEED', seed), patch.object(data, 'HERE', Path(d)), patch.object(data, 'SIZES', SIZES):
        data.make_subsets()


# ------------------------------------------------------------------------------------------ R10 fits
@contextmanager
def old_gate():
    """fit_recipe constructs the R8 judge (TriggerCommitJudge, noprog_span), whose constructors and unpickler raise
    since 65b1cc5f. For one fit inside this process: the pre-65b1cc5f bodies (the same code without the raise)."""
    from exp.offline_search.rounds.r03.h3_judge import judge as h3
    from exp.offline_search.rounds.r06.p2_ablations import judge as p2

    def trigger_init(self, disabled_guard="none", **kwargs):  # r06/p2_ablations/judge.py _TriggerMask.__init__
        if disabled_guard not in (*p2.BITS, "all", "none"):
            raise ValueError(f"unknown disabled_guard: {disabled_guard}")
        super(p2._TriggerMask, self).__init__(**kwargs)
        if (not self.guards or self.events or self.burst != 0 or self.monitor != "off"
                or self.policy_tail_gate != "lifecycle" or self.stuck_guard != "vision_confirmed"
                or self.progress_guard != "noprog_span" or self.memo_reset_after_miss
                or self.base.serving != "anchor_tail" or self.base.budget != 1
                or self.base.gates != "budget_only"):
            raise ValueError("P2 trigger ablation requires deployed B settings; use disabled_guard='all' for B-off")
        self.disabled_guard = disabled_guard
        self.disabled_mask = 15 if disabled_guard == "all" else p2.BITS.get(disabled_guard, 0)
        self.name = f"P2_without_{disabled_guard}__" + self.name

    def mixed_init(self, base=h3.DEFAULT_BASE, base_kwargs=None, guards=True, events="all", burst=0, ret_margin=0.1,
                   ret_n=1, ret_hold=4, disp_abs=1.0, disp_q=None, vote_thr=0.8, stuck_thr=2, lag_thr=5.0, noprog_n=3,
                   prog_eps=0.5, calib="iso", ncal=3000, m_pct=10.0, c_pct=95.0):  # r03/h3_judge/judge.py MixedJudge.__init__
        if base_kwargs is None:
            base_kwargs = dict(h3.DEFAULT_BASE_KWARGS) if base == h3.DEFAULT_BASE else {}
        if isinstance(events, str):
            events = list(h3.EVENTS) if events == "all" else ([] if events in ("none", "") else [events])
        events = tuple(sorted(set(events)))
        for e in events:
            if e not in h3.EVENTS:
                raise ValueError(f"events must be a subset of {h3.EVENTS}, got {e!r}")
        self.guards, self.events = bool(guards), events
        self.burst, self.ret_margin = max(int(burst), 0), float(ret_margin)
        self.ret_n, self.ret_hold = max(int(ret_n), 1), max(int(ret_hold), 1)
        self.disp_abs = float(disp_abs)
        self.disp_q = None if disp_q is None else float(disp_q)
        self.vote_thr = float(vote_thr)
        self.noprog_n, self.prog_eps = max(int(noprog_n), 2), float(prog_eps)
        self._disp_pool = None
        super(h3.MixedJudge, self).__init__(base, base_kwargs, calib=calib, ncal=ncal, m_pct=m_pct, c_pct=c_pct,
                                            lag_thr=lag_thr, stuck_thr=stuck_thr, ot_still=True, anchor="self")

    with patch.object(p2._TriggerMask, '__init__', trigger_init), patch.object(h3.MixedJudge, '__init__', mixed_init), \
            patch.object(h3.MixedJudge, '__setstate__', lambda self, state: self.__dict__.update(state)):
        yield


def r10_fit(job):
    """One subset's R10 fit, written as R10Recipe(inner = fitted corrected cache); an existing file is reused."""
    d, cell = job
    out = cs_path(d, cell)
    if out.exists():
        return str(out)
    from exp.offline_search.harness import api
    from exp.offline_search.rounds.r10.data import STORE
    from exp.offline_search.rounds.r10.recipe.build import SPEC, subset_kwargs
    from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
    model, suite, _ = cell.split('_')
    kwargs = subset_kwargs(model, suite, subset_json(d, cell))
    ctx = api.Context(root=STORE, cell=f'{model}_{suite}_cache', seed=0, scratch=Path(d)/'scratch'/cell)
    method, started = R10Recipe(**kwargs), time.monotonic()
    with old_gate():
        method.fit(None, ctx)
        api.check_method_attrs(method)
    judge, base = method.inner, method.inner.base
    base.invalidate_anchor()
    method.inner, method.uses_nonlibrary_action = base, True
    method.name = 'SeedCS_corrected__' + base.name
    method.fit_info = dict(method.fit_info, guard=None, seed_split_1023=dict(
        subset=str(subset_json(d, cell)), subset_seed=read(subset_json(d, cell))['seed'],
        judge=f'{type(judge).__name__} built inside the fit only, then dropped (cache-study _cr form)'))
    fit_s = time.monotonic() - started
    write(out, dict(method=method, registered={}, spec=SPEC, kwargs=kwargs, cell=ctx.cell, fit_s=fit_s))
    print('R10_FIT', out, f'{fit_s:.0f}s', flush=True)
    return str(out)


def q3_fit(job):
    d, cell = job
    return q3.fit_cell(cell, out=d, base_fit=cs_path(d, cell))


# ----------------------------------------------------------------------------------- success filter
def audit(d, cell):
    """Kept episodes (the subset's successes) and empty tasks of one cell, from the parent's episode labels."""
    from exp.offline_search.rounds.r10.data import PARENT
    from exp.offline_search.rounds.r17.filter_failed.audit_counts import LEGACY, summarize
    model, suite, size = cell.split('_')
    spec = read(subset_json(d, cell)); parent = LEGACY/f'{model}_{suite}'/PARENT[model]
    ids = sorted(i for es in spec['episode_ids_by_task'].values() for i in es)
    return dict(cell=cell, subset_seed=spec['seed'], size=int(size), parent=str(parent), subset=str(subset_json(d, cell)),
                **summarize(read(parent/'episodes.json'), ids, id_key='episode_id', task_key='task_id'))


def cell_record(d, cell):
    record = next(c for c in read(Path(d)/'CELLS.json')['cells'] if c['cell'] == cell)
    if record['empty_tasks']:
        raise ValueError(f'{d}/{cell}: empty tasks {record["empty_tasks"]}; c10 is not a no-retrieval fallback')
    return record


# --------------------------------------------------------------------------------------- knob fits
def knob_kwargs(d, cell, level):
    """Fit and arm kwargs: the subset, every task at `level`, frozen caps; reject empty-task cells."""
    from exp.offline_search.rounds.r10.recipe.build import subset_kwargs
    model, suite, _ = cell.split('_')
    cell_record(d, cell)
    levels = {str(t): level for t in range(10)}
    period = None if level == 'off' else float(read(Path(d)/'PERIODS.json')[cell][level])
    caps = [0. if levels[str(t)] in ('off', 'c10') else period for t in range(10)]
    return dict(subset_kwargs(model, suite, subset_json(d, cell)), task_levels=levels, task_caps=caps)


def knob_blob(d, cell, level):
    """q3_port_1020.build.knob_blob (filtered) with this seed's base, Q3, periods, kwargs and kept episodes."""
    from exp.offline_search.rounds.r06.ideation_Q1.method_c.stall_only import Q3StallInner
    from exp.offline_search.rounds.r15.descent.recipe import TaskDescentKnob
    from exp.offline_search.rounds.r17.filter_failed_knob.vla import candidates
    source = pickle.loads(cs_path(d, cell).read_bytes()); r = source['method']
    lib, _ = q3.library(cell, source)
    inner = Q3StallInner(r.inner, NS(task=np.asarray(lib['task_id'])), q3.stall_model(cell, d))
    kwargs, record = knob_kwargs(d, cell, level), cell_record(d, cell)
    m = TaskDescentKnob(kwargs['library'], kwargs['episode_subset'], task_levels=kwargs['task_levels'], task_caps=kwargs['task_caps'])
    m.knob_model = cell.split('_')[0]
    m.inner, m.row_subset = inner, r.row_subset
    m.size = sum(len(v) for v in kwargs['episode_subset'].values())
    m.fit_info = dict(r.fit_info, guard='Q3 stall gate (R6 Q3 successful-library LOEO); plugin --os-judge stall_only',
                      seed_split_1023=dict(r.fit_info.get('seed_split_1023', {}), cell=cell, level=level,
                                           levels=kwargs['task_levels'], caps=kwargs['task_caps'], q3=str(q3.q3_path(cell, d)),
                                           base_fit=str(cs_path(d, cell)), empty_tasks=record['empty_tasks']))
    m.r14_fit = dict(cell=cell, variant='task_descent_q3', levels=kwargs['task_levels'], K=kwargs['task_caps'], q3=str(q3.q3_path(cell, d)))
    m.name = f'Q3Knob_{level}__{inner.name}'
    blob = dict(method=m, registered=source['registered'], spec=KNOB_SPEC, kwargs=kwargs, cell=source['cell'],
                fit_s=source.get('fit_s'))
    return candidates(blob, record['kept_episode_ids'], record['empty_tasks'])


def knob_fit(job):
    d, cell, level = job
    cell_record(d, cell)  # An existing c10-placeholder fit must not bypass the empty-task refusal.
    path = fit_path(d, cell, level)
    if not path.exists():
        write(path, knob_blob(d, cell, level))
    return str(path)


def knob_fits(d, cell_list, levels, workers=8):
    jobs = [(Path(d), c, l) for c in cell_list for l in levels]
    with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
        return list(pool.map(knob_fit, jobs))


# ------------------------------------------------------------------------------------------ actions
def prepare(seeds, workers, cell_list=None, root=None):
    """Per seed: subsets -> R10 fits -> Q3 + periods -> CELLS.json -> knob-off fits (root: ROOT by default)."""
    alias()
    dirs = {seed: Path(root)/f's{seed}' if root else seed_dir(seed) for seed in seeds}
    names = list(cell_list or cells())
    for seed, d in dirs.items():
        subsets(seed, d)
    jobs = [(d, c) for d in dirs.values() for c in names]
    with ProcessPoolExecutor(min(workers, len(jobs))) as pool:
        list(pool.map(r10_fit, jobs))
        receipts = list(pool.map(q3_fit, jobs))
    for d in dirs.values():
        q3.summarize([r for (dd, _), r in zip(jobs, receipts) if dd == d], d)
        records = [audit(d, c) for c in names]
        dump(d/'CELLS.json', dict(cells=records))
        knob_fits(d, names, ('off',), workers)
        for r in records:
            print('CELL', d.name, r['cell'], 'kept', r['kept'], 'of', r['collected'], 'empty_tasks', r['empty_tasks'], flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('action', choices=('prepare',))
    p.add_argument('--seeds', type=int, nargs='+', default=list(SEEDS)); p.add_argument('--workers', type=int, default=8)
    a = p.parse_args()
    prepare(a.seeds, a.workers)


if __name__ == '__main__':
    main()
