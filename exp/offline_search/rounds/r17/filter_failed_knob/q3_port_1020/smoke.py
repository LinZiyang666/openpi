"""Minimal CPU smoke: load built fits as the plugin does and run one library trajectory through them.

Queries every even step of one failed library episode (C10 anchors; no policy, no server). The Q3
forced calls must equal fit.forced_mask for that episode (a failed episode is no template, so the
served estimate and the LOEO estimate coincide). One level fit is exercised in memory.
"""
from __future__ import annotations

import json
import pickle
import sys

import numpy as np

from exp.offline_search.harness import api
from .fit import STORE, library, load_cs, stall_model, forced_mask
from .build import fit_file, knob_blob


class LibQ:
    def __init__(self, arr, rows, i, episode, task):
        r, before = rows[i], rows[:i]
        self.step, self.task_id, self.episode = i, task, episode
        self.key_v0, self.key_v1, self.rs, self.raw_state = arr['key_v0'][r], arr['key_v1'][r], arr['rs'][r], arr['rs'][r]
        self.hist_key_v0, self.hist_key_v1, self.hist_rs = arr['key_v0'][before], arr['key_v1'][before], arr['rs'][before]
        self.hist_a_exec = arr['action'][before]
        self.prev_hit = True if i else None
        self.prev_a_exec = arr['action'][rows[i-1]] if i else None
        self.hist_hit = np.ones(i, np.int64)
        self.hist_has_vision = (np.arange(i) % 2 == 0)


def run(cell):
    source = load_cs(cell)
    lib, parent = library(cell, source)
    rows_all = np.asarray(source['method'].row_subset)
    arr = {k: np.load(parent/f'{k}.npy', mmap_mode='r') for k in ('key_v0', 'key_v1', 'rs', 'action')}
    forced, _ = forced_mask(stall_model(cell), lib)
    failed = sorted(set(np.unique(lib['episode'][~lib['success'].astype(bool)]).tolist()))
    ep = max(failed, key=lambda e: (sum(v for k, v in forced.items() if k[0] == e), -e))
    local = np.flatnonzero(lib['episode'] == ep); local = local[np.argsort(lib['step'][local])]
    rows, task = rows_all[local], int(lib['task_id'][local[0]])
    expected = [forced[(ep, int(s))] for s in lib['step'][local] if s % 2 == 0]
    out = {}
    tests = [(kind, tag, None) for kind, tag in (('original', 'off'), ('filtered', 'off'), ('cs', 'c'), ('cs', 'cr'))]
    tests.append(('original', 'ir32', knob_blob(cell, 'original', 'ir32')))
    for kind, tag, blob in tests:
        if blob is None:
            with open(fit_file(kind, cell, tag), 'rb') as f:
                blob = pickle.load(f)
        method = blob['method']
        episode = api.EpisodeView(f'A:smoke:{ep}', '', task, ep % 50, 0, 0)
        method.reset(episode)
        calls, knob, states = [], 0, []
        for i in range(0, len(rows), 2):
            res = method.query(LibQ(arr, rows, i, episode, task))
            calls.append(bool(res.extras['os_q3_call'])); states.append(int(res.extras['os_q3_state']))
            knob += int(res.extras.get('os_reason') == 73.)
        out[f'{kind}/{tag}'] = dict(type=type(method).__name__, inner=type(method.inner).__name__,
                                    base=type(method.inner.base).__name__, anchors=len(calls), q3_calls=sum(calls),
                                    knob_calls=knob, equals_library_mask=calls == expected,
                                    states={s: states.count(s) for s in sorted(set(states))})
    return dict(cell=cell, episode=ep, task=task, steps=len(rows), mask_calls=sum(expected), fits=out)


if __name__ == '__main__':
    print(json.dumps([run(c) for c in sys.argv[1:]], indent=1))
