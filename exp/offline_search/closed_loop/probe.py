"""Verification methods for the closed-loop plugin (not retrieval proposals).

ProbeB0   B0Current (harness baseline) + extras = crc32 digests of every QueryView field the method can see. Run it
          online (plugin) and offline (harness run_jobs_inprocess over the logged episode cell): identical digests =
          the plugin handed the method exactly the QueryView the offline harness constructs for that trajectory.
ProbeHist A deterministic, history-driven toy selector (not B0): scores = -L2(rs, library rs) plus a tie-break on the
          previous executed chunk; exercises hist_* / prev_a_exec paths and returns a synthesized action (mean of the
          top-3 library actions) so the synthesized-payload serving path is covered too.
ProbeForce ProbeB0 that raises the mixed-judge flags (extras os_force_miss = 1 with os_reason = 1 + step % 7) on two
          consecutive steps of every five (step % 5 in {2, 3}) and adds ``prev_hit`` / ``n_miss_hist`` (count of 0 in
          hist_hit) so the verdict-aware bookkeeping after a MISS is digested too.

Each crc32 is split into two 16-bit halves so it survives the harness' float32 extras storage exactly.
"""
from __future__ import annotations

import zlib

import numpy as np

from exp.offline_search.harness import api, baselines, dims

DIGEST_FIELDS = ("key_v0", "key_v1", "rs", "raw_state", "hist_key_v0", "hist_key_v1", "hist_rs", "hist_raw_state",
                 "hist_a_exec", "prev_a_exec", "hist_hit")


def _crc(a) -> int:
    if a is None:
        return 0xFFFFFFFF
    a = np.ascontiguousarray(a)
    return zlib.crc32(str(a.dtype).encode() + str(a.shape).encode() + a.tobytes()) & 0xFFFFFFFF


def digests(q) -> dict:
    out = {}
    for f in DIGEST_FIELDS:
        c = _crc(getattr(q, f))
        out[f"d_{f}_hi"] = float(c >> 16)
        out[f"d_{f}_lo"] = float(c & 0xFFFF)
    ph = q.prev_hit
    out["prev_hit"] = -1.0 if ph is None else float(bool(ph))
    out["step"] = float(q.step)
    out["task_id"] = float(q.task_id)
    e = q.episode
    c = zlib.crc32(f"{e.uid}|{e.task}|{e.task_id}|{e.init}|{e.seed}".encode()) & 0xFFFFFFFF
    out["d_episode_hi"] = float(c >> 16)
    out["d_episode_lo"] = float(c & 0xFFFF)
    return out


class ProbeB0(baselines.B0Current):
    name = "probe_b0"
    family = "closed_loop_probe"

    def __init__(self):
        super().__init__(extras=True)
        self.name = "probe_b0"

    def query(self, q):
        r = super().query(q)
        ex = dict(r.extras or {})
        ex.update(digests(q))
        return api.Result(topk=r.topk, scores=r.scores, confidence=r.confidence, extras=ex)


class ProbeForce(ProbeB0):
    """ProbeB0 + the H3 judge contract flags: forced MISS on steps with step % 5 in {2, 3}."""

    name = "probe_force"

    def __init__(self, period=5, force_steps=(2, 3)):
        super().__init__()
        self.name = "probe_force"
        self.period = int(period)
        self.force_steps = tuple(int(s) for s in force_steps)

    def query(self, q):
        r = super().query(q)
        ex = dict(r.extras or {})
        forced = (q.step % self.period) in self.force_steps
        ex["os_force_miss"] = 1.0 if forced else 0.0
        if forced:
            ex["os_reason"] = float(1 + q.step % 7)
        ex["n_miss_hist"] = float(int((np.asarray(q.hist_hit) == 0).sum()))
        return api.Result(topk=r.topk, scores=r.scores, confidence=r.confidence, extras=ex)


class ProbeHist(api.Method):
    name = "probe_hist"
    tier = "T0"
    family = "closed_loop_probe"

    def fit(self, lib, ctx):
        self.rows = {t: lib.rows_of_task(t) for t in lib.tasks()}
        self.rs = np.ascontiguousarray(dims.valid_state(lib.rs, lib.model), np.float32)
        self.act = np.ascontiguousarray(lib.action, np.float32)

    def reset(self, episode):
        self.n = 0

    def query(self, q):
        r = self.rows[q.task_id]
        d = np.linalg.norm(self.rs[r] - dims.valid_state(np.asarray(q.rs, np.float32), q.model), axis=1)
        if q.prev_a_exec is not None:
            tail = dims.valid_action_chunk(q.prev_a_exec)[5:6]           # first unexecuted step of the previous chunk
            head = dims.valid_action_chunk(self.act[r])[:, 0]
            d = d + 1e-3 * np.linalg.norm(head - tail, axis=1)
        o = np.argsort(d, kind="stable")[: api.TOPK_SAVE]
        a = self.act[r[o[:3]]].mean(0).astype(np.float32)
        self.n += 1
        ex = digests(q)
        ex["n_calls"] = float(self.n)
        return api.Result(topk=r[o], scores=-d[o], confidence=float(-d[o[0]]), action=a, extras=ex)

    def bytes_per_entry(self):
        return float(dims.RS_VALID["pi05"] * 4)
