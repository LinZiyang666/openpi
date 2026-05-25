"""Automated optimal-worker-count explorer for a routed serving replica set.

Goal: for one server (a100 or jupyter, behind the connection-sticky router),
find the worker count N* that maximizes sustained throughput WITHOUT pushing the
server into backlog. Too few workers starve the GPU (low inf/s); too many cause
request pile-up (queue depth + tail latency climb while throughput plateaus or
regresses). The sweet spot is the knee of the throughput-vs-concurrency curve.

Method (optimization, not brute grid):
  1. BRACKET — geometric ramp of N. Stop when either (a) marginal throughput
     gain per added worker falls below `--knee-gain` (saturated), (b) stage3
     queue depth exceeds `--backlog-q3` (backlog onset), or (c) the client host
     load exceeds its core count (client-bound, samples no longer reflect the
     server). This brackets the knee.
  2. REFINE — golden-section search on the throughput curve between the last
     pre-knee N and the bracket end, to pinpoint the knee N.
  3. MODEL — fit the Universal Scalability Law
         X(N) = N * X1 / (1 + alpha*(N-1) + beta*N*(N-1))
     to all (N, throughput) samples. The beta (coherency/backlog) term yields a
     closed-form peak N_usl = sqrt((1 - alpha) / beta). Report both the measured
     knee and the USL optimum; recommend N* = the knee (saturated, not yet
     backlogged), cross-checked against N_usl.

Each sample: spawn N LIBERO worker processes on the client host targeting the
server's public endpoint, warm up, then measure a steady-state window via the
server's aggregated `dump_metrics` (throughput = stage1 inferences / window;
backlog = stage3 queue-depth avg + wait_max tail).

Usage:
  python exp/serving_benchmark/autotune_workers.py \
      --server-host 149.165.151.106 --server-port 8000 --label a100 \
      --client-host timan107 --max-workers 160 --out /tmp/autotune_a100.json
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import time

sys.path.insert(0, "packages/openpi-client/src")
from openpi_client import msgpack_numpy
import websockets.sync.client as _ws

CLIENT_REPO_DEFAULT = "/scratch/zixuans8/openpi"
CLIENT_PYTHON_DEFAULT = "/scratch/zixuans8/libero_sim/bin/python"


# --------------------------------------------------------------------------- #
# client-host orchestration (tether)                                          #
# --------------------------------------------------------------------------- #

def _tether(client_host: str, script: str, timeout: float = 120.0,
            *, allow_fail: bool = False) -> str:
    out = subprocess.run(
        ["tether", "exec", client_host, "--", "bash", "-lc", script],
        capture_output=True, text=True, timeout=timeout,
    )
    if out.returncode != 0:
        msg = (f"tether exec on {client_host} exited {out.returncode}: "
               f"{out.stderr.strip()[:300]}")
        if not allow_fail:
            # Fail fast: a failed spawn/measure command must NOT silently fall
            # through to measuring a stale/empty server (false worker count).
            raise RuntimeError(msg)
        print(f"WARN: {msg}", file=sys.stderr, flush=True)
    return out.stdout.strip()


def kill_workers(client_host: str) -> None:
    # Best-effort cleanup: tolerate non-zero exit (sessions may already be gone).
    _tether(client_host,
            'for n in $(tmux ls 2>/dev/null | awk -F: "{print \\$1}" | grep "^drv_"); '
            'do tmux kill-session -t $n 2>/dev/null; done; '
            'rm -f /tmp/at_worker_*.log; echo cleared',
            allow_fail=True)


def spawn_workers(client_host: str, n: int, server_host: str, server_port: int,
                  repo: str, python: str, trials: int) -> None:
    script = f"""
cd {repo} || exit 1
for N in $(seq 0 {n - 1}); do
  G=$((N % 8))
  tmux new -s drv_$N -d "cd {repo} && {python} examples/libero/main.py \
    --host {server_host} --port {server_port} \
    --task-suite-name libero_spatial --num-workers 1 \
    --num-trials-per-task {trials} --task-ids 0 \
    --cuda-visible-devices $G --seed $((42+N)) > /tmp/at_worker_$N.log 2>&1; \
    echo WX; exec bash"
done
SPAWNED=$(tmux ls 2>/dev/null | grep -c "^drv_")
echo "spawned $SPAWNED"
# Fail fast on missing tmux / failed session starts (the per-`tmux new` failures
# above are swallowed by the loop). A non-zero exit propagates through _tether
# (raises) so the autotuner never warms up + measures a stale/empty server.
if [ "$SPAWNED" -lt {n} ]; then
  echo "ERROR: only $SPAWNED/{n} tmux sessions started on {client_host}" >&2
  exit 1
fi
# A tmux session only proves a SHELL exists ("exec bash" outlives a crashed
# python). Settle, then assert the worker PROCESSES are actually alive — catches
# an import/config/runtime failure that exits python immediately.
sleep 4
# `pgrep -f` matches the FULL command line, which includes the shell running
# THIS script (its cmdline contains the worker command text). Exclude self ($$)
# so the self-match cannot mask exactly one crashed worker (+1 self / -1 dead).
ALIVE=$(pgrep -f "examples/libero/main.py" 2>/dev/null | grep -vxc "$$")
echo "alive $ALIVE"
if [ "$ALIVE" -lt {n} ]; then
  echo "ERROR: only $ALIVE/{n} worker processes alive 4s after launch (python exited early)" >&2
  exit 1
fi
"""
    print(f"    [spawn] {n} workers -> {server_host}:{server_port}: "
          f"{_tether(client_host, script, timeout=180.0)}", flush=True)


def client_load(client_host: str) -> tuple[float, int]:
    out = _tether(client_host, 'echo "$(cut -d\\  -f1 /proc/loadavg) $(nproc)"')
    try:
        load, ncpu = out.split()
        return float(load), int(ncpu)
    except Exception:
        return 0.0, 1


def _objective(sample: dict, backlog_q3: float, penalty: float) -> float:
    """Backlog-penalized throughput: reward inf/s but penalize stage3 queue
    depth above the backlog threshold, so the search prefers the
    'saturated but not piling up' operating point rather than raw peak."""
    return sample["throughput"] - penalty * max(0.0, sample["q3_avg"] - backlog_q3)


# --------------------------------------------------------------------------- #
# server measurement (aggregated dump_metrics through the router)             #
# --------------------------------------------------------------------------- #

def _dump_metrics(server_host: str, server_port: int, clear: bool) -> dict:
    c = _ws.connect(f"ws://{server_host}:{server_port}", compression=None,
                    max_size=None, open_timeout=20)
    c.recv()  # metadata greeting
    c.send(msgpack_numpy.Packer().pack(
        {"__ctrl__": "dump_metrics", "include_mem": False,
         "include_timing": False, "clear": clear}))
    r = msgpack_numpy.unpackb(c.recv())
    c.close()
    return r


def set_batch_params(server_host: str, server_port: int,
                     max_batch: int, max_wait_ms: float) -> None:
    """Hot-switch coordinator batch params on ALL replicas (router broadcasts)."""
    c = _ws.connect(f"ws://{server_host}:{server_port}", compression=None,
                    max_size=None, open_timeout=20)
    c.recv()
    c.send(msgpack_numpy.Packer().pack(
        {"__ctrl__": "set_batch_params",
         "max_batch_size": max_batch, "max_wait_ms": max_wait_ms}))
    c.recv()
    c.close()


def measure_window(server_host: str, server_port: int, window_s: float) -> dict:
    """Clear, wait `window_s`, dump. Return throughput + backlog signals."""
    _dump_metrics(server_host, server_port, clear=True)
    time.sleep(window_s)
    r = _dump_metrics(server_host, server_port, clear=False)
    batch = r.get("batch", [])
    util = r.get("util", [])
    s1 = [e for e in batch if e.get("stage") == 1]
    s3 = [e for e in batch if e.get("stage") == 3]
    if not s1:
        return {"throughput": 0.0, "q3_avg": 0.0, "wait_max": 0.0,
                "fill1": 0.0, "n_inf": 0}
    n_inf = sum(e["size"] for e in s1)
    # The metrics were cleared then dumped exactly ``window_s`` apart, so every
    # event here accumulated over that known interval. Dividing by ``window_s``
    # (not the observed min/max event-timestamp span) is both correct and robust:
    # a single clustered batch can't collapse the denominator to ~0 and explode
    # the rate to a garbage value that would then poison the bracket / USL fit.
    q3 = [x["q3"] for x in util if "q3" in x]
    wait_max = max((e.get("wait_max_ms", 0.0) for e in s3), default=0.0)
    fill1 = n_inf / max(len(s1), 1)
    return {
        "throughput": n_inf / max(window_s, 1e-6),
        "q3_avg": (sum(q3) / len(q3)) if q3 else 0.0,
        "wait_max": wait_max,
        "fill1": fill1,
        "n_inf": n_inf,
    }


# --------------------------------------------------------------------------- #
# Universal Scalability Law fit                                               #
# --------------------------------------------------------------------------- #

def fit_usl(ns: list[int], xs: list[float]) -> dict:
    """Fit X(N) = N*X1 / (1 + a*(N-1) + b*N*(N-1)). Returns params + N_opt."""
    if len(ns) < 3:
        return {}
    try:
        import numpy as np
        from scipy.optimize import curve_fit

        def usl(n, x1, a, b):
            return n * x1 / (1.0 + a * (n - 1.0) + b * n * (n - 1.0))

        n_arr = np.array(ns, dtype=float)
        x_arr = np.array(xs, dtype=float)
        x1_0 = max(x_arr[0] / max(n_arr[0], 1.0), 1e-3)
        popt, _ = curve_fit(usl, n_arr, x_arr, p0=[x1_0, 0.1, 1e-3],
                            bounds=([1e-6, 0, 0], [np.inf, 10, 1]), maxfev=20000)
        x1, a, b = popt
        n_opt = math.sqrt((1.0 - a) / b) if b > 0 and a < 1 else float("inf")
        return {"X1": float(x1), "alpha": float(a), "beta": float(b),
                "N_opt": float(n_opt),
                "X_at_opt": float(usl(n_opt, *popt)) if math.isfinite(n_opt) else None}
    except Exception as exc:
        return {"error": str(exc)}


# --------------------------------------------------------------------------- #
# adaptive search                                                             #
# --------------------------------------------------------------------------- #

def run_sample(args, n: int) -> dict:
    kill_workers(args.client_host)
    time.sleep(2)
    spawn_workers(args.client_host, n, args.server_host, args.server_port,
                  args.client_repo, args.client_python, args.trials)
    print(f"    [warmup] {args.warmup_s}s ...", flush=True)
    time.sleep(args.warmup_s)
    m = measure_window(args.server_host, args.server_port, args.window_s)
    # Sample client load at the END of the window — the spawn-burst spike in
    # load1 has decayed by now, so this reflects steady sim-stepping cost.
    load, ncpu = client_load(args.client_host)
    m["N"] = n
    m["client_load"] = load
    m["client_ncpu"] = ncpu
    m["client_saturated"] = load > 1.25 * ncpu
    print(f"  N={n:4d}  X={m['throughput']:6.2f} inf/s  q3_avg={m['q3_avg']:.1f}  "
          f"wait_max={m['wait_max']:.0f}ms  fill={m['fill1']:.1f}  "
          f"client_load={load:.0f}/{ncpu}{' SAT' if m['client_saturated'] else ''}",
          flush=True)
    return m


def wait_sweep(args, n: int) -> dict:
    """At a FIXED worker count, sweep max_wait_ms via hot-switch (no respawn).

    Coordinate-descent second axis: workers stay up; only the coordinator's
    batch-wait window changes (broadcast to all replicas), so each point costs
    just a settle + measurement window. Picks the wait that maximizes
    backlog-penalized throughput and leaves the server set to it.
    """
    kill_workers(args.client_host)
    time.sleep(2)
    spawn_workers(args.client_host, n, args.server_host, args.server_port,
                  args.client_repo, args.client_python, args.trials)
    print(f"    [warmup] {args.warmup_s}s before wait-sweep at N={n} ...", flush=True)
    time.sleep(args.warmup_s)
    results = []
    for w in args.wait_ladder:
        set_batch_params(args.server_host, args.server_port, args.max_batch, w)
        time.sleep(args.wait_settle_s)  # drain in-flight requests into new regime
        m = measure_window(args.server_host, args.server_port, args.window_s)
        m["max_wait_ms"] = w
        results.append(m)
        print(f"  wait={w:6.0f}ms  X={m['throughput']:6.2f} inf/s  q3_avg={m['q3_avg']:.1f}  "
              f"wait_max={m['wait_max']:.0f}ms  fill={m['fill1']:.1f}", flush=True)
    best = max(results, key=lambda s: _objective(s, args.backlog_q3, args.backlog_penalty))
    set_batch_params(args.server_host, args.server_port, args.max_batch, best["max_wait_ms"])
    print(f"  -> best wait = {best['max_wait_ms']:.0f}ms (X={best['throughput']:.2f}); "
          f"server left at this value", flush=True)
    return {"fixed_N": n, "wait_samples": results, "best_wait_ms": best["max_wait_ms"]}


def autotune(args) -> dict:
    samples: list[dict] = []
    # 1. geometric bracket
    ladder = [n for n in args.ladder if n <= args.max_workers]
    prev_x = 0.0
    knee_n = ladder[0]
    for n in ladder:
        m = run_sample(args, n)
        samples.append(m)
        gain = (m["throughput"] - prev_x) / max(prev_x, 1e-6)
        backlog = m["q3_avg"] > args.backlog_q3
        if m["throughput"] > prev_x:
            knee_n = n
        # stop conditions
        if prev_x > 0 and gain < args.knee_gain:
            print(f"  -> marginal gain {gain:.0%} < {args.knee_gain:.0%}: saturated near N={n}", flush=True)
            break
        if backlog:
            print(f"  -> q3_avg {m['q3_avg']:.1f} > {args.backlog_q3}: backlog onset at N={n}", flush=True)
            break
        if m["client_saturated"]:
            # Not a hard stop: if the client is the limiter, server throughput
            # plateaus and the marginal-gain test ends the bracket anyway. Just
            # flag it so the report can distinguish server- vs client-bound.
            print(f"  -> note: client load {m['client_load']:.0f}>{m['client_ncpu']} (client-bound region)", flush=True)
        prev_x = m["throughput"]

    # 2. golden-section refine between knee_n and the last sampled N
    lo = knee_n
    hi = samples[-1]["N"]
    sampled = {s["N"]: s for s in samples}
    if hi - lo >= args.refine_min_gap:
        print(f"  [refine] golden-section in [{lo}, {hi}]", flush=True)
        gr = (math.sqrt(5) - 1) / 2
        a, b = lo, hi
        for _ in range(args.refine_iters):
            c = int(round(b - gr * (b - a)))
            d = int(round(a + gr * (b - a)))
            if c == d or c <= a or d >= b:
                break
            for nn in (c, d):
                if nn not in sampled:
                    sampled[nn] = run_sample(args, nn)
                    samples.append(sampled[nn])
            # objective with light backlog penalty so the search prefers the
            # saturated-but-not-backlogged point
            oc = _objective(sampled[c], args.backlog_q3, args.backlog_penalty)
            od = _objective(sampled[d], args.backlog_q3, args.backlog_penalty)
            if oc < od:
                a = c
            else:
                b = d
        knee_n = max(sampled,
                     key=lambda k: _objective(sampled[k], args.backlog_q3, args.backlog_penalty))

    ns = sorted(sampled)
    usl = fit_usl(ns, [sampled[n]["throughput"] for n in ns])
    best = max(samples, key=lambda s: s["throughput"])
    return {
        "label": args.label,
        "server": f"{args.server_host}:{args.server_port}",
        "samples": sorted(samples, key=lambda s: s["N"]),
        "best_measured": {"N": best["N"], "throughput": best["throughput"]},
        "recommended_N": knee_n,
        "usl": usl,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--server-host", required=True)
    p.add_argument("--server-port", type=int, default=8000)
    p.add_argument("--label", required=True)
    p.add_argument("--client-host", default="timan107")
    p.add_argument("--client-repo", default=CLIENT_REPO_DEFAULT)
    p.add_argument("--client-python", default=CLIENT_PYTHON_DEFAULT)
    p.add_argument("--ladder", type=int, nargs="+",
                   default=[16, 32, 48, 64, 96, 128, 160])
    p.add_argument("--max-workers", type=int, default=160)
    p.add_argument("--trials", type=int, default=6,
                   help="--num-trials-per-task: keep workers busy through the window")
    p.add_argument("--warmup-s", type=float, default=45.0)
    p.add_argument("--window-s", type=float, default=30.0)
    p.add_argument("--knee-gain", type=float, default=0.10,
                   help="stop bracket when marginal throughput gain < this")
    p.add_argument("--backlog-q3", type=float, default=6.0,
                   help="stage3 queue-depth avg above this = backlog onset")
    p.add_argument("--backlog-penalty", type=float, default=1.0)
    p.add_argument("--refine-iters", type=int, default=3)
    p.add_argument("--refine-min-gap", type=int, default=16)
    # second axis: batch-wait window (hot-switched, coordinate descent)
    p.add_argument("--tune-wait", action="store_true",
                   help="after finding N*, sweep max_wait_ms at N* (coordinate descent)")
    p.add_argument("--fixed-workers", type=int, default=0,
                   help="skip the N-search and run ONLY the wait-sweep at this N")
    p.add_argument("--wait-ladder", type=float, nargs="+",
                   default=[10, 25, 50, 100, 150])
    p.add_argument("--max-batch", type=int, default=32)
    p.add_argument("--wait-settle-s", type=float, default=8.0)
    p.add_argument("--out", default="")
    args = p.parse_args()

    print(f"=== autotune {args.label} ({args.server_host}:{args.server_port}) ===", flush=True)
    t0 = time.time()
    try:
        if args.fixed_workers > 0:
            # wait-sweep only, at the given worker count
            result = {"label": args.label,
                      "server": f"{args.server_host}:{args.server_port}",
                      "wait_tune": wait_sweep(args, args.fixed_workers)}
        else:
            result = autotune(args)
            if args.tune_wait:
                print(f"\n=== wait-sweep at N*={result['recommended_N']} ===", flush=True)
                result["wait_tune"] = wait_sweep(args, result["recommended_N"])
    finally:
        kill_workers(args.client_host)
    result["wall_s"] = round(time.time() - t0, 1)

    print("\n=== RESULT ===", flush=True)
    if "recommended_N" in result:
        print(f"  recommended N* = {result['recommended_N']}  "
              f"(best measured X={result['best_measured']['throughput']:.2f} @ N={result['best_measured']['N']})",
              flush=True)
        u = result.get("usl") or {}
        if "N_opt" in u:
            print(f"  USL: X1={u['X1']:.2f} alpha={u['alpha']:.3f} beta={u['beta']:.5f} "
                  f"-> N_opt={u['N_opt']:.0f}", flush=True)
    if result.get("wait_tune"):
        wt = result["wait_tune"]
        print(f"  best max_wait_ms = {wt['best_wait_ms']:.0f} (at N={wt['fixed_N']})", flush=True)
    if args.out:
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
