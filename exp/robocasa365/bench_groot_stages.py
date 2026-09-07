"""W1 — GR00T N1.5 three-stage CUDA-Graph micro-benchmark (plan §3.0/W1, §5 G-M).

Answers one question: under ``mode="reduce-overhead"`` with each stage compiled
into its **own** graph, what do GR00T's three stages actually cost, and how much
of stage 3 is per-step (``b``) rather than fixed head (``a``)?

    s2act(k) = a + b*k        k = denoising steps
    warm-start ceiling = 3b / T      (NOT 0.75 * s3/T -- see plan §5 G-M)

Three modes
-----------
``--mode diagnose-stage1``
    Plan §7.4-1. Compiles the *whole* stage-1 boundary itself (never the
    production one-shot vision gate, which raises at 0.999 and would truncate
    the diagnosis before any number is produced) and reports max-abs delta,
    relative Frobenius, cosine quantiles, the worst token's norm and both
    dtypes, for ``reduce-overhead`` and ``default`` side by side. This is what
    decides whether the known 0.8716 failure is an autocast dtype problem or an
    over-strict judge.

``--mode measure``
    The G-M latency cell. Writes an **uncertified** raw record.

``--mode certify``
    Second phase. Parses an external CUDA API trace and rewrites the raw record
    with the real ``cudagraph_launch_count``. Separate on purpose: an external
    profiler only finalises its report after the measured process exits, so a
    single process cannot validate its own trace.

Why this lives in a standalone script and not in the serving path
----------------------------------------------------------------
Production ``GrootStagedRunner`` is **two** stages: ``run_stage1`` (vision) and
``run_stage2`` (LLM + the action head's whole Euler loop as one atomic
``get_action`` call). The measurement needs three. Plan D2 forbids touching
production code before the payoff gate, so the split is replicated here, and
all copied boundaries carry drift pins (``UPSTREAM_GET_ACTION_SHA256`` for the
upstream action head and ``RUN_STAGE{1,2}_SRC_SHA256`` for the two in-repo
runner methods). Stage 1 uses a benchmark-local ``index_copy`` equivalent so
its production data-dependent guard stays eager while all tensor work is one
``fullgraph=True`` graph; the initial production eager call executes that guard.

⚠ A pin only catches drift *after* transcription. It cannot prove the copy was
correct to begin with, so stage 3 parity is measured against the **real**
``action_head.get_action`` under matched noise, not against another copy of
itself.

Everything fails closed
-----------------------
A cell is VOID unless every one of these holds: stage-1/2/3 parity passes, the
GPU was idle, the host is the frozen measurement machine, the checkpoint
identity is recorded, fixed noise replays identically, freshly sampled noise
does not, no inductor cudagraph skip fired, no capture happened after warmup,
and (after ``--mode certify``) the trace shows ``cudaGraphLaunch`` inside the
measurement window. Any check that cannot be *performed* is also a VOID, never
a pass.

Run (weilandserver, idle GPU, one process per cell, serially -- never in parallel).
The ``cudaProfilerApi`` range starts after warmup and stops after measurement,
so the exported summary cannot hide a recapture among allowed warmup captures::

    PYTHONPATH=/home/weiland/gr00t_n15:/home/weiland/openpi/src:/home/weiland/openpi \\
    nsys profile --trace=cuda,nvtx --capture-range=cudaProfilerApi --capture-range-end=stop \\
      --output /tmp/groot_cg_k4_p0_r0 \\
      /home/weiland/gr00t_n15_venv/.venv/bin/python exp/robocasa365/bench_groot_stages.py \\
        --mode measure --checkpoint <ckpt> --k 4 --prompt-index 0 --proc-idx 0 \\
        --out exp/robocasa365/data/latency/groot_cg_k4_p0_r0.json
    nsys stats --report cuda_api_sum,nvtx_sum --format csv \\
      /tmp/groot_cg_k4_p0_r0.nsys-rep > /tmp/groot_cg_k4_p0_r0.trace.csv
    python exp/robocasa365/bench_groot_stages.py --mode certify \\
      --out <same json> --cuda-trace /tmp/groot_cg_k4_p0_r0.trace.csv
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime
import hashlib
import inspect
import json
import logging
import os
import pathlib
import platform
import random
import socket
import subprocess
import time
from collections.abc import Iterator
from typing import Any, Callable

import numpy as np
import torch

# --- pinned sources ---------------------------------------------------------
# Bump deliberately after re-reading the source and re-running parity, never to
# "make it pass": a drift means this benchmark is measuring code that no longer
# matches what the copies below assume.
UPSTREAM_GET_ACTION_SHA256 = (
    "8a8e6cf7ec63e2a335559990c4ab62bbb81e487d82ea4a969f452a93e0dbdd69"
)
UPSTREAM_ACTION_HEAD_REL = "gr00t/model/action_head/flow_matching_action_head.py"
RUN_STAGE2_SRC_SHA256 = (
    "cc81f0fa91c38e385b1c577cb86686d149f85b82c303f5a88ba32ce1bf4786a1"
)
RUN_STAGE1_SRC_SHA256 = (
    "8cb20edfc06a93ae21ec90cde7f117159fbff7a44e7c216d2461f18fa5f44b98"
)

# Step count is part of the identity: the same GR00T checkpoint runs 4 steps on
# RoboCasa and 8 on LIBERO (serve_groot_libero.py:60), so a single shared id
# would let two incompatible libraries validate against each other (plan D4).
SCHEDULE_ID = "groot_n15_k4_v1"
EXPECTED_HOST = "weilandserver"
COMPILE_MODE = "reduce-overhead"

# Plan §5 G-M: every prompt is its own graph shape, so each is measured and
# reported separately. A single-shape number cannot be extrapolated -- variable
# N is precisely the CUDA-Graph risk this benchmark exists to size.
PROMPTS = [
    "Pick the turmeric from the counter and place it in the cabinet.",
    "Pick the corn from the plate and place it in the pan.",
    "Pick the tongs from the drawer and place it on the counter.",
    "Pick the apple from the sink and place it on the plate located on the counter.",
    "Place the toasted item on a plate.",
]

# The G-M record schema, frozen in plan §5. Kept as a module-level constant so
# the contract is inspectable and every written cell can be checked against it,
# rather than living implicitly in a dict literal halfway down a function.
MEASURE_RECORD_FIELDS: tuple[str, ...] = (
    "schedule_id",
    "k",
    "prompt_sha256",
    "trace_marker",
    "prompt_shape_n",
    "mode",
    "warmup",
    "iters",
    "proc_idx",
    "stage1_ms",
    "stage2_llm_ms",
    "stage3_ms",
    "total_ms",
    "parity_stage1",
    "parity_stage2",
    "parity_stage3_copy_vs_upstream",
    "parity_stage3_eager_vs_compiled",
    "parity_worst",
    "fixed_noise_replay_equal",
    "sampled_noise_distinct",
    "cuda_trace_path",
    "cudagraph_launch_count",
    "expected_cudagraph_launch_count",
    "capture_calls_after_warmup",
    "compile_count",
    "cudagraph_skips",
    "inductor_perf_hints",
    "busy_gpu_override",
    "gpu_name",
    "gpu_uuid",
    "torch",
    "driver",
    "ckpt_sha256",
    "upstream_get_action_sha256",
    "git_commit",
    "host",
    "ts",
)


def assert_record_schema(record: dict) -> None:
    """Fail closed if a cell is missing any frozen G-M field."""
    missing = [f for f in MEASURE_RECORD_FIELDS if f not in record]
    if missing:
        raise SystemExit(f"record is missing frozen G-M fields: {missing}")


# Production stage-1 gate threshold, reused so the diagnosis is expressed on the
# same scale as the failure it is diagnosing (staged.py:432).
STAGE1_GATE_COS = 0.999
STAGE3_PARITY_REL_TOL = 1e-2


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def sha256_file(path: pathlib.Path) -> str:
    """Return the SHA-256 of a file, streamed so large checkpoints stay cheap."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    """Return the SHA-256 of a UTF-8 string."""
    return hashlib.sha256(text.encode()).hexdigest()


def checkpoint_identity(checkpoint: pathlib.Path) -> str:
    """Content identity of a checkpoint directory.

    Hashes the sorted ``(relative path, file digest)`` pairs of every weight and
    config file, so the value survives a re-download but changes if any tensor
    does. Recording only ``config.json`` -- an earlier version of this script --
    would have called two different weight sets the same checkpoint.
    """
    parts = []
    for pattern in ("*.json", "*.safetensors", "*.bin"):
        for f in sorted(checkpoint.rglob(pattern)):
            parts.append(f"{f.relative_to(checkpoint)}:{sha256_file(f)}")
    if not parts:
        raise SystemExit(
            f"no weight/config files under {checkpoint}; cannot record identity"
        )
    return sha256_text("\n".join(parts))


def _run(*argv: str) -> subprocess.CompletedProcess:
    """Run a command, capturing output. Raises on a missing binary."""
    return subprocess.run(argv, capture_output=True, text=True, timeout=30, check=False)


def gpu_provenance(device_index: int) -> dict[str, str]:
    """Return name/UUID/driver for one GPU, failing closed if unobtainable.

    ⚠ The card in this host was physically replaced on 2026-08-26, after the
    authoritative pi0.5 ledger was measured. ``gpu_uuid`` plus the torch version
    are the only fields that can reveal that confound when the two tables are
    read side by side, so an unreadable value voids the cell rather than
    defaulting to blank.
    """
    proc = _run(
        "nvidia-smi",
        f"--id={device_index}",
        "--query-gpu=name,uuid,driver_version",
        "--format=csv,noheader",
    )
    if proc.returncode != 0 or not proc.stdout.strip():
        raise SystemExit(
            f"cannot read GPU provenance for index {device_index} "
            f"(rc={proc.returncode}): {proc.stderr.strip()!r}. Failing closed."
        )
    parts = [p.strip() for p in proc.stdout.strip().split(",")]
    if len(parts) < 3 or not parts[1]:
        raise SystemExit(f"unexpected nvidia-smi provenance output: {proc.stdout!r}")
    return {"gpu_name": parts[0], "gpu_uuid": parts[1], "driver": parts[2]}


def assert_idle_gpu(device_index: int, own_pid: int) -> None:
    """Fail closed unless this process is the only compute app on the GPU.

    Owner constraint (plan §2 D3): the measurement runs on an otherwise idle
    card. batch=1 inference is launch-bound, so a co-tenant competing for CPU or
    SMs does not merely add noise -- it changes the number being measured. An
    ``nvidia-smi`` that cannot be run is treated as "unknown", which is a
    failure, not an idle card.
    """
    proc = _run(
        "nvidia-smi",
        f"--id={device_index}",
        "--query-compute-apps=pid,used_memory,process_name",
        "--format=csv,noheader",
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"cannot verify GPU {device_index} is idle (rc={proc.returncode}): "
            f"{proc.stderr.strip()!r}. Failing closed rather than assuming idle."
        )
    others = []
    for row in proc.stdout.splitlines():
        parts = [p.strip() for p in row.split(",")]
        if parts and parts[0].isdigit() and int(parts[0]) != own_pid:
            others.append(
                {"pid": parts[0], "used": parts[1] if len(parts) > 1 else "?"}
            )
    if others:
        raise SystemExit(
            f"GPU {device_index} is not idle -- refusing to measure. Other compute apps: "
            f"{others}. Wait for the window; never evict another session's work (plan §3.0.1)."
        )


def assert_host() -> str:
    """Fail closed unless running on the owner-frozen measurement machine."""
    host = socket.gethostname()
    if host.split(".", 1)[0] != EXPECTED_HOST:
        raise SystemExit(
            f"host {host!r} is not the frozen measurement machine {EXPECTED_HOST!r}. "
            "Plan §2 D3 pins the latency measurement to one host; numbers from "
            "anywhere else are not comparable to the ledger."
        )
    return host


@contextlib.contextmanager
def measurement_gpu_guard(device_index: int, *, allow_busy: bool) -> Iterator[None]:
    """Check the idle-GPU invariant immediately before and after measurement."""
    if not allow_busy:
        assert_idle_gpu(device_index, os.getpid())
    try:
        yield
    finally:
        if not allow_busy:
            assert_idle_gpu(device_index, os.getpid())


@contextlib.contextmanager
def cuda_profiler_range() -> Iterator[None]:
    """Bracket only the measurement loop for Nsight ``cudaProfilerApi`` capture."""
    cudart = torch.cuda.cudart()
    start_rc = int(cudart.cudaProfilerStart())
    if start_rc != 0:
        raise SystemExit(f"cudaProfilerStart failed with CUDA error {start_rc}")
    try:
        yield
    finally:
        stop_rc = int(cudart.cudaProfilerStop())
        if stop_rc != 0:
            raise SystemExit(f"cudaProfilerStop failed with CUDA error {stop_rc}")


def trace_marker(record: dict[str, Any]) -> str:
    """Return the cell identity that must also appear in the profiler export."""
    fields = {
        key: record.get(key)
        for key in (
            "schedule_id",
            "k",
            "prompt_sha256",
            "proc_idx",
            "mode",
            "warmup",
            "iters",
            "seed",
            "expected_cudagraph_launch_count",
            "ckpt_sha256",
            "gpu_uuid",
            "ts",
        )
    }
    identity = json.dumps(
        fields, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    )
    return f"openpi_w1_{sha256_text(identity)[:24]}"


@contextlib.contextmanager
def nvtx_measurement_range(marker: str) -> Iterator[None]:
    """Put a cell-identity marker in the trace around the measurement window."""
    torch.cuda.nvtx.range_push(marker)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def atomic_write_json(path: pathlib.Path, payload: dict[str, Any]) -> None:
    """Write JSON via a temp file + rename, so an interrupt leaves no half record."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=1))
    tmp.replace(path)


def positive_int(value: str) -> int:
    """argparse type: a strictly positive integer (zero iterations -> NaN stats)."""
    n = int(value)
    if n <= 0:
        raise argparse.ArgumentTypeError(f"expected a positive integer, got {n}")
    return n


# ---------------------------------------------------------------------------
# numerical comparison
# ---------------------------------------------------------------------------


def tensor_stats(compiled: torch.Tensor, eager: torch.Tensor) -> dict[str, float | str]:
    """Full divergence profile between a compiled and an eager tensor.

    The production stage-1 gate thresholds the *minimum* per-token cosine at
    0.999 and failed at 0.8716 (staged.py:412-438). A low-norm token's cosine is
    extremely sensitive to a fixed absolute perturbation, so the minimum alone
    cannot separate "miscompiled" from "one small token". Reporting the
    quantiles, the worst token's norm, the absolute delta and both dtypes is
    what makes that call decidable -- which is the whole point of plan §7.4-1.
    """
    import torch.nn.functional as F  # noqa: PLC0415

    x = compiled.detach().float().reshape(-1, compiled.shape[-1])
    y = eager.detach().float().reshape(-1, eager.shape[-1])
    cos = F.cosine_similarity(x, y, dim=-1)
    worst = int(torch.argmin(cos))
    q = torch.quantile(cos.cpu(), torch.tensor([0.0, 0.01, 0.05, 0.5]))
    return {
        "cos_min": float(q[0]),
        "cos_p01": float(q[1]),
        "cos_p05": float(q[2]),
        "cos_p50": float(q[3]),
        "worst_token_norm": float(y[worst].norm()),
        "max_abs_delta": float((x - y).abs().max()),
        "rel_frobenius": float((x - y).norm() / y.norm().clamp_min(1e-12)),
        "dtype_compiled": str(compiled.dtype),
        "dtype_eager": str(eager.dtype),
    }


def rel_err(a: torch.Tensor, b: torch.Tensor) -> float:
    """Max absolute difference normalised by the reference's peak magnitude."""
    a32, b32 = a.detach().float(), b.detach().float()
    return float((a32 - b32).abs().max() / b32.abs().max().clamp_min(1e-12))


# ---------------------------------------------------------------------------
# stage 3: the upstream denoise loop, with the noise hoisted out
# ---------------------------------------------------------------------------


def denoise_step(
    action_head: Any,
    vl: torch.Tensor,
    state_features: torch.Tensor,
    embodiment_id: torch.Tensor,
    actions: torch.Tensor,
    timesteps_tensor: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """Run one GR00T flow-matching step; this is the Stage-3 compiled graph."""
    action_features = action_head.action_encoder(
        actions, timesteps_tensor, embodiment_id
    )
    if action_head.config.add_pos_embed:
        pos_ids = torch.arange(
            action_features.shape[1], dtype=torch.long, device=vl.device
        )
        action_features = action_features + action_head.position_embedding(
            pos_ids
        ).unsqueeze(0)
    future_tokens = action_head.future_tokens.weight.unsqueeze(0).expand(
        vl.shape[0], -1, -1
    )
    sa_embs = torch.cat((state_features, future_tokens, action_features), dim=1)
    model_output = action_head.model(
        hidden_states=sa_embs, encoder_hidden_states=vl, timestep=timesteps_tensor
    )
    pred = action_head.action_decoder(model_output, embodiment_id)
    return actions + dt * pred[:, -action_head.action_horizon :]


def denoise_loop(
    action_head: Any,
    backbone_output: Any,
    action_input: Any,
    *,
    noise: torch.Tensor,
    num_steps: int,
    step_fn: Callable[..., torch.Tensor] = denoise_step,
) -> torch.Tensor:
    """Copy upstream ``get_action`` with external noise and a pluggable one-step body.

    Upstream draws ``torch.randn`` inside the function body
    (flow_matching_action_head.py:364-368). Hoisting it out is what lets parity
    separate "the compiled output is numerically wrong" from "the two runs drew
    different noise", and it matches how pi0.5 already does it
    (``policy.py:122-128``).

    Everything else is upstream's, character for character: the **ascending**
    ``t_cont = t/N``, the integer bucket discretisation, the position embedding,
    the ``future_tokens`` expansion and the ``+dt*v`` Euler update.
    """
    processed = action_head.process_backbone_output(backbone_output)
    vl = processed.backbone_features
    embodiment_id = action_input.embodiment_id
    state_features = action_head.state_encoder(action_input.state, embodiment_id)

    batch_size = vl.shape[0]
    actions = noise
    dt = 1.0 / num_steps
    for t in range(num_steps):
        t_cont = t / float(num_steps)  # ascending: 0, 1/N, 2/N, ...
        t_discretized = int(t_cont * action_head.num_timestep_buckets)
        timesteps_tensor = torch.full(
            size=(batch_size,), fill_value=t_discretized, device=vl.device
        )
        # A reduce-overhead graph returns a static output buffer. Clone after
        # every replay because the next denoise step consumes this value.
        actions = step_fn(
            action_head,
            vl,
            state_features,
            embodiment_id,
            actions,
            timesteps_tensor,
            dt,
        ).clone()
    return actions


def upstream_reference_action(
    action_head, backbone_output, action_input, *, noise: torch.Tensor
) -> torch.Tensor:
    """Run the **real** ``action_head.get_action`` with our fixed noise.

    This is the independent reference for stage-3 parity. Comparing the local
    copy's eager run against the local copy's compiled run is a closed loop: a
    mis-transcription makes both wrong in the same way and parity passes anyway.
    Upstream samples its own noise internally, so ``torch.randn`` is redirected
    for exactly one call to hand back the same tensor the copy was given.
    """
    real_randn = torch.randn

    def fake_randn(*args, **kwargs):
        # Upstream calls torch.randn(size=..., dtype=..., device=...). Serve the
        # pinned noise once, then restore, so nothing else in the call is
        # affected if upstream ever draws again.
        nonlocal patched
        if not patched:
            patched = True
            want = kwargs.get("size", args[0] if args else None)
            got = tuple(noise.shape)
            if want is not None and tuple(want) != got:
                raise SystemExit(
                    f"upstream asked for noise of shape {tuple(want)} but the pinned "
                    f"noise is {got}; the copy and upstream disagree on geometry."
                )
            return noise.to(
                device=kwargs.get("device", noise.device),
                dtype=kwargs.get("dtype", noise.dtype),
            )
        return real_randn(*args, **kwargs)

    patched = False
    torch.randn = fake_randn
    try:
        out = action_head.get_action(backbone_output, action_input)
    finally:
        torch.randn = real_randn
    if not patched:
        raise SystemExit(
            "upstream get_action never called torch.randn; the noise-injection "
            "assumption behind this parity check no longer holds."
        )
    return out["action_pred"]


# ---------------------------------------------------------------------------
# inductor bookkeeping
# ---------------------------------------------------------------------------


def inductor_counters() -> dict[str, int]:
    """Snapshot inductor/dynamo integer counters.

    ⚠ Corroborating only. Plan §5 G-M is explicit that neither a latency
    distribution nor a counter written by the measured process itself proves
    graph replay; the binding evidence is an external CUDA API trace.
    """
    out: dict[str, int] = {}
    try:
        from torch._dynamo.utils import counters  # noqa: PLC0415

        for group, entries in counters.items():
            for key, value in entries.items():
                if isinstance(value, int):
                    out[f"{group}.{key}"] = value
    except Exception:  # noqa: BLE001 - absence is recorded, never fatal here
        pass
    return out


def cudagraph_skips(counters: dict[str, int]) -> int:
    """Total inductor cudagraph skips seen so far (any skip voids the cell)."""
    return sum(
        v
        for k, v in counters.items()
        if "cudagraph_skip" in k or "skipped_cudagraph" in k
    )


def unique_graph_count(counters: dict[str, int]) -> int:
    """Return Dynamo's exact unique-graph counter, or fail if it is unavailable."""
    key = "stats.unique_graphs"
    if key not in counters:
        raise SystemExit(
            f"inductor counter {key!r} is unavailable; cannot certify graph count"
        )
    return counters[key]


@contextlib.contextmanager
def capture_inductor_perf_hints() -> Iterator[list[str]]:
    """Collect Inductor ``perf_hints`` log records for the cell JSON."""
    hints: list[str] = []

    class _HintHandler(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            if "perf_hints" in record.name:
                hints.append(self.format(record))

    handler = _HintHandler()
    torch._logging.set_logs(perf_hints=True)  # noqa: SLF001 - public torch logging API
    # set_logs() rebuilds torch's handlers, so attach ours only afterwards and
    # at the non-propagating Inductor parent used by artifact loggers.
    torch_logger = logging.getLogger("torch._inductor")
    torch_logger.addHandler(handler)
    try:
        yield hints
    finally:
        torch_logger.removeHandler(handler)


# ---------------------------------------------------------------------------
# trace certification (phase 2)
# ---------------------------------------------------------------------------


def parse_cuda_trace(
    path: pathlib.Path, *, expected_marker: str = ""
) -> dict[str, int]:
    """Parse the CUDA-API table from a possibly multi-report Nsight CSV export."""
    if not path.exists():
        raise SystemExit(f"CUDA trace not found: {path}")
    text = path.read_text(errors="replace")
    if not text.strip():
        raise SystemExit(f"CUDA trace is empty: {path}. An empty file is not evidence.")
    rows = list(csv.reader(text.splitlines()))
    if expected_marker and not any(
        value.strip() == expected_marker for row in rows for value in row
    ):
        raise SystemExit(
            f"CUDA trace {path} does not contain cell marker {expected_marker!r}"
        )
    call_labels = ("Num Calls", "Instances", "Calls")
    header_index = None
    name_index = None
    count_index = None
    for index, row in enumerate(rows):
        labels = [value.strip() for value in row]
        if "Name" not in labels:
            continue
        count_label = next((label for label in call_labels if label in labels), None)
        if count_label is not None:
            header_index = index
            name_index = labels.index("Name")
            count_index = labels.index(count_label)
            break
    if header_index is None:
        raise SystemExit(
            f"CUDA trace {path} is not an nsys cuda_api_sum CSV: missing Name/Num Calls header"
        )

    assert name_index is not None and count_index is not None
    launches = 0
    captures = 0
    parsed_rows = 0
    for row in rows[header_index + 1 :]:
        # ``nsys stats --report cuda_api_sum,nvtx_sum`` emits two complete CSV
        # tables. Stop at the separator/new header instead of interpreting the
        # NVTX ``Instances`` column using CUDA-table positions.
        if not row or not any(value.strip() for value in row):
            if parsed_rows:
                break
            continue
        if row[0].strip() == "Time (%)":
            break
        if len(row) <= max(name_index, count_index):
            if parsed_rows:
                break
            continue
        name = row[name_index].strip()
        count_text = row[count_index].strip()
        if not name or not count_text:
            continue
        try:
            count = int(count_text.replace(",", ""))
        except ValueError as exc:
            raise SystemExit(
                f"invalid CUDA API call count {count_text!r} for {name!r} in {path}"
            ) from exc
        parsed_rows += 1
        if "cudaGraphLaunch" in name:
            launches += count
        if any(
            api in name
            for api in (
                "cudaStreamBeginCapture",
                "cudaStreamEndCapture",
                "cudaGraphInstantiate",
            )
        ):
            captures += count
    if not parsed_rows:
        raise SystemExit(f"CUDA trace {path} contains no parseable API summary rows")
    return {"cudagraph_launch_count": launches, "graph_capture_calls": captures}


def certify_cell(record_path: pathlib.Path, trace_path: pathlib.Path) -> dict[str, Any]:
    """Phase 2: fold real trace evidence into a raw cell and re-decide validity.

    Separate from the measurement process on purpose -- an external profiler
    finalises its report only after the measured process exits, so a cell can
    never certify itself.
    """
    record = json.loads(record_path.read_text())
    assert_record_schema(record)
    if record.get("certified") is not False:
        raise SystemExit(f"cell is not an uncertified measurement: {record_path}")
    uncertified = "uncertified: no CUDA trace parsed"
    if uncertified not in record.get("void_reasons", []):
        raise SystemExit(f"cell is not a raw uncertified W1 record: {record_path}")
    if str(record.get("host", "")).split(".", 1)[0] != EXPECTED_HOST:
        raise SystemExit(
            f"raw cell host {record.get('host')!r} is not the frozen host {EXPECTED_HOST!r}"
        )
    marker = record.get("trace_marker")
    if not isinstance(marker, str) or not marker.startswith("openpi_w1_"):
        raise SystemExit(f"raw cell has no valid trace marker: {marker!r}")
    derived_marker = trace_marker(record)
    if marker != derived_marker:
        raise SystemExit(
            f"raw cell trace marker does not match its identity: "
            f"recorded={marker!r}, derived={derived_marker!r}"
        )
    expected_launches = record.get("expected_cudagraph_launch_count")
    if (
        not isinstance(expected_launches, int)
        or isinstance(expected_launches, bool)
        or expected_launches <= 0
    ):
        raise SystemExit(
            "raw cell has no positive integer expected_cudagraph_launch_count: "
            f"{expected_launches!r}"
        )
    expected_from_schedule = record.get("iters") * (2 + record.get("k"))
    if expected_launches != expected_from_schedule:
        raise SystemExit(
            "raw cell replay expectation disagrees with its schedule: "
            f"recorded={expected_launches}, derived={expected_from_schedule}"
        )

    counts = parse_cuda_trace(trace_path, expected_marker=marker)
    record["cuda_trace_path"] = str(trace_path)
    record["cuda_trace_sha256"] = sha256_file(trace_path)
    record["cudagraph_launch_count"] = counts["cudagraph_launch_count"]
    record["capture_calls_after_warmup"] = counts["graph_capture_calls"]

    reasons = [r for r in record.get("void_reasons", []) if r != uncertified]
    if counts["cudagraph_launch_count"] <= 0:
        reasons.append("trace shows no cudaGraphLaunch -- graphs were never replayed")
    if counts["cudagraph_launch_count"] != expected_launches:
        reasons.append(
            "cudaGraphLaunch count mismatch: "
            f"expected {expected_launches}, got {counts['cudagraph_launch_count']}"
        )
    if counts["graph_capture_calls"]:
        reasons.append(
            "measurement trace contains graph capture APIs: "
            f"count={counts['graph_capture_calls']}"
        )
    record["void_reasons"] = reasons
    record["valid"] = not reasons
    record["certified"] = True
    atomic_write_json(record_path, record)
    return record


# ---------------------------------------------------------------------------
# production-faithful input
# ---------------------------------------------------------------------------


def build_production_input(policy: Any, checkpoint: pathlib.Path, prompt: str) -> Any:
    """Reproduce the serving path's observation shaping, all four steps of it.

    Production is ``build_groot_observation`` (validates and adds the ``T=1``
    axis) then the collector's batch unsqueeze (``B=1``) then a numpy coercion
    then ``apply_transforms`` (groot_policy_adapter.py:206,
    groot_cache_collector.py:118-126). Handing the raw wire observation straight
    to ``apply_transforms`` -- which an earlier version of this script did --
    measures a shape the server never sees, and ``run_stage1`` asserts ``B=1``
    anyway.
    """
    from exp.robocasa365 import groot_keys  # noqa: PLC0415
    from openpi.cache.groot.interceptor import _is_batched, _unsqueeze_values  # noqa: PLC0415
    from exp.robocasa365.groot_policy_adapter import build_groot_observation  # noqa: PLC0415
    from exp.robocasa365.serve_groot_n15 import _dummy_observation  # noqa: PLC0415

    obs = _dummy_observation(checkpoint)
    for key in groot_keys.LANGUAGE_KEYS:
        obs[key] = prompt  # prompt length is the graph shape under test
    groot_obs = build_groot_observation(obs)
    if not _is_batched(groot_obs):
        groot_obs = _unsqueeze_values(groot_obs)
    groot_obs = {
        k: (v if isinstance(v, np.ndarray) else np.array(v))
        for k, v in groot_obs.items()
    }
    return policy.apply_transforms(groot_obs)


def load_policy(checkpoint: pathlib.Path, *, device: str) -> Any:
    """Build the served policy exactly as ``serve_groot_n15.main()`` does (:551-558)."""
    from gr00t.model.policy import Gr00tPolicy  # noqa: PLC0415

    from exp.robocasa365.groot_data_config import RoboCasa365DataConfig  # noqa: PLC0415
    from exp.robocasa365.serve_groot_n15 import EMBODIMENT_TAG  # noqa: PLC0415

    data_config = RoboCasa365DataConfig()
    return Gr00tPolicy(
        model_path=str(checkpoint),
        embodiment_tag=EMBODIMENT_TAG,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        device=device,
    )


def assert_source_pins(gr00t_root: pathlib.Path, runner_cls: type) -> None:
    """Fail closed if either copied source has drifted since transcription."""
    upstream = gr00t_root / UPSTREAM_ACTION_HEAD_REL
    got = sha256_file(upstream)
    if got != UPSTREAM_GET_ACTION_SHA256:
        raise SystemExit(
            f"upstream action head drifted: {upstream}\n  expected {UPSTREAM_GET_ACTION_SHA256}\n"
            f"  got      {got}\nRe-read get_action, update denoise_loop(), re-run parity, "
            "and only then bump the pin."
        )
    stage1_sha = sha256_text(inspect.getsource(runner_cls.run_stage1))
    if stage1_sha != RUN_STAGE1_SRC_SHA256:
        raise SystemExit(
            f"GrootStagedRunner.run_stage1 drifted:\n  expected {RUN_STAGE1_SRC_SHA256}\n"
            f"  got      {stage1_sha}\nThe fullgraph stage-1 copy may be stale."
        )
    stage2_sha = sha256_text(inspect.getsource(runner_cls.run_stage2))
    if stage2_sha != RUN_STAGE2_SRC_SHA256:
        raise SystemExit(
            f"GrootStagedRunner.run_stage2 drifted:\n  expected {RUN_STAGE2_SRC_SHA256}\n"
            f"  got      {stage2_sha}\nThe stage-2 copy in this script may be stale."
        )


# ---------------------------------------------------------------------------
# timing
# ---------------------------------------------------------------------------


def timed(fn: Callable[[], Any]) -> tuple[Any, float]:
    """Wall clock around a synchronised call, matching the authoritative contract.

    ``policy.py:101-131`` times the pi0.5 stages with ``time.monotonic()`` and a
    ``torch.cuda.synchronize()`` per stage. CUDA events measure the GPU
    timeline, and at batch=1 these models are launch-bound (GPU util 6-15%), so
    an event-based number would systematically omit the dominant term. Matching
    the ledger's contract is what makes the two tables readable side by side.
    """
    torch.cuda.synchronize()
    t0 = time.monotonic()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.monotonic() - t0) * 1e3


# ---------------------------------------------------------------------------
# modes
# ---------------------------------------------------------------------------


def _stage_callables(
    runner: Any, image_positions: torch.Tensor
) -> tuple[Callable, Callable]:
    """Return ``(stage1_full, stage2_llm)`` as plain callables ready to compile.

    ``stage1_full`` is the *whole* stage-1 boundary, not just the vision tower:
    the production runner compiles only ``extract_feature`` and leaves the
    embedding lookup and the scatter eager, which would make "three stages,
    three graphs" untrue for the boundary this benchmark reports.
    ``stage2_llm`` is the LLM half of ``run_stage2`` (staged.py:444-458).
    """
    model = runner._model  # noqa: SLF001 - the bench measures the split it reaches into
    eagle = runner._eagle  # noqa: SLF001
    backbone = runner._backbone  # noqa: SLF001

    def stage1_full(normalized: dict) -> tuple[Any, ...]:
        # Benchmark-local tensor equivalent of GrootStagedRunner.run_stage1.
        # The eager production call runs once first and owns all fail-loud
        # schema/count guards. Keeping those data-dependent Tensor.item guards
        # out of this callable lets fullgraph=True prove there is no eager tail.
        backbone_inputs, action_inputs = model.prepare_input(normalized)
        eagle_input = {
            key.removeprefix("eagle_"): value
            for key, value in backbone_inputs.items()
            if key.startswith("eagle_")
        }
        eagle_input.pop("image_sizes", None)
        input_ids = eagle_input["input_ids"]
        input_embeds = eagle.language_model.get_input_embeddings()(input_ids)
        vit_embeds = eagle.extract_feature(eagle_input["pixel_values"])
        b, n, width = input_embeds.shape
        flat = input_embeds.reshape(b * n, width)
        old_image_values = flat.index_select(0, image_positions)
        replacement = old_image_values * 0.0 + vit_embeds.reshape(-1, width)
        flat = flat.index_copy(0, image_positions, replacement)
        return (
            flat.reshape(b, n, width),
            eagle_input["attention_mask"],
            input_ids == eagle.image_token_index,
            action_inputs,
        )

    def stage2_llm(
        input_embeds: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        outputs = eagle.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            position_ids=None,
            past_key_values=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=True,
        )
        return backbone.eagle_linear(outputs.hidden_states[backbone.select_layer])

    return stage1_full, stage2_llm


def compile_stages(runner: Any, image_positions: torch.Tensor) -> dict[str, Any]:
    """Compile all three stage boundaries into three separate graphs.

    Returned keyed by name so the set of compiled boundaries is inspectable
    without running anything: the owner's constraint is that stage 1, stage 2
    and stage 3 each get their **own** graph, and "one big graph" would be
    invisible if the compiled callables only existed as locals.
    """
    stage1_full, stage2_llm = _stage_callables(runner, image_positions)
    return {
        "compiled_stage1": torch.compile(
            stage1_full, mode=COMPILE_MODE, dynamic=False, fullgraph=True
        ),
        "compiled_stage2_llm": torch.compile(
            stage2_llm, mode=COMPILE_MODE, dynamic=False, fullgraph=True
        ),
        "compiled_denoise_step": torch.compile(
            denoise_step, mode=COMPILE_MODE, dynamic=False, fullgraph=True
        ),
    }


def _materialize_stage1(values: tuple[Any, ...], output_type: type) -> Any:
    """Wrap a compiled tensor tuple in the production stage-output type."""
    return output_type(
        input_embeds=values[0],
        attention_mask=values[1],
        image_token_mask=values[2],
        action_inputs=values[3],
    )


def _values_equal(left: Any, right: Any) -> bool:
    """Compare tensor or scalar metadata without weakening tensor equality."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        return bool(torch.equal(left, right))
    return bool(left == right)


def stage1_parity(compiled: Any, eager: Any) -> dict[str, Any]:
    """Compare every Stage-1 value consumed by cache keys or later stages."""
    stats: dict[str, Any] = tensor_stats(compiled.input_embeds, eager.input_embeds)
    stats.update(
        {
            "attention_mask_equal": bool(
                torch.equal(compiled.attention_mask, eager.attention_mask)
            ),
            "image_token_mask_equal": bool(
                torch.equal(compiled.image_token_mask, eager.image_token_mask)
            ),
            "state_rel_err": rel_err(compiled.state, eager.state),
            "state_mask_equal": bool(
                torch.equal(compiled.state_mask, eager.state_mask)
            ),
            "embodiment_id_equal": _values_equal(
                compiled.action_inputs.embodiment_id,
                eager.action_inputs.embodiment_id,
            ),
        }
    )
    return stats


def build_measure_record(
    base: dict[str, Any], result: dict[str, Any], forced_void: list[str]
) -> dict[str, Any]:
    """Assemble and validate one G-M cell. Pure, so the schema is unit-testable."""
    record = {**base, **result}
    record["void_reasons"] = list(forced_void) + list(result["void_reasons"])
    record["valid"] = not record["void_reasons"]
    assert_record_schema(record)
    return record


def run_diagnose_stage1(
    args: argparse.Namespace, runner: Any, normalized: dict
) -> dict[str, Any]:
    """Plan §7.4-1: profile stage-1 compiled-vs-eager divergence, both modes.

    Deliberately does **not** go through the production gate: that gate raises
    below 0.999 and would abort before printing a single diagnostic number,
    which is exactly why the 0.8716 failure was never explained.
    """
    out: dict[str, Any] = {}
    with runner.session():
        eager = runner.run_stage1(normalized)
        image_positions = torch.nonzero(
            eager.image_token_mask.reshape(-1), as_tuple=False
        ).flatten()
        stage1_full, _ = _stage_callables(runner, image_positions)
        for mode in ("reduce-overhead", "default"):
            kwargs = {} if mode == "default" else {"mode": mode}
            compiled_fn = torch.compile(
                stage1_full, dynamic=False, fullgraph=True, **kwargs
            )
            got = _materialize_stage1(compiled_fn(normalized), type(eager))
            out[mode] = stage1_parity(got, eager)
            out[mode]["passes_production_gate"] = (
                out[mode]["cos_min"] >= STAGE1_GATE_COS
                and out[mode]["state_rel_err"] <= STAGE3_PARITY_REL_TOL
                and out[mode]["attention_mask_equal"]
                and out[mode]["image_token_mask_equal"]
                and out[mode]["state_mask_equal"]
                and out[mode]["embodiment_id_equal"]
            )
    return out


def run_measure(
    args: argparse.Namespace,
    runner: Any,
    policy: Any,
    normalized: dict,
    device: torch.device,
    measurement_marker: str,
) -> dict[str, Any]:
    """The G-M latency cell: three separately compiled graphs, parity, RNG, timings."""
    from transformers.feature_extraction_utils import BatchFeature  # noqa: PLC0415

    action_head = policy.model.action_head

    def wrap(features: torch.Tensor, mask: torch.Tensor) -> Any:
        # Rebuilt every call: process_backbone_output writes the normalised
        # features back in place, so a reused mapping gets vlln applied twice.
        return BatchFeature(
            data={"backbone_features": features, "backbone_attention_mask": mask}
        )

    void: list[str] = []
    counters_before = inductor_counters()
    unique_before = counters_before.get("stats.unique_graphs", 0)
    skips_before = cudagraph_skips(counters_before)

    with capture_inductor_perf_hints() as perf_hints, runner.session():
        horizon = action_head.config.action_horizon
        adim = action_head.config.action_dim
        generator = torch.Generator(device=device).manual_seed(args.seed)
        fixed_noise = torch.randn(1, horizon, adim, generator=generator, device=device)

        # The production eager call owns all data-dependent guards. The fixed
        # image-token positions then let the benchmark-local equivalent use an
        # index_copy and remain one full graph instead of breaking at Tensor.item().
        s1_eager = runner.run_stage1(normalized)
        image_positions = torch.nonzero(
            s1_eager.image_token_mask.reshape(-1), as_tuple=False
        ).flatten()
        _, stage2_llm = _stage_callables(runner, image_positions)
        stages = compile_stages(runner, image_positions)
        compiled_stage1 = stages["compiled_stage1"]
        compiled_stage2_llm = stages["compiled_stage2_llm"]
        compiled_denoise_step = stages["compiled_denoise_step"]

        def call_stage1() -> Any:
            return _materialize_stage1(compiled_stage1(normalized), type(s1_eager))

        def call_stage3(
            stage1: Any, features: torch.Tensor, noise: torch.Tensor
        ) -> torch.Tensor:
            return denoise_loop(
                action_head,
                wrap(features, stage1.attention_mask),
                stage1.action_inputs,
                noise=noise,
                num_steps=action_head.num_inference_timesteps,
                step_fn=compiled_denoise_step,
            )

        f_eager = stage2_llm(s1_eager.input_embeds, s1_eager.attention_mask)
        torch.compiler.cudagraph_mark_step_begin()
        s1_comp = call_stage1()
        parity_s1 = stage1_parity(s1_comp, s1_eager)

        # Isolated Stage-2 parity: both sides consume the same eager Stage-1.
        f_comp_isolated = compiled_stage2_llm(
            s1_eager.input_embeds, s1_eager.attention_mask
        ).clone()
        parity_s2 = tensor_stats(f_comp_isolated, f_eager)

        # Build the separate end-to-end compiled chain only after isolated parity.
        torch.compiler.cudagraph_mark_step_begin()
        s1_chain = call_stage1()
        f_chain = compiled_stage2_llm(
            s1_chain.input_embeds, s1_chain.attention_mask
        ).clone()

        noise_for_head = fixed_noise.to(dtype=f_eager.dtype)
        upstream_act = upstream_reference_action(
            action_head,
            wrap(f_eager.clone(), s1_eager.attention_mask),
            s1_eager.action_inputs,
            noise=noise_for_head,
        )
        copy_eager = denoise_loop(
            action_head,
            wrap(f_eager.clone(), s1_eager.attention_mask),
            s1_eager.action_inputs,
            noise=noise_for_head,
            num_steps=action_head.num_inference_timesteps,
        )
        # Isolated Stage-3 parity: both sides consume the same eager Stage-2.
        torch.compiler.cudagraph_mark_step_begin()
        copy_comp = call_stage3(s1_eager, f_eager.clone(), noise_for_head)
        torch.compiler.cudagraph_mark_step_begin()
        chain_comp = call_stage3(s1_chain, f_chain.clone(), noise_for_head)
        parity_upstream = rel_err(copy_eager, upstream_act)
        parity_s3 = rel_err(copy_comp, copy_eager)
        parity_chain = rel_err(chain_comp, upstream_act)

        # RNG: fixed input must repeat; two fresh inputs must not collapse.
        torch.compiler.cudagraph_mark_step_begin()
        a1 = call_stage3(s1_chain, f_chain.clone(), noise_for_head)
        torch.compiler.cudagraph_mark_step_begin()
        a2 = call_stage3(s1_chain, f_chain.clone(), noise_for_head)
        fixed_noise_replay_equal = bool(torch.equal(a1, a2))
        n1 = torch.randn(1, horizon, adim, device=device, dtype=f_chain.dtype)
        n2 = torch.randn(1, horizon, adim, device=device, dtype=f_chain.dtype)
        torch.compiler.cudagraph_mark_step_begin()
        b1 = call_stage3(s1_chain, f_chain.clone(), n1)
        torch.compiler.cudagraph_mark_step_begin()
        b2 = call_stage3(s1_chain, f_chain.clone(), n2)
        sampled_noise_distinct = not bool(torch.equal(b1, b2))

        for _ in range(args.warmup):
            torch.compiler.cudagraph_mark_step_begin()
            s1 = call_stage1()
            features = compiled_stage2_llm(s1.input_embeds, s1.attention_mask).clone()
            call_stage3(s1, features, noise_for_head)
        torch.cuda.synchronize()
        counters_after_warmup = inductor_counters()

        s1_ms: list[float] = []
        s2_ms: list[float] = []
        s3_ms: list[float] = []
        tot_ms: list[float] = []
        with (
            measurement_gpu_guard(args.device_index, allow_busy=args.allow_busy_gpu),
            cuda_profiler_range(),
            nvtx_measurement_range(measurement_marker),
        ):
            for _ in range(args.iters):
                torch.compiler.cudagraph_mark_step_begin()
                t0 = time.monotonic()
                st1, m1 = timed(call_stage1)
                features, m2 = timed(
                    lambda: compiled_stage2_llm(
                        st1.input_embeds, st1.attention_mask
                    ).clone()
                )
                _, m3 = timed(lambda: call_stage3(st1, features, noise_for_head))
                torch.cuda.synchronize()
                s1_ms.append(m1)
                s2_ms.append(m2)
                s3_ms.append(m3)
                tot_ms.append((time.monotonic() - t0) * 1e3)

        counters_end = inductor_counters()

    compile_count = unique_graph_count(counters_end) - unique_before
    compiles_after_warmup = unique_graph_count(counters_end) - unique_graph_count(
        counters_after_warmup
    )
    skips = cudagraph_skips(counters_end) - skips_before

    if parity_s1["cos_min"] < STAGE1_GATE_COS:
        void.append(f"stage1 parity cos_min={parity_s1['cos_min']:.6f}")
    for field in (
        "attention_mask_equal",
        "image_token_mask_equal",
        "state_mask_equal",
        "embodiment_id_equal",
    ):
        if not parity_s1[field]:
            void.append(f"stage1 parity {field}=false")
    if parity_s1["state_rel_err"] > STAGE3_PARITY_REL_TOL:
        void.append(f"stage1 state rel_err={parity_s1['state_rel_err']:.3e}")
    if parity_s2["cos_min"] < STAGE1_GATE_COS:
        void.append(f"stage2 parity cos_min={parity_s2['cos_min']:.6f}")
    if parity_upstream > STAGE3_PARITY_REL_TOL:
        void.append(f"stage3 copy-vs-upstream rel_err={parity_upstream:.3e}")
    if parity_s3 > STAGE3_PARITY_REL_TOL:
        void.append(f"stage3 eager-vs-compiled rel_err={parity_s3:.3e}")
    if parity_chain > STAGE3_PARITY_REL_TOL:
        void.append(f"compiled chain-vs-upstream rel_err={parity_chain:.3e}")
    if not fixed_noise_replay_equal:
        void.append(
            "fixed noise replayed to a different action (static buffers clobbered)"
        )
    if not sampled_noise_distinct:
        void.append(
            "fresh noise produced identical actions (RNG captured into the graph)"
        )
    if skips:
        void.append(f"cudagraph_skips={skips}")
    if compile_count != 3:
        void.append(f"unique compiled graphs={compile_count}, expected 3")
    if compiles_after_warmup:
        void.append(f"new_compiles_after_warmup={compiles_after_warmup}")
    void.append("uncertified: no CUDA trace parsed")  # cleared only by --mode certify

    return {
        "stage1_ms": s1_ms,
        "stage2_llm_ms": s2_ms,
        "stage3_ms": s3_ms,
        "total_ms": tot_ms,
        "parity_stage1": parity_s1,
        "parity_stage2": parity_s2,
        "parity_stage3_copy_vs_upstream": parity_upstream,
        "parity_stage3_eager_vs_compiled": parity_s3,
        "parity_compiled_chain_vs_upstream": parity_chain,
        "parity_worst": max(
            1.0 - parity_s1["cos_min"],
            parity_s1["state_rel_err"],
            1.0 - parity_s2["cos_min"],
            parity_upstream,
            parity_s3,
            parity_chain,
        ),
        "fixed_noise_replay_equal": fixed_noise_replay_equal,
        "sampled_noise_distinct": sampled_noise_distinct,
        "capture_calls_after_warmup": None,
        "expected_cudagraph_launch_count": args.iters * (2 + args.k),
        "compile_count": compile_count,
        "cudagraph_skips": skips,
        "inductor_counters": counters_end,
        "inductor_perf_hints": list(perf_hints),
        "prompt_shape_n": int(s1_eager.input_embeds.shape[1]),
        "void_reasons": void,
    }


def main() -> None:
    """CLI entry point for the three modes (measure / diagnose-stage1 / certify)."""
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--mode", default="measure", choices=("measure", "diagnose-stage1", "certify")
    )
    ap.add_argument("--checkpoint", default="")
    ap.add_argument("--k", type=int, default=4, choices=(1, 2, 3, 4))
    ap.add_argument("--prompt-index", type=int, default=0, choices=range(len(PROMPTS)))
    ap.add_argument("--proc-idx", type=int, default=0)
    ap.add_argument("--warmup", type=positive_int, default=30)
    ap.add_argument("--iters", type=positive_int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device-index", type=int, default=0)
    ap.add_argument("--out", required=True)
    ap.add_argument("--gr00t-root", default="/home/weiland/gr00t_n15")
    ap.add_argument("--cuda-trace", default="")
    ap.add_argument(
        "--allow-busy-gpu",
        action="store_true",
        help="diagnostics only; the cell is permanently VOID",
    )
    args = ap.parse_args()

    out_path = pathlib.Path(args.out)

    if args.mode == "certify":
        if not args.cuda_trace:
            raise SystemExit("--mode certify requires --cuda-trace")
        record = certify_cell(out_path, pathlib.Path(args.cuda_trace))
        print(
            f"[bench] certified={record['certified']} valid={record['valid']} "
            f"cudaGraphLaunch={record['cudagraph_launch_count']}",
            flush=True,
        )
        raise SystemExit(0 if record["valid"] else 1)

    if not args.checkpoint:
        raise SystemExit("--checkpoint is required for measure / diagnose-stage1")

    host = assert_host()
    forced_void: list[str] = []
    if args.allow_busy_gpu:
        forced_void.append("ran with --allow-busy-gpu (diagnostic only, never valid)")
    else:
        assert_idle_gpu(args.device_index, os.getpid())
    gpu = gpu_provenance(args.device_index)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.set_device(args.device_index)
    device = torch.device(f"cuda:{args.device_index}")

    os.environ.setdefault(
        "TORCHINDUCTOR_CACHE_DIR", os.path.expanduser("~/.cache/openpi_inductor")
    )

    from openpi.cache.groot.staged import GrootStagedRunner  # noqa: PLC0415

    checkpoint = pathlib.Path(args.checkpoint)
    assert_source_pins(pathlib.Path(args.gr00t_root), GrootStagedRunner)

    policy = load_policy(checkpoint, device=f"cuda:{args.device_index}")
    policy.model.action_head.num_inference_timesteps = args.k
    # compile_vision=False: this script owns the whole stage-1 boundary and its
    # own diagnosis; the production one-shot gate would raise before any number.
    runner = GrootStagedRunner(policy.model, compile_vision=False)
    prompt = PROMPTS[args.prompt_index]
    normalized = build_production_input(policy, checkpoint, prompt)

    record = {
        "mode_run": args.mode,
        "schedule_id": SCHEDULE_ID,
        "k": args.k,
        "prompt_index": args.prompt_index,
        "prompt_sha256": sha256_text(prompt),
        "mode": COMPILE_MODE,
        "warmup": args.warmup,
        "iters": args.iters,
        "proc_idx": args.proc_idx,
        "seed": args.seed,
        "device_index": args.device_index,
        "torch": torch.__version__,
        "ckpt_sha256": checkpoint_identity(checkpoint),
        "upstream_get_action_sha256": UPSTREAM_GET_ACTION_SHA256,
        "run_stage1_src_sha256": RUN_STAGE1_SRC_SHA256,
        "run_stage2_src_sha256": RUN_STAGE2_SRC_SHA256,
        "git_commit": _run("git", "rev-parse", "HEAD").stdout.strip(),
        "host": host,
        "python": platform.python_version(),
        "ts": datetime.datetime.now().astimezone().isoformat(timespec="microseconds"),
        "busy_gpu_override": bool(args.allow_busy_gpu),
        "cuda_trace_path": "",
        "cudagraph_launch_count": None,
        "expected_cudagraph_launch_count": args.iters * (2 + args.k),
        "certified": False,
        **gpu,
    }
    record["trace_marker"] = trace_marker(record)

    if args.mode == "diagnose-stage1":
        record["stage1_diagnosis"] = run_diagnose_stage1(args, runner, normalized)
        record["void_reasons"] = forced_void
        record["valid"] = not forced_void
        atomic_write_json(out_path, record)
        for mode, stats in record["stage1_diagnosis"].items():
            print(
                f"[diag] {mode}: cos_min={stats['cos_min']:.6f} "
                f"p01={stats['cos_p01']:.6f} max|Δ|={stats['max_abs_delta']:.3e} "
                f"relF={stats['rel_frobenius']:.3e} worst_norm={stats['worst_token_norm']:.3f} "
                f"gate={'PASS' if stats['passes_production_gate'] else 'FAIL'}",
                flush=True,
            )
        print(f"[diag] wrote {out_path}", flush=True)
        return

    result = run_measure(
        args, runner, policy, normalized, device, record["trace_marker"]
    )
    record = build_measure_record(record, result, forced_void)
    atomic_write_json(out_path, record)

    print(
        f"[bench] k={args.k} prompt={args.prompt_index} N={record['prompt_shape_n']} "
        f"proc={args.proc_idx}",
        flush=True,
    )
    print(
        f"[bench] stage1 {np.median(result['stage1_ms']):7.3f} | "
        f"stage2_llm {np.median(result['stage2_llm_ms']):7.3f} | "
        f"stage3 {np.median(result['stage3_ms']):7.3f} | "
        f"total {np.median(result['total_ms']):7.3f}  (ms, median)",
        flush=True,
    )
    print(
        f"[bench] parity_worst={record['parity_worst']:.3e} "
        f"upstream_rel={result['parity_stage3_copy_vs_upstream']:.3e} "
        f"fixed_replay_equal={result['fixed_noise_replay_equal']} "
        f"sampled_distinct={result['sampled_noise_distinct']}",
        flush=True,
    )
    print(f"[bench] VOID: {record['void_reasons']}", flush=True)
    print(
        f"[bench] wrote {out_path} -- run --mode certify with the CUDA trace next",
        flush=True,
    )


if __name__ == "__main__":
    main()
