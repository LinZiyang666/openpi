"""Teacher-only Pi0.5 server with the denoise step count pinned to k.

The production server (``scripts/serve_policy.py``) deliberately has no
step-count flag, and the flag could not be threaded through even if it had
one: ``Policy.infer`` takes the staged path for the PyTorch model and calls
``PI0Pytorch._stage3_action_expert`` with its default of ten steps, ignoring
``sample_kwargs``. The step sweep needs the *same* server at k = 1..N without
touching that entry point, so this wrapper -- the sibling of
``exp/robocasa365/serve_groot_n15_ksweep.py`` -- pins ``num_steps`` on the
model class in this interpreter, then hands control to the production
``main`` with the remaining arguments. Handshake, transforms, metadata and
ports are the production code path.

Two things are enforced rather than assumed:

*   ``--replicas`` beyond 1 is refused. The supervisor spawns replica children
    with ``multiprocessing`` ``spawn``, which re-imports ``scripts.serve_policy``
    in a fresh interpreter where the patch never ran; a child would silently
    serve ten steps. Run one process per port instead.
*   ``--cache_config`` is refused: a real cache stack would answer decisions
    from a library. The bare ``--cache`` flag is allowed and is the
    recommended way to serve many clients: it installs the interceptor
    *without* an orchestrator (no library, no retrieval, no writes, every
    decision a MISS) purely so that stage 1/2/3 go through the
    ``BatchingCoordinator`` and are batched across connections, the way the
    RIT-Pareto servers ran. That path reaches stage 3 through
    ``run_stage3``, which is pinned here too, so both serving modes run
    exactly k Euler steps from Gaussian noise.

Without ``--cache`` the concurrent server hands every connection the same
bare ``Policy``, and ``Policy.infer`` is then serialised behind one lock (as
the GR00T LIBERO server does for its teacher-only factory) so the model
forward is one request at a time rather than interleaved across threads.
With ``--cache`` the coordinator owns the GPU work and no lock is taken: it
would only serialise the per-connection interceptors ahead of the batching
queue and defeat the point.

The pinned value is printed once at start (``KSWEEP num_steps=<k>``) and once
on the first call of each stage-3 entry point (``KSWEEP stage3 first call``
for the bare path, ``KSWEEP run_stage3 first call`` for the batched path), so
the server log carries the identity of the served loop and the evidence that
the patched path is the one being run. It is also stamped into the server
metadata as ``nfe_num_steps`` -- the GR00T LIBERO server reports
``denoising_steps`` the same way -- so a client can check which loop it is
talking to before it launches a run (``probe_metadata.py``).

Usage (weilandserver, main venv, repo root on PYTHONPATH)::

    python -m exp.nfe_baseline.serve_pi05_ksweep --denoising-steps 3 \\
        --cache --port 23150 policy:checkpoint --policy.config pi05_libero \\
        --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch
"""

from __future__ import annotations

import argparse
import sys
import threading

REFUSED_FLAGS = ("--cache_config", "--cache-config")


def _parse(argv: list[str]) -> tuple[int, list[str], bool]:
    """``(k, remaining argv, batched)``; ``batched`` is whether ``--cache`` is present."""
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--denoising-steps", type=int, required=True)
    args, rest = ap.parse_known_args(argv)
    k = args.denoising_steps
    if k < 1:
        raise SystemExit(f"--denoising-steps must be >= 1, got {k}")
    batched = False
    for i, tok in enumerate(rest):
        head = tok.split("=", 1)[0]
        if head in REFUSED_FLAGS:
            raise SystemExit(f"{head} is refused: this server is the policy alone")
        if head == "--cache":
            batched = True
        if head == "--replicas":
            val = (
                tok.split("=", 1)[1]
                if "=" in tok
                else (rest[i + 1] if i + 1 < len(rest) else "")
            )
            if val.strip() != "1":
                raise SystemExit(
                    "--replicas must be 1: spawned replica children re-import "
                    "scripts.serve_policy unpatched and would serve ten steps"
                )
    return k, rest, batched


def pin_num_steps(k: int) -> None:
    """Pin ``num_steps`` to ``k`` on both stage-3 entry points of ``PI0Pytorch``."""
    from openpi.models_pytorch import pi0_pytorch

    cls = pi0_pytorch.PI0Pytorch
    orig_expert = cls._stage3_action_expert
    orig_stage3 = cls.run_stage3
    announced = threading.Event()
    announced3 = threading.Event()

    def _expert(self, state, prefix_pad_masks, past_key_values, noise, num_steps=10):
        if not announced.is_set():
            announced.set()
            print(f"KSWEEP stage3 first call num_steps={k}", flush=True)
        return orig_expert(self, state, prefix_pad_masks, past_key_values, noise, k)

    def _run_stage3(
        self,
        stage2,
        *,
        noise=None,
        num_steps=10,
        return_intermediates=False,
        save_timesteps=(0.7, 0.5, 0.3),
    ):
        if not announced3.is_set():
            announced3.set()
            print(f"KSWEEP run_stage3 first call num_steps={k}", flush=True)
        return orig_stage3(
            self,
            stage2,
            noise=noise,
            num_steps=k,
            return_intermediates=return_intermediates,
            save_timesteps=save_timesteps,
        )

    cls._stage3_action_expert = _expert
    cls.run_stage3 = _run_stage3


def lock_infer() -> None:
    """Serialise ``Policy.infer`` behind one process-wide lock."""
    from openpi.policies import policy as _policy

    lock = threading.Lock()
    orig_infer = _policy.Policy.infer

    def _infer(self, obs, *a, **kw):
        with lock:
            return orig_infer(self, obs, *a, **kw)

    _policy.Policy.infer = _infer


def stamp_metadata(k: int) -> None:
    """Add ``nfe_num_steps`` to every ``WebsocketPolicyServer``'s handshake metadata."""
    from openpi.serving import websocket_policy_server as wps

    orig_init = wps.WebsocketPolicyServer.__init__

    def _init(self, *a, **kw):
        md = dict(kw.get("metadata") or {})
        md["nfe_num_steps"] = k
        kw["metadata"] = md
        return orig_init(self, *a, **kw)

    wps.WebsocketPolicyServer.__init__ = _init


def main(argv: list[str] | None = None) -> None:
    """Parse ``--denoising-steps``, patch the model class, run the production server."""
    k, rest, batched = _parse(sys.argv[1:] if argv is None else argv)
    pin_num_steps(k)
    if not batched:
        lock_infer()
    stamp_metadata(k)
    mode = "batched-interceptor" if batched else "bare-policy-locked"
    print(f"KSWEEP num_steps={k} mode={mode}", flush=True)

    import tyro

    from scripts import serve_policy

    sys.argv = [sys.argv[0], *rest]
    serve_policy.main(tyro.cli(serve_policy.Args))


if __name__ == "__main__":
    main()
