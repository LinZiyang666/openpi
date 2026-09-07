"""Teacher-only GR00T server with the denoise step count overridden (plan W3 / G-A1).

The production server (``serve_groot_n15.py``) deliberately has no step-count
flag: the RoboCasa checkpoint's baked-in value is what every experiment on
this line has served. The step-sensitivity screen needs the *same* server at
k = 1..N without touching that entrypoint, so this wrapper substitutes a
``Gr00tPolicy`` subclass that injects ``denoising_steps`` and then hands
control to the production ``main``. Everything else -- handshake, adapter,
metadata, ports -- is the production code path.

Usage (weilandserver, GR00T venv)::

    python exp/robocasa365/serve_groot_n15_ksweep.py --denoising-steps 2 --port 23160 [serve_groot_n15 args...]

The effective ``action_head.num_inference_timesteps`` is asserted after
construction and printed as ``KSWEEP num_inference_timesteps=<k>`` so the
driver log carries the identity of the served loop.
"""

from __future__ import annotations

import argparse
import sys


def main() -> None:
    """Parse ``--denoising-steps``, patch the policy class, run the production server."""
    ap = argparse.ArgumentParser(description=__doc__, add_help=False)
    ap.add_argument("--denoising-steps", type=int, required=True)
    args, rest = ap.parse_known_args()
    k = args.denoising_steps
    if k < 1:
        raise SystemExit(f"--denoising-steps must be >= 1, got {k}")

    import gr00t.model.policy as policy_module

    base_cls = policy_module.Gr00tPolicy

    class KSweepPolicy(base_cls):  # type: ignore[misc,valid-type]
        """Gr00tPolicy that always runs ``k`` Euler steps."""

        def __init__(self, *a, **kw):
            kw["denoising_steps"] = k
            super().__init__(*a, **kw)
            live = int(self.model.action_head.num_inference_timesteps)
            if live != k:
                raise RuntimeError(
                    f"requested {k} denoising steps but the head runs {live}"
                )
            print(f"KSWEEP num_inference_timesteps={live}", flush=True)

    policy_module.Gr00tPolicy = KSweepPolicy

    from exp.robocasa365 import serve_groot_n15

    sys.argv = [sys.argv[0], *rest]
    serve_groot_n15.main()


if __name__ == "__main__":
    main()
