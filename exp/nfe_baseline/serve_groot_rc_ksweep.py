"""RoboCasa365 GR00T teacher-only server at k denoising steps, with the step count in its handshake.

``exp/robocasa365/serve_groot_n15_ksweep.py`` (the G-A1 step-sensitivity
wrapper) already injects ``denoising_steps`` into ``Gr00tPolicy`` and asserts
``action_head.num_inference_timesteps`` after construction, but the RoboCasa
server's handshake metadata carries no step key, so a client could not check
which loop it is about to evaluate. This wrapper adds exactly that: it stamps
``nfe_num_steps`` into every ``WebsocketPolicyServer`` handshake (the same
patch the Pi0.5 wrapper uses) and then hands the unchanged argv to the G-A1
wrapper, which hands it to the production server. Nothing about the model,
the adapter or the serving stack changes.

Usage (h100 / weilandserver, GR00T venv, gr00t root + repo on PYTHONPATH)::

    python -m exp.nfe_baseline.serve_groot_rc_ksweep --denoising-steps 2 \\
        --port 23230 --concurrent --checkpoint <target-posttrained checkpoint>
"""

from __future__ import annotations

import sys


def main() -> None:
    """Stamp ``nfe_num_steps`` from ``--denoising-steps``, then run the G-A1 wrapper."""
    argv = sys.argv[1:]
    k = None
    for i, tok in enumerate(argv):
        if tok == "--denoising-steps" and i + 1 < len(argv):
            k = int(argv[i + 1])
        elif tok.startswith("--denoising-steps="):
            k = int(tok.split("=", 1)[1])
    if k is None or k < 1:
        raise SystemExit("--denoising-steps <k>=1> is required")

    from exp.nfe_baseline.serve_pi05_ksweep import stamp_metadata

    stamp_metadata(k)
    print(f"KSWEEP num_steps={k} mode=groot-rc-teacher-only", flush=True)

    from exp.robocasa365 import serve_groot_n15_ksweep

    serve_groot_n15_ksweep.main()


if __name__ == "__main__":
    main()
