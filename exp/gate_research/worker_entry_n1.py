"""N1 live-validation worker entry (Stage 1b).

Same as ``examples.libero.worker_entry`` but constructs the ``LiberoEpisodeRunner``
with an N1 gate client factory instead of the default websocket client, so each
worker drives the server-side ``ClientControlledGate`` with the N1 hysteresis
decision. N1 thresholds are read from the environment (``N1_THETA_LOW`` /
``N1_THETA_HIGH`` / ``N1_J`` / ``N1_M``) set by ``run_n1_live.py``.

Named as ``WorkerSpec.worker_module="exp.gate_research.worker_entry_n1"``. It is
GPU/LIBERO-only (manual): LIBERO is imported lazily via the reused
``worker_entry._build_episode_setup`` so this module stays importable in CI, but
``main()`` requires a real environment + a running driver.
"""

from __future__ import annotations

import argparse
import socket

from openpi.conductor.worker import WorkerLoop

from exp.gate_research.n1_gate_client import make_n1_client_factory, n1_params_from_env


def main() -> None:
    ap = argparse.ArgumentParser(description="conductor LIBERO N1-gate worker")
    ap.add_argument("--worker-id", required=True)
    ap.add_argument("--server-key", required=True, help='bound server endpoint "host:port"')
    ap.add_argument("--driver-host", required=True)
    ap.add_argument("--driver-port", type=int, required=True)
    ap.add_argument("--task-suite-name", default="libero_spatial")
    ap.add_argument("--init-states-dir", default="")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    from examples.libero import main as m
    from examples.libero.episode_runner import LiberoEpisodeRunner
    from examples.libero.worker_entry import _build_episode_setup  # noqa: SLF001 - shared helper

    # Fail fast at startup if the N1 params are missing/malformed (before any
    # rollout), so a misconfigured run never silently mis-gates.
    params = n1_params_from_env()

    args = m.Args(task_suite_name=a.task_suite_name, seed=a.seed)
    runner = LiberoEpisodeRunner(
        args,
        _build_episode_setup(a, a.seed, a.init_states_dir),
        client_factory=make_n1_client_factory(params),
    )

    def connect():
        return socket.create_connection((a.driver_host, a.driver_port))

    WorkerLoop(a.worker_id, a.server_key, runner, connect=connect).run_forever()


if __name__ == "__main__":
    main()
