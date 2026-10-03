"""Print coordinator-only GPU server and recorded-image replay commands.

This command only prints shell-quoted argv; it never launches a subprocess,
opens a socket, loads a model, or changes CUDA visibility in this process.
"""
import argparse
import json
import shlex
from pathlib import Path

REPO = Path("/home/weiland/projects/openpi")
GROOT = Path("/home/weiland/projects/openpi_ext/third_party/gr00t_n15")


def commands(arm, port, log_root, debug):
    if type(port) is not int or not 1024 <= port <= 65535 or 23100 <= port <= 23199:
        raise ValueError("use a coordinator-selected port outside 23100-23199")
    model, log = arm["model"], Path(log_root)
    if model not in ("pi05", "groot"):
        raise ValueError("unsupported model")
    env = dict(OMP_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1", CUDA_VISIBLE_DEVICES="0",
               PYTHONDONTWRITEBYTECODE="1", PYTHONPATH=str(REPO) + ":" + str(REPO / "src"),
               BATCHING_MAX_BATCH_SIZE="1", HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    env.update(arm.get("server_env", {}))
    py = REPO / ".venv/bin/python"
    args = ["--os-method", arm["method"], "--os-kwargs", json.dumps(arm["kwargs"], sort_keys=True),
            "--os-cell", arm["cell"], "--os-log-dir", str(log), "--os-tag", "r8_gpu_replay",
            "--os-seed", "0", "--os-log-inputs", *arm["plugin_args"]]
    if debug:
        args += ["--os-debug-dir", str(log / "debug" / ("server_" + str(port))),
                 "--os-debug-config", json.dumps(dict(campaign="r8_gpu_replay", writer_queue_bytes=512 * 1024**2))]
    if model == "pi05":
        args += ["--port", str(port), "--replicas", "1", "--cache-config", arm["yaml"],
                 "policy:checkpoint", "--policy.config", "pi05_libero", "--policy.dir",
                 "/home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch"]
    else:
        py = Path("/home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python")
        env["PYTHONPATH"] = ":".join(str(p) for p in (GROOT, GROOT / "examples/Libero", REPO, REPO / "src", REPO / "packages/openpi-client/src"))
        args += ["--checkpoint", "/data/ckpt/n15_libero_10" if arm["suite_short"] == "l10" else "/data/ckpt/n15_libero_spatial",
                 "--port", str(port), "--denoising-steps", "8", "--concurrent", "--allow-dynamic-bundles", "--cache-config", arm["yaml"]]
    server = ["taskset", "-c", "30-33,74-77", "env", *[k + "=" + v for k, v in env.items()],
              str(py), str(REPO / ("exp/offline_search/closed_loop/serve_" + model + ".py")), *args]
    client = ["taskset", "-c", "30-33,74-77", "env", "OMP_NUM_THREADS=1", "OPENBLAS_NUM_THREADS=1", "MKL_NUM_THREADS=1",
              "CUDA_VISIBLE_DEVICES=", "PYTHONDONTWRITEBYTECODE=1", "PYTHONPATH=.:src:packages/openpi-client/src",
              str(REPO / ".venv/bin/python"), "-m", "exp.offline_search.closed_loop.replay_client", "--port", str(port),
              "--cell", arm["cell"], "--root", "/home/weiland/trace_runs/offline_search_store", "--episodes", "20",
              "--tag", "r8_gpu_replay", "--log-dir", str(log), "--out", str(log / "replay_report.json")]
    return dict(server=server, replay=client)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", type=Path, default=Path("/tmp/r8_S5/smoke/arms.json"))
    ap.add_argument("--arm", required=True)
    ap.add_argument("--port", type=int, default=23240)
    ap.add_argument("--log-root", type=Path, required=True)
    ap.add_argument("--mode", choices=("off", "on"), required=True)
    a = ap.parse_args(argv)
    arm = next(r for r in json.loads(a.arms.read_text()) if r["arm"] == a.arm)
    result = commands(arm, a.port, a.log_root, a.mode == "on")
    print("# Coordinator only: run the server in one terminal, the replay in another.")
    print(shlex.join(result["server"]))
    print(shlex.join(result["replay"]))


if __name__ == "__main__":
    main()
