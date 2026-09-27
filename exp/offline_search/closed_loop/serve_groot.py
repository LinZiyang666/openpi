"""GR00T N1.5 LIBERO pure-cache server whose CP1 retrieval is an offline-harness Method (see plugin.py / README.md).

Runs in the GR00T island venv with the same PYTHONPATH as exp/trace_dual/ops/start_groot.sh:

    PYTHONPATH=$G:$G/examples/Libero:$R:$R/src:$R/packages/openpi-client/src taskset -c <cpus> \
      /home/weiland/projects/openpi_ext/envs/gr00t_n15_venv/.venv/bin/python exp/offline_search/closed_loop/serve_groot.py \
        --os-method exp.offline_search.harness.baselines:B0Current --os-cell groot_spatial_cache \
        --os-log-dir <dir> --os-tag <arm>_<port> \
        --checkpoint /data/ckpt/n15_libero_spatial --port 23111 --denoising-steps 8 --concurrent \
        --allow-dynamic-bundles --cache-config <arm yaml> [--stage1-only]

Every ``--os-*`` flag is the plugin's; the rest goes to exp/libero_groot/serve_groot_libero.py unchanged.
Refused: no --concurrent, --trace-out, --rit-shadow-out, --loto-log-out.
"""
from __future__ import annotations

import logging
import pathlib
import sys

REPO = pathlib.Path(__file__).resolve().parents[3]
for _p in (str(REPO), str(REPO / "src")):
    if _p not in sys.path:
        sys.path.insert(0, _p)


def main() -> None:
    logging.basicConfig(level=logging.INFO, force=True)
    from exp.offline_search.closed_loop import plugin

    opts, rest = plugin.parse_cli(sys.argv[1:])
    if "--concurrent" not in rest:
        raise SystemExit("osplug: --concurrent is required (the per-connection factory is what gets wrapped)")
    for bad in ("--trace-out", "--trace-build-cache", "--rit-shadow-out", "--loto-log-out"):
        if any(a == bad or a.startswith(bad + "=") for a in rest):
            raise SystemExit(f"osplug: {bad} is not supported with the plugin")
    if not any(a == "--cache-config" or a.startswith("--cache-config=") for a in rest):
        raise SystemExit("osplug: --cache-config <yaml> is required (the served library and key builder)")
    import argparse
    from exp.offline_search.closed_loop import stage_overrides as cost
    cp = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    cp.add_argument("--cache-config", required=True)
    cp.add_argument("--denoising-steps", type=int, default=8)
    parsed, _ = cp.parse_known_args(rest)
    steps = cost.miss_steps_from_yaml(parsed.cache_config, "groot", parsed.denoising_steps)
    cost.install_startup_hook(plugin, stage1_mode="full", miss_steps=steps)
    plugin.install(opts, model="groot")     # fits the method before the model loads (fail fast)
    from exp.libero_groot import serve_groot_libero as sg

    sys.argv = [sg.__file__, *rest]
    sg.main()


if __name__ == "__main__":
    main()
