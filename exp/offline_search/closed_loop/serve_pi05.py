"""pi0.5 pure-cache server whose CP1 retrieval is an offline-harness Method (see plugin.py / README.md).

    cd /home/weiland/projects/openpi && taskset -c <cpus> .venv/bin/python exp/offline_search/closed_loop/serve_pi05.py \
        --os-method exp.offline_search.harness.baselines:B0Current --os-kwargs '{}' \
        --os-cell pi05_spatial_cache --os-log-dir <dir> --os-tag <arm>_<port> [--os-log-inputs] \
        --port 23110 --cache-config <arm yaml> policy:checkpoint --policy.config pi05_libero \
        --policy.dir /home/weiland/.cache/openpi/openpi-assets/checkpoints/pi05_libero_pytorch

Every ``--os-*`` flag is the plugin's; everything else goes to scripts/serve_policy.py unchanged. Refused here:
--replicas > 1 (children are spawned without the plugin), --non-concurrent (no per-connection factory to wrap),
--trace-out / --trace-build-cache (the trace twins would be built through the patched builder).
"""
from __future__ import annotations

import importlib.util
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
    import tyro

    spec = importlib.util.spec_from_file_location("serve_policy", REPO / "scripts" / "serve_policy.py")
    sp = importlib.util.module_from_spec(spec)
    sys.modules["serve_policy"] = sp
    spec.loader.exec_module(sp)
    args = tyro.cli(sp.Args, args=rest)
    if args.replicas != 1:
        raise SystemExit("osplug: --replicas must be 1 (replica children would run without the plugin)")
    if args.non_concurrent or not args.concurrent:
        raise SystemExit("osplug: the concurrent server is required")
    if args.trace_out or args.trace_build_cache:
        raise SystemExit("osplug: --trace-out / --trace-build-cache are not supported with the plugin")
    if not args.cache_config:
        raise SystemExit("osplug: --cache-config <yaml> is required (the served library and key builder)")
    plugin.install(opts, model="pi05")      # fits the method before the model loads (fail fast)
    logging.basicConfig(level=logging.INFO, force=True)
    sp.main(args)


if __name__ == "__main__":
    main()
