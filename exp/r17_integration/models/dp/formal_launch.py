"""Local DP launcher shim; P6 retains sealing, workers and process ownership."""
from contextlib import contextmanager
from pathlib import Path
import argparse

from .batch_runtime import BatchConfig


@contextmanager
def launcher_hooks(launcher, config):
    original_prepare, original_python = launcher.prepare, launcher.host_python
    def prepare(campaign, directory):
        campaign["dp_batching"] = config.metadata()
        return original_prepare(campaign, directory)
    def host_python(campaign):
        if campaign["model"] == "dp" and campaign["domain"] == "libero":
            if campaign["server_node"] != "local":
                raise ValueError("DP batching shim supports local 4090 only")
            return str(Path(__file__).with_name("formal_python.sh"))
        return original_python(campaign)
    launcher.prepare, launcher.host_python = prepare, host_python
    try:
        yield
    finally:
        launcher.prepare, launcher.host_python = original_prepare, original_python


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, add_help=False)
    parser.add_argument("--max-batch", "--max-batch-size", type=int, default=1)
    parser.add_argument("--batch-wait-ms", type=float, default=5)
    args, remaining = parser.parse_known_args(argv)
    config = BatchConfig(args.max_batch, args.batch_wait_ms)
    from exp.r17_integration.ops import formal_eval
    if config.max_batch == 1:
        return formal_eval.main(remaining)
    parsed = formal_eval.parse_args(remaining)
    if parsed.mode != "launch" or parsed.model != "dp_libero" or parsed.server_node != "local":
        parser.error("DP batching shim supports local dp_libero launch only")
    if getattr(parsed, "library", False):
        parser.error("formal batch launch supports pure inference only; cache experiments need matched batching qualification")
    with launcher_hooks(formal_eval, config):
        return formal_eval.main(remaining)


if __name__ == "__main__":
    main()
