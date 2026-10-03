"""Direct h100-side wrappers with the original start/stop positional interface."""
import json
import os
from pathlib import Path
import sys

import node


def main():
    action, *args = sys.argv[1:]
    if action == "start":
        model, suite, port, yaml, out, tag, *plugin = args
        full = os.environ.get("STAGE1_ONLY", "1") == "0"
        env = {k: v for k, v in os.environ.items() if k in
               ("STAGE1_ONLY", "STOCK", "OMP", "GROOT_DENOISING_STEPS", "GPU_LOCK", "PI05_CKPT", "GROOT_CKPT")}
        spec = dict(role="server", model=model, suite=suite, port=int(port), yaml=yaml, out=out,
                    tag=tag, plugin=plugin, env=env,
                    need_mb=int(os.environ.get("NEED_MB", ("8000" if model == "groot" else "9000") if full else "3000")))
        if os.environ.get("OSCL_LAUNCH_ID"):
            spec["launch_id"] = os.environ["OSCL_LAUNCH_ID"]
        node.launch(spec)
    elif action == "stop":
        out, tag, *_ = args
        receipt = Path(out) / f"server_{tag}.owner.json"
        if receipt.exists():
            node.stop(json.loads(receipt.read_text())["spec"], timeout_s=int(args[2]) if len(args) > 2 else 90)
        else:
            print("NO_OWNED_LAUNCH")


if __name__ == "__main__":
    main()
