"""Export serving weights only; leave original optimizer checkpoints intact."""
import argparse
from pathlib import Path
import shutil


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("checkpoint", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    if a.output.exists():
        raise FileExistsError(a.output)
    a.output.mkdir(parents=True)
    for name in ("model.safetensors", "metadata.json", "norm_stats.json", "complete.json"):
        shutil.copy2(a.checkpoint / name, a.output / name)
    print(a.output)


if __name__ == "__main__":
    main()
