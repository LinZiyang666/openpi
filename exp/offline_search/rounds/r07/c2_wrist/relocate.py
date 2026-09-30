"""Repackage metadata in scratch for coordinator copy to a concrete <RUN>."""
import argparse
import json
import pickle
import shutil
from pathlib import Path
from .specs import make_specs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", type=Path, default=Path("/tmp/r7_C2/fits"))
    ap.add_argument("--out", type=Path, default=Path("/tmp/r7_C2/relocated"))
    ap.add_argument("--target-run", required=True)
    a = ap.parse_args()
    if Path("/tmp/r7_C2") not in a.out.resolve().parents:
        raise ValueError("relocation output must stay under /tmp/r7_C2")
    a.out.mkdir(parents=True, exist_ok=True)
    arms = make_specs(a.target_run)
    for row in arms:
        filename = row["name"]+".pkl"
        with (a.source/filename).open("rb") as f:
            blob = pickle.load(f)
        blob["kwargs"] = row["kwargs"]
        method = blob["method"]
        for field in ("base_kwargs", "base_fit", "stage_fit", "wrist_fit"):
            setattr(method, field, row["kwargs"][field])
        if hasattr(method.base, "follow_stages_path"):
            method.base.follow_stages_path = row["kwargs"]["base_kwargs"]["stages_path"]
        with (a.out/filename).open("wb") as f:
            pickle.dump(blob, f, protocol=4)
    for path in a.source.glob("*.pkl"):
        if path.name.startswith(("stages_", "wrist_")):
            shutil.copyfile(path, a.out/path.name)
    (a.out/"arms_in.json").write_text(json.dumps(arms, indent=2)+"\n")
    print(json.dumps(dict(arms=len(arms), output=str(a.out), target=a.target_run)))


if __name__ == "__main__":
    main()
