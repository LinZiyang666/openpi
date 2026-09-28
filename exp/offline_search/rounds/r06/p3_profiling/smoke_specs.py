"""Make small smoke specs using the SAME superset mode, including p=0 controls."""
import argparse
import copy
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    rows = json.loads(Path(__file__).with_name("arms.json").read_text())
    out = []
    for row in rows:
        if not row["name"].endswith("_r0"):
            continue
        for p in (0., .2, 1.):
            item = copy.deepcopy(row)
            item["name"] += "_smoke_p" + str(int(p * 100))
            item["kwargs"]["p"] = p
            item["plugin_args"][-1] = f'<RUN>/fits/{item["name"]}.pkl'
            out.append(item)
    a.out.write_text(json.dumps(out, indent=2) + "\n")


if __name__ == "__main__":
    main()
