"""Compare native checkpoint channels with the locally downloaded LeRobot stats."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", type=Path, default=Path("/data/mw_finetune/datasets/metaworld_mt50"))
    ap.add_argument("--output", type=Path, default=Path("exp/mw_finetune/cosmos/results/dataset_comparison.json"))
    a = ap.parse_args()
    root = Path(__file__).parent
    cp_path = root / "sources/dataset_statistics.json"
    ds_path = a.dataset / "meta/stats.json"
    cp = json.loads(cp_path.read_text())
    ds = json.loads(ds_path.read_text())
    result = {"checkpoint_stats_sha256": hashlib.sha256(cp_path.read_bytes()).hexdigest(),
              "dataset_stats_sha256": hashlib.sha256(ds_path.read_bytes()).hexdigest(), "comparisons": {}}
    for name, prefix, columns in (("action", "actions", [0, 1, 2, 6]),
                                  ("observation.state", "proprio", [2, 3, 4, 0])):
        result["comparisons"][name] = {}
        for statistic in ("min", "max", "mean", "std"):
            original = np.asarray(cp[prefix + "_" + statistic])[columns]
            dataset = np.asarray(ds[name][statistic])
            delta = original - dataset
            result["comparisons"][name][statistic] = {"checkpoint": original.tolist(), "dataset": dataset.tolist(),
                "delta": delta.tolist(), "max_abs_delta": float(np.abs(delta).max()),
                "allclose_1e_minus_6": bool(np.allclose(original, dataset, rtol=0, atol=1e-6))}
    tasks = pq.read_table(a.dataset / "meta/tasks.parquet").to_pylist()
    labels = set(json.loads((root / "sources/t5_cache_metadata.json").read_text()))
    dataset_labels = {str(x.get("task", x.get("__index_level_0__"))) for x in tasks}
    result["prompts"] = {"dataset_count": len(dataset_labels), "cache_count": len(labels),
                         "intersection_count": len(labels & dataset_labels),
                         "dataset_only": sorted(dataset_labels - labels), "cache_only": sorted(labels - dataset_labels)}
    result["exact_minmax_match"] = all(result["comparisons"][feature][stat]["allclose_1e_minus_6"]
                                          for feature in result["comparisons"] for stat in ("min", "max"))
    result["interpretation"] = "Similar extrema support related data, but mismatched statistics do not identify an exact training split or camera."
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
