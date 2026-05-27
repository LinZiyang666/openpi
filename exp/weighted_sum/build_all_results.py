"""合并 baseline + round_1/2/3 的 summarize 结果 json → all_results.csv。

每个输入 json 形如 {yaml_id: {"success_rate":..,"n":..,"n_success":..}}。
yaml_id 形如 "{keybuilder}__{cfg}"，cfg 里权重编码：
  grid3_vision_0@25_vision_1@37_robot_state@37  -> v0=.25 v1=.37 rs=.37
  grid_vision_0@50_robot_state@50               -> v0=.50 v1=0  rs=.50
  iso_vision_0 / iso_vision_1 / iso_robot_state -> 该模态=1.0
  末尾可带 __norm2（normalizer 用 shortlist 第二候选）

输出列：stage,keybuilder,yaml_id,v0,v1,rs,normalizer,n,success_rate（按 success_rate 降序）。

用法:
  python build_all_results.py --out all_results.csv \
      baseline=baseline_jup_results.json r1=round_1_results.json r2=... r3=...
"""
from __future__ import annotations

import argparse
import csv
import json
import re


def parse_weights(yaml_id: str):
    cfg = yaml_id.split("__", 1)[1] if "__" in yaml_id else yaml_id
    norm = "norm2" if "norm2" in cfg else "zscore"
    v0 = v1 = rs = 0.0
    m0 = re.search(r"vision_0@(\d+)", cfg)
    m1 = re.search(r"vision_1@(\d+)", cfg)
    mr = re.search(r"robot_state@(\d+)", cfg)
    if m0 or m1 or mr:
        if m0:
            v0 = int(m0.group(1)) / 100.0
        if m1:
            v1 = int(m1.group(1)) / 100.0
        if mr:
            rs = int(mr.group(1)) / 100.0
    else:  # iso_* 无 @，纯单模态权重 1.0
        if "iso_vision_0" in cfg:
            v0 = 1.0
        elif "iso_vision_1" in cfg:
            v1 = 1.0
        elif "iso_robot_state" in cfg:
            rs = 1.0
    return v0, v1, rs, norm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("stages", nargs="+", help="stage=path.json (e.g. baseline=foo.json r1=bar.json)")
    a = ap.parse_args()
    rows = []
    for spec in a.stages:
        stage, _, path = spec.partition("=")
        try:
            d = json.load(open(path))
        except OSError:
            print("skip missing", path)
            continue
        for yid, v in d.items():
            v0, v1, rs, norm = parse_weights(yid)
            kb = yid.split("__")[0]
            rows.append({
                "stage": stage, "keybuilder": kb, "yaml_id": yid,
                "v0": "%.4f" % v0, "v1": "%.4f" % v1, "rs": "%.4f" % rs,
                "normalizer": norm, "n": v.get("n", 0),
                "success_rate": "%.4f" % v.get("success_rate", 0.0),
            })
    rows.sort(key=lambda r: -float(r["success_rate"]))
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["stage", "keybuilder", "yaml_id", "v0", "v1", "rs", "normalizer", "n", "success_rate"])
        w.writeheader()
        w.writerows(rows)
    print("wrote %s (%d rows)" % (a.out, len(rows)))


if __name__ == "__main__":
    main()
