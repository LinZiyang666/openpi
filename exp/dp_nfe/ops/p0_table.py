import json, glob, pathlib, sys
root = pathlib.Path(sys.argv[1])
for task in sorted(p.name for p in root.iterdir() if p.name.startswith("official_")):
    row = {}
    for f in sorted((root / task).glob("test_*/summary.json")):
        d = json.loads(f.read_text()); row[f.parent.name.replace("test_", "")] = (round(d["mean_score"], 3), d["n_test"], d["manifest"].get("wall_s"))
    order = ["ddpm_100", "ddim_100", "ddim_10", "ddim_4", "ddim_2", "ddim_1"]
    print(task, {k: row[k] for k in order if k in row})
