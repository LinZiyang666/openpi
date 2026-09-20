import json, pathlib, sys
root = pathlib.Path(sys.argv[1]); order = ["screen_ddim_100", "test_ddpm_100", "test_ddim_100", "test_ddim_10", "test_ddim_4", "test_ddim_2", "test_ddim_1"]
for cell in sorted(p.name for p in root.iterdir() if p.is_dir() and not p.name.startswith("official")):
    row = []
    for a in order:
        f = root / cell / a / "summary.json"
        if f.exists():
            d = json.loads(f.read_text()); row.append(f"{a.replace('test_','').replace('screen_','S:')}={d['mean_score']:.3f}" if d.get("complete") else f"{a}=INCOMPLETE")
    print(cell, " ".join(row))
