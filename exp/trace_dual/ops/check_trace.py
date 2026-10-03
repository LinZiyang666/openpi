"""check_trace.py <trace-dir> [--detail]: per closed .h5 -> identity, closed/terminal/errors, steps, verdict mix."""
import sys, glob, json, collections, h5py
root = sys.argv[1]; detail = "--detail" in sys.argv
files = sorted(glob.glob(f"{root}/**/*.h5", recursive=True))
tmp = glob.glob(f"{root}/**/*.tmp", recursive=True) + glob.glob(f"{root}/**/*.reserved", recursive=True)
agg = collections.Counter()
for p in files:
    with h5py.File(p, "r") as f:
        a = f.attrs
        steps = [k for k in f.keys() if k.startswith("step_")]
        vc = collections.Counter()
        for k in steps:
            g = f[k]
            v = g.attrs.get("trace_verdict", g.attrs.get("verdict", "?"))
            vc[v if isinstance(v, str) else v.decode() if isinstance(v, bytes) else str(v)] += 1
        row = dict(file=p.split("/")[-1], uid=a.get("trace_task_uid"), att=int(a.get("trace_attempt", -1)),
                   yaml=a.get("trace_yaml_id"), closed=bool(a.get("trace_closed_ok", False)),
                   terminal=bool(a.get("trace_terminal", False)), werr=int(a.get("trace_write_errors", -1)),
                   n=int(a.get("num_steps", -1)), nstep_groups=len(steps), success=bool(a.get("success")),
                   noise=bool(a.get("trace_noise_actions_recorded", False)), verdicts=dict(vc))
        print(json.dumps(row))
        agg["files"] += 1; agg["closed_ok"] += row["closed"]; agg["terminal"] += row["terminal"]; agg["werr"] += row["werr"]
        if detail and steps:
            g = f[steps[0]]
            def walk(name, obj):
                if isinstance(obj, h5py.Dataset): print("   ", name, obj.shape, obj.dtype)
            print("  attrs:", sorted(a.keys()))
            print("  step0 attrs:", {k: (g.attrs[k] if not hasattr(g.attrs[k], 'shape') or g.attrs[k].size < 5 else '...') for k in g.attrs})
            g.visititems(walk)
            detail = False
print("SUMMARY", dict(agg), "unfinished_tmp_or_reserved", len(tmp))
