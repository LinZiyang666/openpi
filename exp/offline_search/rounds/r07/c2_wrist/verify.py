"""Compare pre/post-install streams and collect exact verification counts."""
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
SCRATCH = Path("/tmp/r7_C2")


def arrays_equal(a, b):
    return np.array_equal(a, b, equal_nan=True) if a.dtype.kind in "fc" else np.array_equal(a, b)


def main():
    reports = []
    for label in ("pi05_cache", "groot_cache", "pi05_miss", "groot_miss"):
        before, after = [SCRATCH / "tests" / (phase+"_"+label) for phase in ("before", "after")]
        old = {p.name: p for p in (before/"inputs").glob("*.npz")}
        new = {p.name: p for p in (after/"inputs").glob("*.npz")}
        assert old.keys() == new.keys()
        n, array_fields = 0, 0
        for name in old:
            with np.load(old[name]) as a, np.load(new[name]) as b:
                assert a.files == b.files
                for key in a.files:
                    if key != "meta" and not key.endswith(("_ms", "_us")):
                        assert arrays_equal(a[key], b[key]), (label, name, key)
                        array_fields += 1
                n += len(a["step"])
        semantic = ("uid", "step", "top1", "topk", "scores", "conf", "synth", "nfix", "winner", "extras",
                    "exec_ok", "hit", "judge", "tau", "src", "vision", "served_head")
        streams = []
        for directory in (before, after):
            path = next(directory.glob("decisions_*.jsonl"))
            rows = [row for line in path.read_text().splitlines() if (row := json.loads(line)).get("ev") == "dec"]
            streams.append([{key: row.get(key) for key in semantic} for row in rows])
        assert streams[0] == streams[1]
        report = json.loads((after/"selftest_report.json").read_text())
        assert report["PASS"]
        reports.append(dict(test=label, episodes=len(old), decisions=n, array_fields_exact=array_fields, PASS=True))
    for label in ("pi05", "groot"):
        report = json.loads((SCRATCH/"tests"/("blind_"+label)/"selftest_report.json").read_text())
        assert report["PASS"]
        reports.append(dict(test="blind_"+label, **report))
    replay = json.loads((SCRATCH/"replay_report.json").read_text())
    assert all(row["PASS"] for row in replay)
    result = dict(PASS=True, shared_file_identity=reports,
                  method_off_identity=dict(episodes=sum(r["episodes"] for r in replay),
                      decisions=sum(r["decisions"] for r in replay), datasets=len(replay)),
                  enabled_replay=replay,
                  wrist_metric_comparisons=sum(r["enabled_fixed_stream"][v]["wrist_metric_exact"] for r in replay for v in ("sw", "sf_sw")))
    (HERE/"VERIFICATION.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "enabled_replay"}, indent=2))


if __name__ == "__main__":
    main()
