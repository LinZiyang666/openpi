from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/replay_client.py');s=p.read_text()
s=s.replace('def main(argv=None):',Path('exp/offline_search/rounds/r04/k2_serving/dev/wire_snippet.txt').read_text()+'def main(argv=None):')
s=s.replace('    a = ap.parse_args(argv)', '''    ap.add_argument("--max-decisions", type=int, default=0, help="limit each replayed episode (smoke only)")
    ap.add_argument("--wire-checkpoint", default="", help="CPU-only offline output-transform parity against this checkpoint")
    a = ap.parse_args(argv)''')
s=s.replace('    resp_rows = {}', '    resp_rows = {}\n    wire_rows, observation_rows = {}, {}')
s=s.replace('        metas = []\n        for r in range(e["start"], e["end"]):', '''        metas, wires, observations = [], [], []
        end = min(e["end"], e["start"] + a.max_decisions) if a.max_decisions else e["end"]
        for r in range(e["start"], end):''')
s=s.replace('            metas.append(out.get("__hit_meta__") or {})', '''            metas.append(out.get("__hit_meta__") or {})
            wires.append(np.asarray(out["actions"]))
            observations.append(obs)''')
s=s.replace('        resp_rows[uid] = (ei, metas)', '        resp_rows[uid] = (ei, metas)\n        wire_rows[uid], observation_rows[uid] = wires, observations')
s=s.replace('        rows = np.arange(e["start"], e["end"])','        rows = np.arange(e["start"], e["start"] + len(metas))')
s=s.replace('        v0_max = max(v0_max, float(np.abs(z["key_v0"] - sv0).max()))','        vision = np.asarray(z.get("has_vision", np.ones(n, bool)), bool)\n        v0_max = max(v0_max, float(np.abs(z["key_v0"][vision] - sv0[vision]).max(initial=0)))')
s=s.replace('        v1_max = max(v1_max, float(np.abs(z["key_v1"] - sv1).max()))','        v1_max = max(v1_max, float(np.abs(z["key_v1"][vision] - sv1[vision]).max(initial=0)))')
s=s.replace('            c = (A * B).sum(1)', '            A, B = A[vision], B[vision]\n            c = (A * B).sum(1)')
s=s.replace('    if verdict_n:\n', '    if verdict_n and any("judge" in z for z in logged.values()):\n')
s=s.replace('    print(json.dumps(rep, indent=1))', '''    if any("has_vision" in z for z in logged.values()):
        vision_n = sum(int(z["has_vision"].sum()) for z in logged.values())
        stage_calls = sum(int(z["stage1_calls"][-1] - z["stage1_calls"][0] + int(z["has_vision"][0])) for z in logged.values())
        rep["blind"] = {"vision": vision_n, "blind": acc["n"] - vision_n, "stage1_calls": stage_calls,
                        "stage1_equals_vision": stage_calls == vision_n,
                        "s1_null_on_blind": all(np.isnan(z["s1_ms"][~z["has_vision"]]).all() for z in logged.values()),
                        "wire_equals_log": sum(int(np.array_equal(np.asarray(wire_rows[uid]), z["wire_actions"]))
                                               for uid, z in logged.items())}
    if a.wire_checkpoint:
        transform = offline_wire_transform(store.parse_cell(a.cell)[0], a.wire_checkpoint)
        equal = checked = 0
        max_abs = 0.
        for uid, z in logged.items():
            for s, action in enumerate(z["a_exec"]):
                want = transform(observation_rows[uid][s], action)
                got = wire_rows[uid][s]
                checked += 1
                equal += int(np.array_equal(want, got))
                max_abs = max(max_abs, float(np.max(np.abs(want - got))))
        rep["offline_wire"] = {"checked": checked, "bit_equal": equal, "max_abs": max_abs}
    print(json.dumps(rep, indent=1))''')
s=s.replace('    return 0\n', '''    if "blind" in rep:
        b = rep["blind"]
        if not (b["stage1_equals_vision"] and b["s1_null_on_blind"] and b["wire_equals_log"] == len(logged)):
            return 1
    if "offline_wire" in rep and rep["offline_wire"]["bit_equal"] != rep["offline_wire"]["checked"]:
        return 1
    return 0
''')
p.write_text(s)
