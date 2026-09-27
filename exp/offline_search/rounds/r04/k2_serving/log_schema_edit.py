from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/plugin.py');s=p.read_text()
s=s.replace('            arrays["vision"] = arrays["has_vision"]','            arrays["vision"] = arrays["has_vision"]\n            arrays["periodic_global"] = np.full(n, rt.blind, bool)')
p.write_text(s)
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/verify_logs.py');s=p.read_text()
s=s.replace('decision_index=int(z["decision_index"][s]) if "has_vision" in z else None)', 'decision_index=int(z["decision_index"][s]) if "has_vision" in z\n                                                    and ("periodic_global" not in z or z["periodic_global"][s]) else None)')
s=s.replace('    _, meta, _ = eps[0]\n', '''    configs = {(m["method_spec"], json.dumps(m["kwargs"], sort_keys=True), m["cell"], m["root"], m["run_seed"],
                json.dumps(m.get("judge"), sort_keys=True)) for _, m, _ in eps}
    if len(configs) != 1:
        raise ValueError("blind logs mix configurations; select one server tag")
    _, meta, _ = eps[0]
''')
p.write_text(s)
