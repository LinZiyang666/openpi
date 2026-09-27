from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/verify_logs.py');s=p.read_text()
s=s.replace('def main(argv=None):',Path('exp/offline_search/rounds/r04/k2_serving/dev/verify_blind_snippet.txt').read_text()+'def main(argv=None):')
s=s.replace('    metas = {(m["method_spec"]', '    if any(m.get("blind", False) for _, m, _ in eps):\n        return verify_blind_logs(log_dir, eps, out=a.out or None)\n    metas = {(m["method_spec"]')
s=s.replace('                     burst_left: int):','                     burst_left: int, decision_index=None):')
s=s.replace('    if step == 0 and J.get("step0", "judge") != "judge":', '''    if mode == "periodic" and decision_index is not None and decision_index % int(J["k"]) == int(J["k"])-1:
        hit, why = False, "periodic"
    elif step == 0 and J.get("step0", "judge") != "judge":''')
s=s.replace('        hit, why = (step % k != k - 1), "periodic"', '        hit, why = ((step if decision_index is None else decision_index) % k != k - 1), "periodic"')
s=s.replace('                                                    forced=forced, prev_reason=None, burst_left=burst_left)', '''                                                    forced=forced, prev_reason=None, burst_left=burst_left,
                                                    decision_index=int(z["decision_index"][s]) if "has_vision" in z else None)
        if "has_vision" in z and not z["has_vision"][s]:
            exp_hit, why = True, "blind"''')
p.write_text(s)
