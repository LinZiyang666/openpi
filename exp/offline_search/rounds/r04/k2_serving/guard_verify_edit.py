from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/selftest.py');s=p.read_text()
s=s.replace('and s.blind_age < s.method.budget)', '''and s.blind_age < s.method.budget
                             and not (rt.judge and (s.burst_left > 0 or
                                 (rt.judge.cap > 0 and rt.judge.mode not in ("always", "periodic")
                                  and plugin._trailing_hits(s.hits) >= rt.judge.cap))))''')
s=s.replace('        if a.judge == "guard_only" and json.loads(a.kwargs).get("budget", 2) == 2:', '        if a.judge == "guard_only" and json.loads(a.kwargs).get("budget", 2) == 2 and not a.judge_cap and a.judge_burst == 1:')
p.write_text(s)
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/verify_logs.py');s=p.read_text()
s=s.replace('        age, last_vision = 0, -1', '        age, last_vision, burst_left = 0, -1, 0')
s=s.replace('            elif not hasattr(method, "blind_step"):', '''            elif J and (burst_left > 0 or (J.get("cap", 0) > 0 and J["mode"] not in ("always", "periodic")
                                          and int(z["run"][s]) >= J["cap"])):
                reason = LookReason(8, "judge requires vision")
            elif not hasattr(method, "blind_step"):''')
s=s.replace('            checks["dense_history"] += dense_ok', '''            checks["dense_history"] += dense_ok
            if J:
                why = str(z["judge"][s])
                forced = (z.get("x_os_force_miss", np.zeros(len(hit)))[s] == 1)
                if why == "burst":
                    burst_left -= 1
                elif forced and not hit[s] and J.get("burst", 1) > 1 and J["mode"] not in ("always", "periodic"):
                    burst_left = J["burst"] - 1''')
p.write_text(s)
