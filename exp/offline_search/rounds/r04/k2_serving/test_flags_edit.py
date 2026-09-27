from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/selftest.py');s=p.read_text()
s=s.replace('        args += ["--os-judge", a.judge]', '''        args += ["--os-judge", a.judge, "--os-judge-cap", str(a.judge_cap),
                 "--os-judge-step0", a.judge_step0, "--os-judge-burst", str(a.judge_burst)]''')
s=s.replace('blind = bool(is_probe and step % 6 in (1, 2) and s.hits and s.hits[-1] and not due)', 'blind = bool(is_probe and step % 6 in (1, 2) and s.hits and s.hits[-1] and not due\n                             and s.blind_age < s.method.budget)')
s=s.replace('                if is_probe and step == 1 and not due:', '                if is_probe and step == 1 and blind:')
s=s.replace('        if a.judge == "guard_only":','        if a.judge == "guard_only" and json.loads(a.kwargs).get("budget", 2) == 2:')
p.write_text(s)
