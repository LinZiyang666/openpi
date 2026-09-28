from pathlib import Path
B=Path(__file__).resolve().parent;R=B/'regression'
for p in R.rglob('*.py'):
    s=p.read_text().replace("prefix='k10_estimator_'", "prefix='q2_estimator_'")
    s=s.replace("(BASE/'dev'/name).read_bytes()", "(BASE.parent/'dev'/name).read_bytes()")
    p.write_text(s)
# Checks still use the original K10 method/logic; default source is an argument.
for tag in ('k2','k1','k4'):
    p=R/'dev'/f'installed_{tag}.sh'
    (R/'dev'/f'dev_{tag}.sh').write_text(p.read_text().replace('/tmp/q2_installed','/tmp/q2_dev').replace('launch_test.py installed','launch_test.py dev'))
(R/'results/dev').mkdir(parents=True,exist_ok=True)
