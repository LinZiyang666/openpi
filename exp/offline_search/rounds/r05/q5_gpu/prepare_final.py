"""Fresh final-test outputs; retain development evidence and original assertions."""
from pathlib import Path
B=Path(__file__).resolve().parent
R=B/'regression';F=B/'final_regression'
for p in R.rglob('*'):
    if not p.is_file() or p.suffix not in ('.py','.sh','.json') or 'results' in p.relative_to(R).parts:continue
    s=p.read_text().replace(str(R),str(F)).replace('/tmp/q5_','/tmp/q5_final_')
    # Exact same already-validated verification fits may be reused.
    s=s.replace('/tmp/q5_final_regression_fits/','/tmp/q5_regression_fits/')
    q=F/p.relative_to(R);q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
(F/'results/installed').mkdir(parents=True,exist_ok=True)
Q=B/'q2checks';T=B/'final_q2checks'
for p in Q.glob('*.py'):
    s=p.read_text().replace(str(R),str(F)).replace('/tmp/q5_','/tmp/q5_final_')
    q=T/p.name;q.parent.mkdir(parents=True,exist_ok=True);q.write_text(s)
(T/'results').mkdir(exist_ok=True)
s=(B/'run_q2.py').read_text().replace("B/'q2checks'","B/'final_q2checks'").replace("B/'regression'","B/'final_regression'").replace('/tmp/q5_','/tmp/q5_final_').replace("'results/q2_commands.json'","'results/final_q2_commands.json'").replace("f'q2_{name}.log'","f'final_q2_{name}.log'")
(B/'run_final_q2.py').write_text(s)
print(F,T)
