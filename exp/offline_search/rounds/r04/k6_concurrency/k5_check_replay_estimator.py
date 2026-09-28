"""Join installed plugin replay logs to synthetic accepted journals, then estimate.
The journal labels come from replay episode metadata; these are NOT new LIBERO outcomes.
Also read both historical guard-only arms to validate the real 500-init layout.
"""
import json
from pathlib import Path
from exp.offline_search.rounds.r04.k5_rand.estimate import load_arm,estimate
BASE=Path(__file__).resolve().parent
root=Path('/tmp/k6_replay_estimator'); root.mkdir(exist_ok=True)
reports=[]
for scale in (50,500):
    rows=[]
    for rep in (1,2):
        arm=f'r4k5_p_l10_g{scale}_r{rep}'
        source=Path(f'/tmp/k6_installed_replays/{arm}')
        directory=root/'runs'/arm
        (directory/'client').mkdir(parents=True,exist_ok=True)
        if not (directory/'server_1').exists(): (directory/'server_1').symlink_to(source)
        eps=[json.loads(l) for l in (source/'decisions_selftest.jsonl').read_text().splitlines() if '"ev": "episode"' in l]
        jr=[dict(task_uid=e['uid'],attempt=e['attempt'],accepted=True,status='done',success=e['success']) for e in eps]
        (directory/'client/journal.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in jr))
        r,audit=load_arm(root,arm);rows+=r
    historical='r3mx_p_l10_g'+('500' if scale==500 else '')
    baseline,audit=load_arm('/home/weiland/trace_runs/os_closed_loop/r03_mx',historical,baseline=True)
    assert len(baseline)==500
    report=estimate(rows,499,20260927,{historical:baseline})
    (BASE/'results'/'installed'/f'replay_estimate_g{scale}.json').write_text(json.dumps(report,indent=2,allow_nan=False))
    reports.append(dict(scale=scale,replay_episodes=len(rows),replay_decisions=sum(r['N'] for r in rows),baseline_episodes=len(baseline),baseline_requests=sum(r['N'] for r in baseline),baseline_misses=sum(r['M'] for r in baseline),baseline_SR=sum(r['Y'] for r in baseline)/len(baseline),exposed=sum(r['exposed'] for r in rows),PASS=True))
(BASE/'results/installed/replay_estimator.json').write_text(json.dumps(reports,indent=2))
print(json.dumps(reports,indent=2))
