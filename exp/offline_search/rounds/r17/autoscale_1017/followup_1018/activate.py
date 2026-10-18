"""Publish the fully checked candidate; never changes campaign directories."""
raise SystemExit("DISABLED by owner: both live-path trials must pass before any activation")
import datetime,hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent;BASE=HERE.parent;ROOT=BASE.parents[4]
assert not (HERE/'PUBLICATION_HOLD').exists(), 'publication held for urgent generation identity fix'
comparison=json.loads((HERE/'BEFORE_AFTER.json').read_text());assert comparison['status']=='PASS'
a=json.loads((HERE/'ACTIVATION.json').read_text());assert a['state']=='STAGED_NOT_ACTIVE'
before=json.loads((HERE/'BEFORE_SOURCES.json').read_text())
for name,pin in before.items():assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==pin['sha256'],name
for name,pin in a['candidate_sha256'].items():assert hashlib.sha256((HERE/'candidate'/Path(name).name).read_bytes()).hexdigest()==pin,name
assert hashlib.sha256((BASE/'fairness.py').read_bytes()).hexdigest()==a['new_modules']['fairness.py']
generation=a['generation_fix'];core=ROOT/generation['target']
assert hashlib.sha256(core.read_bytes()).hexdigest()==generation['before_sha256']
assert hashlib.sha256((HERE/'generation_candidate/worker.py').read_bytes()).hexdigest()==generation['after_sha256']
final=a['candidate_sha256']|a['new_modules']|{generation['target']:generation['after_sha256']}
records=json.loads((HERE/'combined_regression/COMMANDS.json').read_text())
assert len(records)==8
for record in records:assert record['source_sha256']==final
from exp.offline_search.rounds.r17.autoscale_1017.regression_report import cases
checked=cases(HERE/'COMBINED_CANDIDATE.xml')
assert checked and all(v['status']=='passed' for v in checked.values())
docs={name:(HERE/'candidate'/name).read_bytes() for name in ('DESIGN.md','DEFAULT_ON.md','TRIAL.md','trial_commands.sh')}
for name in docs:assert (BASE/name).read_bytes()==(HERE/'before'/name).read_bytes(),name
config=ROOT/'exp/r17_integration/ops/RUN_CONFIGS.md'
def block(text):return text[text.index('- **Worker count is a desired maximum;'):text.index('- Every batched Fast-WAM arm needs')]
old=block((HERE/'before/RUN_CONFIGS.md').read_text());new=block((HERE/'candidate/RUN_CONFIGS.md').read_text())
config_text=config.read_text();assert old in config_text
# Include the new dependency first; publish larger default maxima only after all
# runtime consumers are in place. Queue prepares are separate Python processes.
order=('bundle','driver','worker','node','fleet','legacy','defaults','formal_trial','legacy_release')
def publish(path,data):
    part=path.with_name(path.name+'.publish.part');part.write_bytes(data)
    os.chmod(part,path.stat().st_mode&0o777);os.replace(part,path)
publish(core,(HERE/'generation_candidate/worker.py').read_bytes())
for name in order:
    p=BASE/(name+'.py');publish(p,(HERE/'candidate'/p.name).read_bytes())
for name,data in docs.items():publish(BASE/name,data)
publish(config,config_text.replace(old,new,1).encode())
a.update(state='ACTIVE',activated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    source_checks='all 10 original runtime hashes matched; all final combined candidate hashes matched',
    preparations_executed=0,launches=0)
(HERE/'ACTIVATION.json').write_text(json.dumps(a,indent=2)+'\n')
print('ACTIVE',a['activated_utc'])
