from pathlib import Path
import json,hashlib
here=Path(__file__).resolve().parent;base=here.parent
# Keep the staged fairness/maxima implementation, adding the same OFF guards.
p=here/'candidate/defaults.py';s=p.read_text();s=s.replace('from .policy import Config, NODES','from .policy import Config, NODES\nfrom .switch import enabled',1)
live=(base/'defaults.py').read_text();guard=live[live.index('    if not enabled():',live.index('def apply')):live.index("    if c.get('worker_nodes')",live.index('def apply'))]
s=s.replace('    c = campaign\n','    c = campaign\n'+guard,1).replace("    explicit = unit.get('autoscale_node_maxima')","    if not enabled():return dict(unit.get('workers', {}))\n    explicit = unit.get('autoscale_node_maxima')",1);p.write_text(s)
p=here/'candidate/bundle.py';s=p.read_text().replace("'fairness')","'fairness', 'switch')",1).replace("return [PREFIX + name + '.py' for name in MODULES] + [","return [PREFIX + name + '.py' for name in MODULES] + [PREFIX + 'ENABLED',",1);p.write_text(s)
p=here/'candidate/legacy_release.py';s=p.read_text().replace('from .node import atomic, read','from .node import atomic, read\nfrom .switch import enabled',1).replace('    from exp.r17_integration.ops.formal_eval import transfer\n','    if not enabled():return None\n    from exp.r17_integration.ops.formal_eval import transfer\n',1).replace("    record = read(Path(root)/'LEGACY_AUTOSCALE.json')","    if not enabled():return None\n    record = read(Path(root)/'LEGACY_AUTOSCALE.json')",1);p.write_text(s)
p=here/'candidate/formal_trial.py';s=p.read_text().replace('    from .compare import wait_io\n    wait_io()','    from .compare import wait_io\n    from .switch import enabled\n    if not enabled():\n        if maxima is not None:raise ValueError("global OFF: native preparation retains original counts")\n        return stage.stage(root)\n    wait_io()',1);p.write_text(s)
(here/'CANDIDATE_REVISION.json').write_text(json.dumps(dict(state='STAGED_OFF_PENDING_OWNER_TRIALS',files={str(p.relative_to(here)):hashlib.sha256(p.read_bytes()).hexdigest() for p in (here/'candidate').glob('*.py')},global_switch='off'),indent=2)+'\n')
