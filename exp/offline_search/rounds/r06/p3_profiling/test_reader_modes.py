"""Reader fallbacks and measured-count/snapshot checks, with synthetic fixtures."""
import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np

from .read_v2 import main as read_main
from .test_matrix import PREFIX


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out",type=Path,required=True)
    a=ap.parse_args()
    subprocess.run(PREFIX+["-m","exp.offline_search.rounds.r06.p3_profiling.test_v2_reader",
        "--matrix","/tmp/p3_v2_matrix_published","--out",str(a.out)],check=True)
    arm='pi05_l10_50_p1'
    journal=a.out/'runs'/arm/'client/journal.jsonl'
    lines=journal.read_text().splitlines(); journal.write_text('\n'.join(lines[:-1])+'\n')
    base=['read_v2','--run-root',str(a.out),'--arms',arm]
    with patch.object(sys,'argv',base+['--server-only','--out',str(a.out/'server_only')]):
        read_main()
    episodes=list(csv.DictReader((a.out/'server_only/episodes.csv').open()))
    anchors=list(csv.DictReader((a.out/'server_only/anchors.csv').open()))
    assert len(episodes)==4 and len(anchors)==48
    assert all(r['collection_mode']=='server_only' and r['controls_verified']=='False' and r['active_controls']=='' for r in episodes)
    assert all(r['client_environment_seed']=='' and r['environment_seed_verified']=='False' for r in episodes)
    assert all(r['actual_commit_controls']=='' and r['commit_truncated']=='' for r in anchors)
    assert not (a.out/'server_only/controls.csv').exists()
    try:
        with patch.object(sys,'argv',base+['--server-only','--require-stage-counts','--out',str(a.out/'missing_counters')]):
            read_main()
    except ValueError as exc:
        assert 'stage counts' in str(exc)
    else:
        raise AssertionError('mock stage counters accepted as measured')
    # A new local log copy with explicitly SYNTHETIC measured-count fixtures.
    server=a.out/'runs'/arm/'server_cpu'; original=server.resolve(); server.unlink(); server.mkdir()
    for folder in ('p3_inputs','inputs'):
        if (original/folder).exists(): (server/folder).symlink_to(original/folder,target_is_directory=True)
    for f in original.glob('decisions_*.jsonl'):
        records=[json.loads(x) for x in f.read_text().splitlines()]
        extras={(r['uid'],r['step']):len(r['resampling']['extra_chunks']) for r in records if r['ev']=='p3_anchor'}
        for r in records:
            if r['ev']=='p3_decision':
                r['stage_invocations']=dict(stage1=1,stage2=1,stage3=1+extras.get((r['uid'],r['step']),0))
        (server/f.name).write_text(''.join(json.dumps(x)+'\n' for x in records))
    for f in (a.out/'telemetry').rglob('controls.jsonl'):
        rows=[json.loads(x) for x in f.read_text().splitlines()]
        for r in rows:
            if r['ev']=='decision' and r['marker']['p3_anchor']:
                name='step_{}.npz'.format(r['decision_step'])
                r['snapshot']=dict(selected=True,probability=1.,path=name)
                meta=dict(task_uid=r['task_uid'],attempt=r['attempt'],decision_step=r['decision_step'])
                np.savez_compressed(f.parent/name,metadata_json=np.asarray(json.dumps(meta)),sim_state=np.zeros(4),rng_json=np.asarray('{}'),controller_skipped_json=np.asarray('[]'))
        f.write_text(''.join(json.dumps(x)+'\n' for x in rows))
    strict=base+['--client-root',str(a.out/'telemetry'),'--require-stage-counts','--require-snapshots']
    with patch.object(sys,'argv',strict+['--out',str(a.out/'strict')]): read_main()
    victim=next((a.out/'telemetry').rglob('step_*.npz'))
    victim.unlink()
    try:
        with patch.object(sys,'argv',strict+['--out',str(a.out/'missing_snapshot')]): read_main()
    except FileNotFoundError:
        pass
    else:
        raise AssertionError('missing selected snapshot accepted')
    report=dict(PASS=True,episodes=4,anchors=48,server_only_unknown_controls=True,
                absent_measured_counters_rejected=True,synthetic_snapshot_joins=48,missing_snapshot_rejected=True,
                evidence_scope='synthetic control/counter/snapshot fixtures; no simulator or real forwards')
    (Path(__file__).parent/'results/reader_modes.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report))


if __name__=='__main__': main()
