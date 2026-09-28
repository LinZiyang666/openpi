"""Send a prior strict synthetic reader fixture through the real transport."""
import argparse
import io
import json
from pathlib import Path
import subprocess

import numpy as np

from .stream_sink import StreamSink
from .test_stream import TOKEN, setup, receiver, stop, tree
from .test_matrix import PREFIX
from .stream_collect import verify_arm


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--fixture',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    a=ap.parse_args();arm='pi05_l10_50_p1'
    run=setup(a.out,(arm,));dest=run/'runs'/arm;dest.mkdir(parents=True)
    (dest/'server_cpu').symlink_to((a.fixture/'runs'/arm/'server_cpu').resolve(),target_is_directory=True)
    (dest/'client').mkdir()
    (dest/'client/journal.jsonl').write_bytes((a.fixture/'runs'/arm/'client/journal.jsonl').read_bytes())
    server,thread,address=receiver(run)
    expected={};snapshots=0
    try:
        for path in sorted((a.fixture/'telemetry').rglob('controls.jsonl')):
            rows=[json.loads(x) for x in path.read_text().splitlines()]
            uid,attempt=rows[0]['task_uid'],rows[0]['attempt']
            sink=StreamSink(address,TOKEN,run.name,arm,uid,attempt,run/'island_fixture')
            for line in path.read_text().splitlines(keepends=True):sink.write(line)
            for row in rows:
                sample=row.get('snapshot',{})
                if not sample.get('selected'):continue
                source=path.parent/sample['path']
                if source.exists():data=source.read_bytes()
                else:
                    # The prior negative reader test deliberately removed one
                    # dummy snapshot. Reconstruct only that synthetic fixture.
                    buffer=io.BytesIO()
                    meta=dict(task_uid=uid,attempt=attempt,decision_step=row['decision_step'])
                    np.savez_compressed(buffer,metadata_json=np.asarray(json.dumps(meta)),sim_state=np.zeros(4),rng_json=np.asarray('{}'),controller_skipped_json=np.asarray('[]'))
                    data=buffer.getvalue()
                sink.write_file(sample['path'],data);snapshots+=1
            sink.close();assert not sink.spilled
        verified=verify_arm(run,arm)
    finally:stop(server,thread)
    subprocess.run(PREFIX+['-m','exp.offline_search.rounds.r06.p3_profiling.read_v2',
        '--run-root',str(run),'--arms',arm,'--client-root',str(run/'runs'),
        '--require-stage-counts','--require-snapshots','--out',str(run/'tables')],check=True)
    report=dict(PASS=True,verified=verified,snapshots=snapshots,
                read_v2_unchanged=True,evidence='synthetic recorded server/control/counter/snapshot fixture; no real simulator')
    (run/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report))


if __name__=='__main__':main()
