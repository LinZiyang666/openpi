"""Seal the unchanged R14 rules plus R15 identity/layout metadata, before outcomes."""
from datetime import datetime,timezone
import copy,json
from pathlib import Path
from exp.offline_search.closed_loop import devset
from .prepare import HERE,R14,REPO,CELLS,write_json,atomic


def main():
    source=R14/'descent_astra'
    p=copy.deepcopy(json.loads((source/'protocol.json').read_text()))
    p.update(schema='r15.descent.protocol.v1',frozen_at_utc=datetime.now(timezone.utc).isoformat(),
        parent_protocol_sha256=devset.sha(source/'protocol.json'),
        parent_protocol_markdown_sha256=devset.sha(source/'PROTOCOL.md'),cells=list(CELLS),
        root_prefix='r15_descent_',comparison_arms=[],
        incomplete='all twelve cells, five arms per cell and four shared C10 references; exact 300 paired B identities per model/suite',
        confidence_scope='unchanged family=200 separately within each four-cell library-size group; no 600-comparison joint claim',
        test_references=json.loads((HERE/'c10_proof.json').read_text())['test_references'],
        leave_own_episode_out_sizes=[500],fits_include_own_episode=True,
        r15_adapter='allocation arms not repeated per D1; no allocation agreement/matched-IR result; selection and statistics unchanged')
    inputs=[HERE/n for n in ('selection.json','cap_tables.json','fits.json','c10_proof.json',
                             'recipe.py','prepare.py','analyze.py','analyze_test_a.py')]
    inputs += [source/'protocol.json',source/'PROTOCOL.md',
               REPO/'exp/offline_search/rounds/r15/COORDINATOR_DECISIONS.md']
    p['input_sha256']={str(x.relative_to(REPO)):devset.sha(x) for x in inputs}
    atomic(HERE/'PROTOCOL.md',(source/'PROTOCOL.md').read_bytes())
    write_json(HERE/'protocol.json',p)
    atomic(HERE/'PROTOCOL.sha256',''.join(f'{devset.sha(HERE/n)}  {n}\n' for n in ('protocol.json','PROTOCOL.md')).encode())
    print('FROZEN',p['frozen_at_utc'],'zero outcome reads')

if __name__=='__main__':main()
