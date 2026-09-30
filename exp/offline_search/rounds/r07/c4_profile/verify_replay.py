"""Reuse C1's CPU replay identity assertion without its writer or launcher."""
from . import common as C


def main():
    p=C.parser(__doc__);a=p.parse_args();reports=[]
    from exp.offline_search.rounds.r07.c1_follow.replay import identity
    for cell in C.cells(a.cells):
        base,lib,manifest=C.bank(cell);table=C.stage_table(lib,manifest,base)
        bound=[lib.dir/'manifest.json',lib.dir/'action.npy']
        before={str(path):C.sha(path) for path in bound}
        for campaign in ['bval','p3']:
            counts=identity(cell,C.sources()[cell],table,campaign)
            reports.append(dict(cell=cell,campaign=campaign,**counts))
            print(cell,campaign,counts,flush=True)
        if before!={str(path):C.sha(path) for path in bound}:raise AssertionError('read-only source modified')
    C.write(a.out/'identity_replay.json',dict(reports=reports,
        definition='C1 recorded-stream A versus both SF and UF extend_blocks=0: full action dtype/shape/bytes, top-k/scores/weights, confidence, library, LOOK decisions and extras. Recorded action/vision/verdict parity checked. B-val exact recording exclusions preserved. No serving or policy model ran; unchanged source action/manifest digests checked. C4 tools introduce no serving lever.'))


if __name__=='__main__':main()
