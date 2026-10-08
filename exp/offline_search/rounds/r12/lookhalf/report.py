"""Summarize completed B pilot numerical shards, without launching anything."""
import argparse
import json
from pathlib import Path
from exp.offline_search.closed_loop.look_recording import read_looks, validate_root
from exp.offline_search.closed_loop.lookhalf_ledger import numerical_report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('root', type=Path)
    a = ap.parse_args()
    validate_root(a.root)
    reports = {}
    for row in json.loads((a.root/'arms.json').read_text()):
        looks = (look for directory in sorted((a.root/'runs'/row['arm']).glob('server_*/looks'))
                 for look in read_looks(directory))
        reports[row['arm']] = numerical_report(looks)
    (a.root/'numerical_paths.json').write_text(json.dumps(reports, indent=2)+'\n')
    print(json.dumps({k:dict(samples=v['samples'], statistics=v['statistics'], status=v['status']) for k,v in reports.items()}, indent=2))


if __name__ == '__main__':
    main()
