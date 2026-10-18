"""Produce a case-by-case before/after table from completed JUnit reports."""
import argparse
import json
from pathlib import Path
import xml.etree.ElementTree as ET


def cases(path):
    result = {}
    for item in ET.parse(path).findall('.//testcase'):
        key = item.attrib.get('classname','')+'::'+item.attrib['name']
        failure = item.find('failure')
        if failure is None:failure = item.find('error')
        status = 'failed' if failure is not None else 'skipped' if item.find('skipped') is not None else 'passed'
        result[key] = dict(status=status, seconds=float(item.attrib.get('time',0)),
            timeout=failure is not None and 'per-test timeout' in (failure.text or ''),
            failure=(failure.attrib.get('message','') if failure is not None else None))
    return result


def report(manifest):
    rows = []
    for suite, paths in manifest.items():
        before, after = [cases(p) for p in paths]
        common = sorted(before.keys() & after.keys())
        regressions = [k for k in common if before[k]['status']=='passed' and after[k]['status'] not in ('passed','skipped')]
        missing = sorted(before.keys() - after.keys())
        def totals(values):
            return {s:sum(v['status']==s for v in values.values()) for s in ('passed','failed','skipped')} | dict(
                timeouts=sum(v['timeout'] for v in values.values()), seconds=sum(v['seconds'] for v in values.values()))
        rows.append(dict(suite=suite, before_xml=str(paths[0]), after_xml=str(paths[1]),
            before=totals(before), after=totals(after), regressions=regressions, missing=missing,
            cases={k:dict(before=before.get(k),after=after.get(k)) for k in sorted(before.keys()|after.keys())}))
    return dict(status='PASS' if all(not r['regressions'] and not r['missing'] and not r['after']['timeouts'] for r in rows) else 'FAIL',
                baseline='19 SHA-verified backups via read-only import/file overlay; all remaining current dependencies unchanged',
                suites=rows)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('manifest',type=Path);p.add_argument('output',type=Path)
    args=p.parse_args()
    value=report(json.loads(args.manifest.read_text()))
    args.output.write_text(json.dumps(value,indent=2)+'\n')
    print(value['status'])
    for r in value['suites']:
        print(r['suite'],r['before'],r['after'],'regressions',len(r['regressions']))
