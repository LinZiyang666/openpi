"""Sealed per-episode metadata views over immutable shared local capture blocks."""
import json
import os
from pathlib import Path

from exp.r17_integration.debug.attempts import capture_counts, json_rows
from exp.r17_integration.debug.validate import digest, file_sha
from .build_client_bundle import safe_name
from .formal_process import write_json


def episode_views(sources, destination, binding):
    result, originals = [], {}
    for source in map(Path,sources):
        originals.update({str(p):file_sha(p) for p in source.rglob('*') if p.is_file()})
        paths=list(source.glob('decisions_*.jsonl'))
        try:
            rows={p.name:json_rows(p) for p in paths}
            acks=json_rows(source/'execution.jsonl')
            intents=json_rows(source/'decision_intents.jsonl')
            keys={(r['task_uid'],r['attempt']) for rr in [*rows.values(),acks,intents] for r in rr}
            if len(keys)<=1:
                result.append(source);continue
            # A view cannot cure an incomplete/torn writer.
            capture_counts([source],binding=binding)
            manifest=json.loads((source/'r17_capture.json').read_text())
        except (ValueError,KeyError,OSError):
            result.append(source);continue
        for key in sorted(keys):
            target=Path(destination)/digest((str(source),key))/'binding_view'
            target.mkdir(parents=True,exist_ok=True)
            selected={name:[r for r in rr if (r['task_uid'],r['attempt'])==key] for name,rr in rows.items()}
            records=[r for rr in selected.values() for r in rr]
            files=set()
            for row in records:
                for folder,field in [('blocks','blk'),('rawkeys','rawkeys_blk')]:
                    if field in row:
                        name=str(safe_name(folder+'/'+row[field]))
                        if name not in manifest['files']:raise ValueError('view references unsealed capture block')
                        files.add(name)
            for name in files:
                src=source/name;dst=target/name;dst.parent.mkdir(parents=True,exist_ok=True)
                if dst.exists():
                    if file_sha(dst)!=manifest['files'][name]:raise ValueError('changed capture view block')
                else:os.link(src,dst)
            filtered=dict(selected,**{'execution.jsonl':[r for r in acks if (r['task_uid'],r['attempt'])==key]})
            if intents:filtered['decision_intents.jsonl']=[r for r in intents if (r['task_uid'],r['attempt'])==key]
            for name,rr in filtered.items():
                (target/name).write_text(''.join(json.dumps(r,sort_keys=True)+'\n' for r in rr));files.add(name)
            stats=dict(manifest['writer_stats'],accepted=len(records),written=len(records))
            view=dict(manifest,writer_stats=stats,files={n:file_sha(target/n) for n in files},
                view_of=dict(source=str(source),manifest_sha256=file_sha(source/'r17_capture.json'),
                    task_uid=key[0],attempt=key[1],kind='sealed_episode_metadata_view'))
            write_json(target/'r17_capture.json',view)
            result.append(target)
    return result,originals
