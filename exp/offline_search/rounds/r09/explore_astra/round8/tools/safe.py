"""Round8 admission: identity fields are parsed before outcome payloads."""
from pathlib import Path
import json
HERE=Path(__file__).resolve().parents[1]
RUNS=Path('/home/weiland/trace_runs/os_closed_loop')
STORE=Path('/home/weiland/trace_runs/offline_search_store')

def admitted(path):
    p = Path(path)
    if any(s.startswith(('r09_holdout', 'r09_astra_holdout')) for s in p.parts):
        raise ValueError('forbidden root')
    p = p.resolve()
    if any(s.startswith(('r09_holdout', 'r09_astra_holdout')) for s in p.parts):
        raise ValueError('forbidden root')
    return p


def json_identity(raw):
    """Lex only scalar identity values, skipping all other values before decoding payload.

    JSON strings are scanned with escapes. Nested payloads are skipped without
    deserialization. Duplicate identity fields fail closed.
    """
    dec = json.JSONDecoder()
    i = 0
    while i < len(raw) and raw[i].isspace(): i += 1
    if i == len(raw) or raw[i] != '{': return None
    i += 1
    fields = {}
    wanted = {'init', 'task_id', 'task_uid', 'uid'}
    while i < len(raw):
        while i < len(raw) and (raw[i].isspace() or raw[i] == ','): i += 1
        if i == len(raw) or raw[i] == '}': break
        key, i = dec.raw_decode(raw, i)
        while raw[i].isspace(): i += 1
        if raw[i] != ':': raise ValueError('bad JSON framing')
        i += 1
        while raw[i].isspace(): i += 1
        start = i
        depth, quote, escape = 0, False, False
        while i < len(raw):
            c = raw[i]
            if quote:
                if escape: escape = False
                elif c == chr(92): escape = True
                elif c == '"': quote = False
            elif c == '"': quote = True
            elif c in '[{': depth += 1
            elif c in ']}':
                if depth == 0: break
                depth -= 1
            elif c == ',' and depth == 0: break
            i += 1
        if key in wanted:
            if key in fields: raise ValueError('duplicate identity field')
            fields[key] = json.loads(raw[start:i])
    identities = []
    if fields.get('init') is not None and fields.get('task_id') is not None:
        if type(fields['init']) is not int or type(fields['task_id']) is not int:
            raise ValueError('identity must contain integer task/init')
        identities.append((fields['task_id'], fields['init']))
    for k in ('uid', 'task_uid'):
        if fields.get(k):
            if not isinstance(fields[k], str): raise ValueError('UID must be a string')
            parts = fields[k].rsplit(':', 2)
            if len(parts) == 3 and parts[-1].isdigit() and parts[-2].isdigit():
                identities.append(tuple(map(int, parts[-2:])))
            else:
                raise ValueError('malformed UID')
    if not identities: return None
    if len(set(identities)) != 1: raise ValueError('conflicting identities')
    return identities[0]


def admitted_jsonl(path, inits=range(30)):
    allowed = set(inits)
    if not allowed <= set(range(30)): raise ValueError('forbidden requested inits')
    with admitted(path).open() as f:
        for raw in f:
            identity = json_identity(raw)
            if identity is None or not (0 <= identity[0] < 10 and identity[1] in allowed):
                continue
            yield json.loads(raw)



def owned(path):
    p=admitted(path)
    if not p.is_relative_to(HERE): raise ValueError('write outside round8')
    p.parent.mkdir(parents=True,exist_ok=True)
    return p

def dump(path,value):
    import numpy as np
    owned(path).write_text(json.dumps(value,indent=2,allow_nan=False,default=lambda x:x.tolist() if isinstance(x,np.ndarray) else x.item() if isinstance(x,np.generic) else str(x))+'\n')
