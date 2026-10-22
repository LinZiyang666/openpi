"""Build the recording group: LIBERO-10 500 episodes, each task on its selected knob arm (LEVELS.json).

Entries are the l500 libero_10 arms of fa_pi_r_f_l_q (ir25/ir40/ir32) and fa_pi_r_f_o_q (off) with their pairs cut
to the tasks that selected them, and without long execution segments (owner 10-21): --os-blind, --os-policy-tail
and --os-policy-tail-blocks 1 removed, so every request looks (stage 1) and the method decides hit or miss.

  python -m exp.cluster_sim.record_group --root /data/mw_finetune/eval/cs_rec_pi05
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from exp.offline_search.rounds.r16 import conductor as rc

HERE = Path(__file__).parent
EVAL = Path('/nvme/weiland/openpi/eval')
LONG_SEGMENT_ARGS = ('--os-blind', '--os-policy-tail')


def plugin_args(args):
    out, skip = [], False
    for a in args:
        if skip:
            skip = False
        elif a == '--os-policy-tail-blocks':
            skip = True
        elif a not in LONG_SEGMENT_ARGS:
            out.append(a)
    return out


def build(root):
    levels = json.loads((HERE / 'LEVELS.json').read_text())
    tasks = {}
    for task, level in levels['selected'].items():
        tasks.setdefault(levels['arms'][level], []).append(int(task))
    groups = {g: json.loads((EVAL / g / 'group.json').read_text()) for g in ('fa_pi_r_f_l_q', 'fa_pi_r_f_o_q')}
    entries, definitions = [], {}
    for name, group in groups.items():
        definitions.update(group['definitions_sha256'])
        for entry in group['entries']:
            arm = entry['row']['arm']
            if arm not in tasks:
                continue
            e = copy.deepcopy(entry)
            keep = [p for p in e['pairs'] if p[0] in tasks[arm]]
            e.update(pairs=keep, dispatch_pairs=copy.deepcopy(keep))
            e['row']['plugin_args'] = plugin_args(e['row']['plugin_args'])
            entries.append(e)
    group = {k: v for k, v in groups['fa_pi_r_f_l_q'].items() if k not in ('entries', 'definitions_sha256', 'group_sha256')}
    group.update(root=str(root), entries=entries, definitions_sha256=definitions,
                 cluster_sim=dict(levels=levels['selected'], long_execution_segments=False))
    group['group_sha256'] = hashlib.sha256(rc.canonical(group)).hexdigest()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    (root / 'group.json').write_text(json.dumps(group, indent=1) + '\n')
    (root / 'ROOT.json').write_text(json.dumps(dict(group_sha256=group['group_sha256'],
                                                    definitions_sha256=definitions), indent=1) + '\n')
    print(json.dumps({e['row']['arm']: sorted({p[0] for p in e['pairs']}) for e in entries}),
          sum(len(e['pairs']) for e in entries), 'episodes')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', default='/data/mw_finetune/eval/cs_rec_pi05')
    build(parser.parse_args().root)


if __name__ == '__main__':
    main()
