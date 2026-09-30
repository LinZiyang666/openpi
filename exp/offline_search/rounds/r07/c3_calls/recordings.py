"""Read-only strict-table/input-archive streams, without outcomes or test fits."""
import json
from pathlib import Path
from types import SimpleNamespace as NS

import numpy as np
import pandas as pd

from exp.offline_search.rounds.r06.ideation_Q1.method_c.fit_calibration import AKEY, EKEY
from .common import RECORDINGS


def streams(cell, cohort):
    root = RECORDINGS if cohort == 'bval' else RECORDINGS.parent / 'r06_p3_pilot'
    name = cell if cohort == 'bval' else cell.replace('spatial', 'sp')
    tables = root / 'tables' / name
    cols = set(AKEY + ['assignment.cohort', 'task_id', 'init', 'retrieval.rows',
                      'retrieval.weights', 'retrieval.metric_code', 'retrieval.metric'])
    anchors = pd.read_csv(tables / 'anchors.csv', usecols=lambda k: k in cols)
    anchors = anchors[anchors['assignment.cohort'] == 'A']
    ids = set(map(tuple, anchors[EKEY].drop_duplicates().to_numpy()))
    lookup = {tuple(row[k] for k in AKEY): row for row in anchors.to_dict('records')}
    cols = set(AKEY + ['task_id', 'init', 'absolute_input_archive', 'actual_controls', 'vision',
        'blind_features.retrieval.rows', 'blind_features.retrieval.weights',
        'blind_features.retrieval.metric_code', 'blind_features.retrieval.metric'])
    decisions = pd.read_csv(tables / 'decisions.csv', usecols=lambda k: k in cols)
    for identity, group in decisions.groupby(EKEY, sort=True):
        if identity not in ids:
            continue
        group = group.sort_values('step')
        if list(group.step) != list(range(len(group))):
            raise ValueError('noncontiguous recording')
        episode = NS(uid=identity[1], task_id=int(group.task_id.iloc[0]), init=int(group.init.iloc[0]))
        data = []
        for row in group.to_dict('records'):
            with np.load(row['absolute_input_archive'], allow_pickle=False) as archive:
                values = {k: np.array(archive[k]) for k in
                          ('vision_0', 'vision_1', 'robot_state', 'raw_state', 'policy_chunk', 'executed_chunk')}
            anchor = lookup.get(tuple(row[k] for k in AKEY))
            rec, prefix = (anchor, 'retrieval.') if anchor else (row, 'blind_features.retrieval.')
            values.update(step=int(row['step']), actual_controls=int(row['actual_controls']),
                          recorded_vision=bool(row['vision']), recorded_anchor=anchor is not None)
            if isinstance(rec.get(prefix + 'rows'), str):
                values.update(rows=np.array(json.loads(rec[prefix + 'rows']), np.int64),
                    weights=np.array(json.loads(rec[prefix + 'weights']), np.float32),
                    key=dict(metric_code=np.array(json.loads(rec[prefix + 'metric_code']), np.float32),
                             metric=rec[prefix + 'metric']))
            data.append(values)
        yield episode, data


def query(episode, data, actions, hits, visions):
    step = len(actions)
    current = data[step]
    def history(key):
        return np.asarray([row[key] for row in data[:step]])
    return NS(episode=episode, task_id=episode.task_id, step=step,
              key_v0=current['vision_0'], key_v1=current['vision_1'], rs=current['robot_state'],
              raw_state=current['raw_state'], hist_key_v0=history('vision_0'),
              hist_key_v1=history('vision_1'), hist_rs=history('robot_state'),
              hist_a_exec=np.array(actions), hist_hit=np.array(hits, np.int8),
              hist_has_vision=np.array(visions, bool), prev_hit=bool(hits[-1]) if hits else None,
              prev_a_exec=actions[-1] if actions else None,
              blind_age=0 if not visions else next((j for j, v in enumerate(reversed(visions)) if v), len(visions)))
