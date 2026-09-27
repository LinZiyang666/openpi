"""Audit exact saved full-16 synthesized gripper values, not top-10 reconstructions."""
import json
import numpy as np
from diagnose import OUT, RESULTS

out = {}
for method in ['AWM_joint_cur_fcur', 'AWM_joint_cur_fcur_kr5', 'AWM_joint_cur_fbig', 'AWM_joint_big_fbig']:
    for suite in ['spatial', 'l10']:
        z = np.load(RESULTS / method / f'pi05_{suite}_cache.npz')
        mask = z['task_id'] == 6
        vote = np.abs(z['synth_seg'][:, 0, 6])
        out[f'{method}/{suite}'] = {
            'n_task6': int(mask.sum()), 'err_task6': float(z['err'][mask].mean()),
            **{f'vote{tag}_{group}': float((vote[m] < cut).mean())
               for tag, cut in [('05', .5), ('08', .8)]
               for group, m in [('task6', mask), ('all', np.ones(len(mask), bool))]}}
(OUT / 'exact_vote_audit.json').write_text(json.dumps(out, indent=2))
