"""Offline planning precision for the owner's superset-profiler data request.

No experiment is launched. Estimates are conditional normal approximations;
they are not power guarantees for a new controller, task set, or seed block.
"""
from pathlib import Path
import json, math
import numpy as np
from scipy.stats import norm

O = Path(__file__).resolve().parent
arms = {r['arm']: r for r in json.loads((O / 'arms.json').read_text())}
keys = json.loads((O / 'key_contrasts.json').read_text())
z95 = float(norm.ppf(.975))
z_one = float(norm.ppf(.95))
z80 = float(norm.ppf(.8))

def outcomes(name):
    return {(e['task'], e['init']): e for e in arms[name]['episodes']}

def needed(variance, half_width):
    return math.ceil(z95**2 * variance / half_width**2)

measured = []
for row in keys:
    a, b = outcomes(row['A']), outcomes(row['B'])
    common = sorted(set(a) & set(b))
    delta = np.array([b[k]['Y'] - a[k]['Y'] for k in common], float)
    dm = np.array([b[k]['M'] - a[k]['M'] for k in common], float)
    v = float(delta.var(ddof=1))
    measured.append(dict(
        cell=row['cell'], library=row['lib'], A=row['A'], B=row['B'],
        paired_inits=len(common), discordance=float(np.mean(delta != 0)),
        observed_delta=float(delta.mean()), observed_variance=v,
        approximate_95_halfwidth_at_500=z95 * math.sqrt(v / 500),
        pairs_for_95_halfwidth_02=needed(v, .02),
        pairs_for_95_halfwidth_05=needed(v, .05),
        pairs_for_80_power_detect_03=math.ceil((z95 + z80)**2 * v / .03**2),
        miss_difference_variance=float(dm.var(ddof=1)),
        pairs_for_mean_M_difference_halfwidth_1=needed(float(dm.var(ddof=1)), 1),
    ))

planning = []
for variance in [.1, .2, .3]:
    planning.append(dict(
        assumed_paired_variance=variance,
        halfwidth_500=z95*math.sqrt(variance/500),
        halfwidth_50=z95*math.sqrt(variance/50),
        pairs_for_halfwidth_02=needed(variance,.02),
        pairs_for_halfwidth_05=needed(variance,.05),
        pairs_for_80_power_detect_03=math.ceil((z95+z80)**2*variance/.03**2),
        pairs_for_80_power_noninferiority_02_true_gap_zero=math.ceil((z_one+z80)**2*variance/.02**2),
    ))

other = dict(
    independent_arms_worst_case_n_each_halfwidth_02=needed(.5,.02),
    independent_arms_p09_n_each_halfwidth_02=needed(2*.9*.1,.02),
    no_cache_miss_path_probability_at_p025_30anchors=.75**30,
    all_call_path_probability_at_p025_30anchors=.25**30,
    binomial_task_frequency_n_halfwidth_02=needed(.25,.02),
    binomial_task_frequency_n_halfwidth_01=needed(.25,.01),
    per_anchor_raw_two_camera_float32_key_bytes=2*32768*4,
    raw_keys_MiB_per_episode_at_30anchors=30*2*32768*4/2**20,
    five_rate_eight_cell_500_episode_screen=5*8*500,
    four_rate_two_cell_500_episode_screen=4*2*500,
    extra_shadow_fraction_if_repeat_one_in_20_anchors=1/20,
    rare_unrecoverable_event_n_for_50_events_at_rate_01=50/.01,
    zero_events_one_sided_95_upper_at_n500=1-.05**(1/500),
    zero_events_one_sided_95_upper_at_n50=1-.05**(1/50),
)
output=dict(
    method='Normal plug-in planning; paired-init variance for historical same-task contrasts; no multiplicity/design-effect adjustment',
    constants=dict(two_sided_95_z=z95,one_sided_95_z=z_one,power_80_z=z80),
    measured=measured,planning=planning,other=other,
)
(O/'data_needs_precision.json').write_text(json.dumps(output,indent=1)+'\n')
print(json.dumps(output,indent=1))
