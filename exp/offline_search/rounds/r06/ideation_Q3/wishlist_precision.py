"""Offline planning arithmetic for the Q3 profiler wishlist; not a profiler."""
import json
import math
from pathlib import Path
from statistics import NormalDist

OUT = Path(__file__).parent
z = NormalDist().inv_cdf(.975)
rows = []
for h in (.05, .03, .02, .01):
    rows.append({
        'half_width': h,
        'independent_opportunities_sr085_p05': math.ceil(z*z*.85*.15/(.5*.5)/h**2),
        'independent_opportunities_worst_p05': math.ceil(z*z*.25/(.5*.5)/h**2),
        'paired_branch_states_q010': math.ceil(z*z*.10/h**2),
        'paired_branch_states_q020': math.ceil(z*z*.20/h**2),
    })
event = json.loads((OUT/'event_probe.json').read_text())
reach = {}
for cell in ('pi05_l10_50', 'pi05_l10_500'):
    f = event[cell]['before_B_first_miss']/event[cell]['episodes']
    reach[cell] = {
        'historical_frozen_path_reach': f,
        'episodes_for_5pp_precision_sr085': math.ceil(rows[0]['independent_opportunities_sr085_p05']/f),
        'episodes_for_3pp_precision_sr085': math.ceil(rows[1]['independent_opportunities_sr085_p05']/f),
    }
out = {
    'purpose': 'planning calculations; not measured profiler performance or power guarantees',
    'z975': z,
    'formulas': {
        'unpaired': 'ceil(z^2*s*(1-s)/(p*(1-p)*h^2)); equal endpoint variances; one independent selected opportunity per episode',
        'paired_branch': 'ceil(z^2*q/h^2); conservative near-zero mean paired effect; q is assumed endpoint discordance',
    },
    'precision': rows,
    'exposure_sensitivity': reach,
    'propensity_01_variance_multiplier_over_half': (.5*.5)/(.1*.9),
    'zero_event_upper95_n500': 1-.05**(1/500),
    'zero_event_upper95_n3000': 1-.05**(1/3000),
    'rare_stratum_5percent_200episodes_total': math.ceil(200/.05),
    'rare_stratum_1percent_200episodes_total': math.ceil(200/.01),
    'current_schema_float32_vision_keys_bytes_per_anchor': 2*32768*4,
    'current_schema_raw_rgb_bytes_per_anchor': 2*224*224*3,
    'independent_500_effect_halfwidth_sr085': z*math.sqrt(.85*.15/(.5*.5*500)),
    'independent_500_effect_halfwidth_worst': z*math.sqrt(.25/(.5*.5*500)),
    'three_policy_samples_extra_inferences_for_100_states': (3-1)*100,
    'initial_cells': 8,
    'initial_episodes_per_cell': 500,
    'initial_total_episodes': 8*500,
    'replicate_block_design': 3,
}
(OUT/'wishlist_precision.json').write_text(json.dumps(out, indent=2)+'\n')
print(json.dumps(out, indent=2))
