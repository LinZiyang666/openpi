"""Read existing counts and compute explicitly assumption-based collection precision. No rollouts."""
import json,math,pathlib
O=pathlib.Path(__file__).resolve().parent
s=json.loads((O/'source_index.json').read_text());anchors=[]
for label in s['A']:
 a=s['arms'][label]
 if a['lib'] not in ('50','500'):continue
 v=a['V'];anchors.append(dict(label=label,cell=a['cell'],library=a['lib'],episodes=a['n'],anchors=v,mean_anchors=v/a['n'],raw_key_GiB=v*2*32768*4/1024**3))
z=1.959963984540054
out={'source':'source_index.json and formula-only design calculations; no new experiments','anchors':anchors,
 'binomial_mean_n':[dict(p_assumed=p,halfwidth=h,n=math.ceil(z*z*p*(1-p)/(h*h))) for p in [.5,.9] for h in [.05,.03,.025]],
 'paired_halfwidths':[dict(n=n,discordance_assumed=q,halfwidth=z*math.sqrt(q/n)) for n in [50,200,500,1000] for q in [.1,.2,.3]],
 'balanced_difference_n':[dict(halfwidth=h,n=math.ceil(z*z/h**2)) for h in [.1,.05,.03]],
 'zero_event_upper':[dict(n=n,one_sided_95_upper=1-.05**(1/n)) for n in [100,300,500]],
 'correlation_plan':[dict(rho_assumed=r,n_independent_units_80pct_power_approx=math.ceil((z+.8416212335729143)**2/math.atanh(r)**2+3)) for r in [.3,.4,.5]],
 'rare_failure_example':dict(target_failures=100,assumed_failure_rate=.018,expected_episode_budget_ceiling=math.ceil(100/.018)),
 'extra_inference_examples':[dict(label=a['label'],two_extra_policy_draws_on_200_episodes_calls=400,additional_shadow_call_fraction=400/a['anchors'],one_extra_vision_on_200_episodes_calls=200,additional_anchor_vision_fraction=200/a['anchors']) for a in anchors]}
(O/'data_wishlist_precision.json').write_text(json.dumps(out,indent=2))
print('Wrote assumption-based precision and storage calculations for',len(anchors),'existing endpoint arms.')
