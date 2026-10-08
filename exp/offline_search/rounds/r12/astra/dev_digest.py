"""Print a reviewable numerical digest, without long per-episode records."""
from .dev_validation import *
out=[]
for p in sorted((DEST/'r12_devlooks_50L').glob('*/assessment.json')):
    a=json.loads(p.read_text());r=a['protocols']['propagated'];m=a['baseline']
    out.append(dict(cell=a['cell'],N=m['N'],V=m['V'],M=m['M'],SR=m['successes']/m['episodes'],
        half_share=r['half_share_pooled'],episode_half_share=r['half_share_episode_balanced'],halves=r['halves'],
        agreement=r['agreement'],guards=r['guards'],errors=r['shadow_policy_error'],
        costs=r['cost_diagnostic'],reasons=r['reasons'],eligibility=r['eligibility_gate_distributions'],
        reject=r['gate_rejections'],selected=r['selected_gate_distributions']))
dump(HERE/'dev_digest.json',out)
print(json.dumps(out,indent=2))
