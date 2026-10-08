"""Write the early coordinator request; constants below are planning inputs."""
from pathlib import Path
from datetime import datetime, timezone
import json

HERE = Path(__file__).resolve().parent
now = datetime.now(timezone.utc).isoformat()
plan = dict(timestamp=now, size50_cells=[f'{m}_{s}_50' for m in ('pi05', 'groot') for s in ('l10', 'spatial')],
            larger_cells=[f'{m}_l10_200' for m in ('pi05', 'groot')],
            doses=[.25, .32, .40, .45], calibration_episodes=100,
            validation_episodes=100, primary_candidates=['distance_feedback', 'periodic'],
            shadow_dose=.32)
(HERE / 'request_plan.json').write_text(json.dumps(plan, indent=2)+'\n')
text = f'''# Early B-dev requests

Written {now}, before examining new R13 variant results. These are requested coordinator jobs, not launches. Exact controller arithmetic and immutable arm files will follow in SPEC.md. Prioritize the controller implementation and its independent validation over more frozen-distance recordings.

The BRIEF recording set is useful for calibrating a fixed distance threshold and testing signal quality on knob-off states. It cannot validate a new feedback controller on its own induced states, and knob-off shadows alone cannot measure wasted calls on the new method.

1. **Independent size-50 validation:** all {len(plan['size50_cells'])} cells ({', '.join(plan['size50_cells'])}); `distance_feedback` and R11 `periodic`, each at {', '.join(str(x) for x in plan['doses'])}; {plan['validation_episodes']} B episodes per arm, identical selected pairs within each cell. This is {len(plan['size50_cells'])*len(plan['doses'])*len(plan['primary_candidates'])} arms. Use a new K=10/task selection disjoint from the coordinator's calibration K=10 set and from every older B init used in our calibration tables. Stratify by task; record the pair lists and hashes. Never call overlapping old inits independent validation. Look recordings on. Once the frozen algorithm passes, these data assess target delivery and engagement; SR does not select doses or cells.

2. **Size-200 calibration/validation:** {', '.join(plan['larger_cells'])}. First off + frozen R11 periodic/distance at .32, {plan['calibration_episodes']} paired B episodes per arm (six arms), with looks; reuse existing complete, correctly labeled recordings where possible. Then the same new distance/periodic validation doses as above, {plan['validation_episodes']} disjoint B episodes per arm (sixteen arms). Existing K=10 recording selections may supply calibration. Keep their union disjoint from validation. There is room for these sets in the stated complement, but the actual pair manifest must prove it.

3. **Shadow labels on active methods:** on the validation `.32` arms in the four size-50 cells, enable shadows for both methods on **every** look. Knob-off shadows in the coordinator's set are already requested. Also retain the corrected cache proposal before a MISS replaces it with the served policy action: normalized first ten controls, or enough immutable donor/metric/corrector state to reproduce it exactly. Record the cache proposal for guard calls too. A MISS shadow equals the served policy and is not by itself a policy-versus-cache label. Shadows must preserve actions/RNG and must not enter the executed IR ledger.

4. **Fixed-threshold checkpoint (conditional, lower priority):** after fitting one cell-wide threshold per target from calibration B only, run `distance_static_b` at .32 and .40 in all four size-50 cells, on the same independent validation selection, {plan['validation_episodes']} episodes per arm (eight arms). This distinguishes better calibration from online feedback. Skip this branch if replay already shows major path-dependent cost failure; retain its offline evidence. No repeated tuning on the validation set. SPEC will freeze this decision before validation.

Size 500 has no B complement: request no fictitious same-library dev run. Use a fully specified library-only initialization and label its A measurement a transfer test. Neither smaller-library validation nor replay is proof of closed-loop target delivery at size 500. If that transfer fails, report the failure without adjusting from A.

All arms of a GR00T model/suite/library cell, including controls, belong on one host. Reuse the same host for its calibration and validation when available. Paired original task/init identities and exact N/V/M ledgers are required. Preserve failed episodes and all attempted/accepted history distinctions.

Open dependency: the exact new recording root and selection manifests are awaited in ../DATA.md. This does not block CPU analysis of the explicitly allowed older B roots.
'''
(HERE / 'REQUESTS.md').write_text(text)
(HERE / 'STATUS.md').write_text(f'''# Status — {now}

Contract read in full. Early REQUESTS.md written. Reading the serving lifecycle and B-only evidence; no new R13 outcomes accessed. Working toward a frozen distance feedback controller, independent B pilot gates, and an exact paired grid. All analysis writes stay in this directory. CPU only; no launch, remote operation, or git command.

Pending: data inventory, B replay calibration, target/engagement tradeoff, freeze SPEC/PREDICTION/grid. Deep shadow-label analysis can continue after handback. Historical R11 report prose was read as contract context; no test-A raw data or test-derived tuning tables enter the new calibration. Do not infer new cell/dose choices from historical test outcomes.
''')
print(json.dumps(plan, indent=2))
