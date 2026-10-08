"""Append a visible correction without altering either frozen prediction."""
from .d2_screen import *
stamp=now();dest=D2/'superseded_1822';dest.mkdir(exist_ok=True)
for name in ('REPORT.md','OFFLINE.md','SPEC.md','EXPERIMENTS.md','HANDBACK.md','PREDICTION.md'):
    p=dest/name
    if not p.exists():p.write_bytes((HERE/name).read_bytes())
for name in ('validation.json','evidence.json','document_generation.json'):
    p=dest/name
    if not p.exists():p.write_bytes((D2/name).read_bytes())
msg=f'''\n\n---\n\n## Source correction — {stamp}\n\n+The coordinator's new 13:0x CDT note, discovered during final handback review, retires `r12_devlooks_50S` and `r12_devlooks_200` and requests their `_local` replacements. **Withdraw the Spatial-50 and size-200 rows in the 18:22:46 forecast above.** Their source roots are no longer admissible validation evidence. L10-50, L10-400 and knob rows are unaffected. All preceding bytes are retained for the audit trail, not endorsed as current forecasts.\n\n+The candidate structure/thresholds remain frozen; no further selection will use the retired roots. The completed Spatial-50 local replacement will be evaluated with those same settings. Size-200 local data is pending and gets no forecast until available. Any new local-root forecast will be appended with its own timestamp. The original-root screens remain only as explicitly superseded historical artifacts in this directory.\n'''.replace('\n+','\n')
with (HERE/'PREDICTION.md').open('ab') as f:f.write(msg.encode())
dump(D2/'source_correction.json',dict(timestamp=stamp,retired=['r12_devlooks_50S','r12_devlooks_200'],valid_roots=VALID_ROOTS,
    predictor_unchanged=True,prediction_prefix_sha256=sha(dest/'PREDICTION.md'),prediction_sha256=sha(HERE/'PREDICTION.md')))
print(stamp,flush=True)
