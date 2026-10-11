"""An outer hard timeout still leaves an explicit recoverable Q receipt."""
import json
from pathlib import Path
import sys
from .assets import write_json
run=Path(sys.argv[1]);run.mkdir(parents=True,exist_ok=True)
path=run/'receipt.json'
r=json.loads(path.read_text()) if path.exists() else dict(slot=sys.argv[2],owned_pids=[],gpu_executed=False)
r.update(status='PARTIAL',reason='outer 885-second TERM/5-second KILL deadline; inspect owned cleanup receipts',outer_timeout=True)
write_json(path,r)
print('PARTIAL '+str(path))
