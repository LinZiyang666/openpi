#!/usr/bin/env bash
# Print reviewed commands only. No prepare/launch runs from this script.
set -euo pipefail
python - <<'PY'
from pathlib import Path
s=Path('exp/offline_search/rounds/r17/autoscale_1017/FORMAL_ACCEPTANCE.md').read_text()
print(s.split('```bash\n',1)[1].split('```',1)[0])
PY
