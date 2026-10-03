"""Inspect serving imports in an otherwise clean process."""
import json
import pickle
import sys
from pathlib import Path
with Path(sys.argv[1]).open('rb') as f: method=pickle.load(f)['method']
assert not any(n.startswith(('exp.offline_search.rounds.r11.opus','exp.offline_search.rounds.r11.astra')) for n in sys.modules)
root=Path(sys.argv[2])
files=sorted({str(Path(m.__file__).resolve()) for n,m in sys.modules.items()
    if n!='__main__' and (getattr(m,'__file__','') or '').endswith('.py')
    and Path(m.__file__).resolve().is_relative_to(root/'exp/offline_search')})
Path(sys.argv[3]).write_text(json.dumps(dict(PASS=True,files=files,no_exploration_imports=True),indent=1)+'\n')
