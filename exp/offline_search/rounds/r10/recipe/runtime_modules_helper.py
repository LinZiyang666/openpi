import json, pickle, sys
from pathlib import Path
from exp.offline_search.rounds.r10.recipe.recipe import R10Recipe
with Path(sys.argv[1]).open('rb') as f: pickle.load(f)
root = Path(sys.argv[3]) / 'exp/offline_search'
files = sorted({str(Path(m.__file__).resolve()) for name, m in sys.modules.items()
                if name != '__main__' and (getattr(m, '__file__', '') or '').endswith('.py')
                and Path(m.__file__).resolve().is_relative_to(root)})
Path(sys.argv[2]).write_text(json.dumps(files, indent=2) + '\n')
