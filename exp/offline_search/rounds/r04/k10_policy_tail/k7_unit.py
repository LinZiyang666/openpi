"""Run original K7 checks with output paths relocated to K10."""
from pathlib import Path
import sys
from exp.offline_search.rounds.r04.k7_guard import evidence
BASE=Path(__file__).resolve().parent
out=BASE/'results/installed/k7_unit';(out/'results').mkdir(parents=True,exist_ok=True)
evidence.HERE=out
mode=sys.argv[1]
if mode=='edges':
    from exp.offline_search.rounds.r04.k7_guard import edge_checks
    edge_checks.main()
elif mode=='parity':evidence.parity(sys.argv[2],int(sys.argv[3]),20)
elif mode=='rates':evidence.rates(sys.argv[2],int(sys.argv[3]))
