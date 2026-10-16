"""Coordinator-only standard prepare; no model, worker or driver starts."""
from exp.offline_search.rounds.r17.filter_failed.stage import stage

if __name__=='__main__':
    import argparse
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True)
    stage(p.parse_args().run)
