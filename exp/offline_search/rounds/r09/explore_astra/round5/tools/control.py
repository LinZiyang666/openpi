"""Existing control CLI with an eval-only metadata store supplied to its planner.

The stock planner hashes the raw multi-init episodes.json. This adapter avoids
opening it: the identity-only, 20–29 store is prepared by build.py. The library
assets and the control implementation are unchanged. No implicit remote action.
"""
from functools import partial
from pathlib import Path
from exp.offline_search.closed_loop.ops.h100 import assets, control
from .build import SAFE_STORE

ORIGINAL_REMAP=assets.remap


def isolated_remap(value, run, store=assets.STORE, base=assets.BASE):
    # Give this eval-only store its own remote namespace. The concurrent sync
    # must never replace the fleet's shared (possibly mixed-init) identity map.
    if isinstance(value, (str,Path)):
        text=str(value)
        prefix=str(SAFE_STORE)
        if text==prefix or text.startswith(prefix+'/'):
            result=str(Path(base)/'runs'/Path(run).name/'serving_store')+text[len(prefix):]
            return Path(result) if isinstance(value,Path) else result
    return ORIGINAL_REMAP(value,run,store,base)


def main():
    assets.remap=isolated_remap
    control.build_plan=partial(assets.build_plan,store=SAFE_STORE)
    control.main()


if __name__=='__main__':main()
