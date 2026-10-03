"""Read-only smoke/usefulness audit of built profiles on discovery episodes."""
import gc
import importlib
import json
import os
from pathlib import Path
import time
import traceback
from types import SimpleNamespace
from concurrent.futures import ThreadPoolExecutor

from exp.offline_search.debug import reader
from exp.offline_search.debug.tools.decision import common as C

ROOT = Path('/home/weiland/trace_runs/os_closed_loop/r08_main')
OUT = Path('/tmp/r8cb_E5_debug_architecture/profiles')


def discovery(name, init0=False):
    arm = reader.open_arm(ROOT, name)
    arm.cache_enabled = False
    original_decisions, original_episodes = arm.decisions, arm.episodes
    def decisions(*args, **kwargs):
        frame = original_decisions(*args, **kwargs)
        return frame.loc[frame.init.eq(0) if init0 else frame.init.lt(30)].copy()
    def episodes(*args, **kwargs):
        frame = original_episodes(*args, **kwargs)
        return frame.loc[frame.init.eq(0) if init0 else frame.init.lt(30)].copy()
    arm.decisions, arm.episodes = decisions, episodes
    return arm


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    specs = json.loads((ROOT/'arms.json').read_text())
    plans = []
    for s in specs:
        if s['r8']['variant'] == 'A':
            plans.append((s['arm'], ['provenance', 'stage_ledger'], False))
    for name in ['r8_groot_l10_50_CU', 'r8_groot_l10_50_CT', 'r8_groot_l10_50_IP',
                 'r8_pi05_l10_50_IP', 'r8_groot_l10_50_FL']:
        plans.append((name, ['exposure_hazard' if name.endswith('_FL') else 'call_value'], False))
    for name in ['r8_groot_l10_50_A', 'r8_groot_l10_500_A']:
        plans.append((name, ['divergence', 'follow_vs_look'], True))
    args = SimpleNamespace(bootstraps=100, seed=0, onsets=None, triggers=None)
    def one(plan):
        name, modules, init0 = plan
        arm = discovery(name, init0)
        for tool in modules:
            dest = OUT/tool/(name+('_init0' if init0 else '_discovery'))
            if (dest/(tool+'.json')).exists():
                continue
            start = time.time()
            print(json.dumps(dict(ev='start', arm=name, tool=tool, init0=init0)), flush=True)
            try:
                report = importlib.import_module('exp.offline_search.debug.tools.decision.'+tool).analyze(arm, args)
                report['stage_coverage'] = getattr(arm, 'profile_stage_coverage', {})
                report['input_diagnostics'] = arm.read_issues
                report['audit_split'] = 'init=0' if init0 else 'init=0..29'
                report['reader_cache_enabled'] = arm.cache_enabled
            except Exception:
                report = dict(status='AUDIT_ERROR', error=traceback.format_exc(), tables={})
            C.write_report(tool, dest, {name:report}, time.time()-start)
            print(json.dumps(dict(ev='done', tool=tool, arm=name, status=report.get('status'),
                coverage=report.get('coverage'), error=report.get('error'), seconds=time.time()-start)), flush=True)
            del report
            gc.collect()
        del arm
        gc.collect()
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(one, plans))
    os.environ['MPLCONFIGDIR'] = str(OUT/'matplotlib')
    from exp.offline_search.debug.tools.physical.cards import card
    from exp.offline_search.debug.tools.physical.common import Episode
    cards = []
    for name in ('r8_groot_l10_50_A','r8_pi05_spatial_500_A'):
        arm = discovery(name, True)
        meta = arm.episodes().loc[lambda f:f.task_id.eq(0)].iloc[0].to_dict()
        ek = meta['episode_key']
        meta.update(reader.read_json(arm.debug_dir/'client'/ek/'episode.json'))
        meta['arm'] = name
        ds = arm.decisions().loc[lambda f:f.episode_key.eq(ek)].to_dict('records')
        episode = Episode(meta=meta, controls=arm.controls(ek), decisions=ds,
                          manifest=arm.manifest, reader_arm=arm)
        cards.append(card(episode, OUT/'cards'))
    (OUT/'cards/results.json').write_text(json.dumps(C.clean(cards),indent=2)+'\n')


if __name__ == '__main__':
    main()
