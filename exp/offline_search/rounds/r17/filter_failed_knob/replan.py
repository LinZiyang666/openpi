"""Finish local draft plans with the source's semantic policy-RNG namespace."""
from pathlib import Path
from dataclasses import asdict
import argparse

from .common import HERE, read, dump
from .build import cpu_plan, register_metadata, publish_study, freeze_bootstrap, direct_routes


def main():
    import yaml
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',action='append',help='Only this unlaunched campaign name')
    args=parser.parse_args()
    for job in read(HERE/'CAMPAIGNS.json')['jobs']:
        if args.run and job['name'] not in args.run:continue
        root=Path(job['run']);source=Path(job['source'])
        if (root/'LAUNCH.json').exists():raise ValueError('Cannot change a launched definition')
        campaign=read(root/'campaign.json');original=read(source/'campaign.json')
        campaign['cache_plan_name']=original.get('cache_plan_name',original['run_name'])
        direct_routes(campaign,root)
        for row in read(root/'assets/CACHE_STUDY.json')['groups']:
            register_metadata(source,yaml.safe_load((root/'assets'/row['file']).read_text()))
        publish_study(root,source,campaign,read(root/'assets/CACHE_STUDY.json'))
        freeze_bootstrap(root,campaign)
        _,model,tasks,_=cpu_plan(campaign)
        plan=read(root/'EPISODE_PLAN.json')
        plan.update(model=model,tasks=[asdict(t) for t in tasks],expected_episodes=len(tasks))
        dump(root/'campaign.json',campaign);dump(root/'EPISODE_PLAN.json',plan)
        assert {t.extra['campaign'] for t in tasks}=={campaign['cache_plan_name']}
        print(job['name'],len(tasks),campaign['cache_plan_name'],flush=True)


if __name__=='__main__':main()
