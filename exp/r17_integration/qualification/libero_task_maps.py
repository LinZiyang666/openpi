"""Python 3.8 CPU-only export from the original installed LIBERO suite tables."""
import argparse
import json
from pathlib import Path


def main():
    from libero.libero import benchmark
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args()
    maps={}
    for name in ('libero_spatial','libero_object','libero_goal','libero_10'):
        suite=benchmark.get_benchmark_dict()[name]()
        rows=[]
        for i in range(suite.n_tasks):
            task=suite.get_task(i)
            rows.append(dict(task_id=i,name=task.name,language=task.language))
        assert len(rows)==10
        maps[name]=dict(suite=name,tasks=rows)
    Path(a.output).write_text(json.dumps(maps,sort_keys=True,indent=2)+'\n')
    print('PASS 40 official LIBERO tasks exported without environment/model construction')


if __name__=='__main__':main()
