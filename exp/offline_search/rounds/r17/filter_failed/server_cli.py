"""Preserve the unfiltered local DP R4 admission bridge over its frozen model.

The original R4 runs used formal_server.dp_capacity_serve (as recorded in
CACHE_CAPACITY_ADMISSION.json). Their archived model's admission still expects
R2. This bootstrap repeats that admission-only R2 presentation; actual residents,
batching, sampler, Cache recipe, CUDA graph and model code remain frozen.
"""


def bridge(module):
    native=module.cache_study_admission
    def admission(campaign,*args):
        # A longseg_1017.pod_copy job serves the source's frozen model, batching and Cache recipe on one MIG
        # replica without MPS; admission sees the 4090 placement (server node, MPS) the job was copied from.
        source=(campaign.get('pod_copy') or {}).get('resources_before')
        if source:campaign=dict(campaign,server_node=source['server_node'],mps=source['mps'])
        return native(dict(campaign,replicas=2),*args)
    module.cache_study_admission=admission


def run(argv,module,serve):
    import argparse
    import json
    from pathlib import Path
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--campaign',required=True)
    p.add_argument('--port',required=True,type=int)
    p.add_argument('--directory',required=True)
    p.add_argument('--cuda-graph',action='store_true')
    args=p.parse_args(argv)
    bridge(module)
    return serve(json.loads(Path(args.campaign).read_text()),args.port,args.directory,args.cuda_graph)


if __name__=='__main__':
    import sys
    from exp.r17_integration.models.dp import formal_batch_server
    from exp.r17_integration.ops import formal_server
    run(sys.argv[1:],formal_batch_server,formal_server.serve)
