"""Q recipes and loads consume complete P2 benchmark contracts."""
from dataclasses import fields
import json
from pathlib import Path


def benchmark_spec(job):
    if job['domain']=='metaworld':
        from exp.r17_integration.benchmarks.metaworld import FORMAL_SPEC
        spec=FORMAL_SPEC
    else:
        from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec
        document=json.loads(Path(job['benchmark_spec']).read_text())
        names={f.name for f in fields(LiberoBenchmarkSpec)}
        spec=LiberoBenchmarkSpec(**{k:v for k,v in document.items() if k in names})
        assert document['protocol_identity']==spec.identity, 'Q LIBERO benchmark seal changed'
    from exp.r17_integration.common.benchmarks import register_benchmark
    identity=register_benchmark(spec)
    if job['domain']=='libero':
        if 'task_table_identity' in document:
            assert document['task_table_identity']==identity, 'Q LIBERO canonical table seal changed'
        if 'ordered_tasks' in document:
            assert tuple(document['ordered_tasks'])==tuple(spec.to_contract().ordered_tasks), 'Q LIBERO canonical task order changed'
    return spec


def recipe_table(protocol):
    from exp.r17_integration.common.benchmarks import get_benchmark,task_table_identity
    return dict(ordered_tasks=list(get_benchmark(protocol).ordered_tasks),
                task_table_identity=task_table_identity(protocol))


def recipe_settings(protocol):
    # Periodic scheduling is explicit; Q never infers IR or invents prices.
    return dict(recipe_table(protocol),periodic_gap=2,target_ir=None)
