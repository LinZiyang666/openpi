import sys, json
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
from exp.r17_integration.benchmarks.libero_bridge import LiberoBenchmarkSpec, SUITES
from exp.r17_integration.models.fastwam.loader import LoadSpec
from exp.r17_integration.models.fastwam.protocols import validate_benchmark_specs
maps = {s: {"suite": s, "tasks": [{"task_id": i, "name": f"task_{i}", "language": f"{s} instruction {i}"} for i in range(10)]} for s in SUITES}
system = LiberoBenchmarkSpec.from_task_maps(maps, wire_size=256)
doc = json.loads(json.dumps(system.manifest()))
print("manifest keys with table identity:", "task_table_identity" in doc, "ordered_tasks" in doc)
spec = LoadSpec("libero", "/data/mw_finetune/models/fastwam_libero", "/data/xwam/wan22_5b",
                benchmark_protocols=(system.identity,), benchmark_specs=(doc,))
print("fast-wam validate:", [s.identity == system.identity for s in validate_benchmark_specs(spec)])
print("P2 task_table_identity attr:", getattr(system, "task_table_identity", None) is not None)
