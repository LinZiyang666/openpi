import sys, json
sys.path[:0] = ["/home/weiland/projects/openpi", "/home/weiland/projects/openpi/src"]
from pathlib import Path
Q = Path("/home/weiland/projects/openpi/exp/r17_integration/qualification")
doc = json.loads((Q / "specs/libero_benchmark_spec.json").read_text())
print("spec keys:", sorted(k for k in doc if k in ("protocol_identity", "task_table_identity", "wire_size", "seed_mode", "replan_steps")),
      "wire", doc.get("wire_size"), "replan", doc.get("replan_steps"), "seed_mode", doc.get("seed_mode"))
from exp.r17_integration.models.dp.protocols import libero_spec_from_document
try:
    s = libero_spec_from_document(doc, replan_steps=5); print("DP parse:", s.identity == doc.get("protocol_identity"))
except Exception as e:
    print("DP parse:", type(e).__name__, e)
from exp.r17_integration.models.fastwam.loader import LoadSpec
from exp.r17_integration.models.fastwam.protocols import validate_benchmark_specs
try:
    r = validate_benchmark_specs(LoadSpec("libero", "/data/mw_finetune/models/fastwam_libero", "/data/xwam/wan22_5b",
        benchmark_protocols=(doc["protocol_identity"],), benchmark_specs=(doc,)))
    print("Fast-WAM validate:", [x.identity == doc["protocol_identity"] for x in r])
except Exception as e:
    print("Fast-WAM validate:", type(e).__name__, e)
obs = Path("/data/mw_finetune/qualification/observations/libero_original256_fix2_v1")
if obs.is_dir():
    sys.path.insert(0, str(Q.parents[2]))
    from exp.r17_integration.qualification.recorded_libero import validate
    validate(obs); print("Q recorded observations validate: PASS")
else:
    print("Q recorded observations directory not present yet:", obs)
for name in ("asset_dp_libero.json", "asset_fastwam_libero.json"):
    j = json.loads((Q / "specs" / name).read_text()); print(name, {k: j[k] for k in j if "spec" in k or "protocol" in k})
