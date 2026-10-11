"""Red/green witness for the round-3 fixes: reintroduce each reviewed mechanism IN MEMORY
(no file edit) and show its new regression fails; then run unmodified (green)."""
import subprocess, sys, json
PRE = {
 "gr_lock_hook_removed": ("exp/r17_integration/tests/models/groot/test_round3_review_a.py::test_round3_native_model_lock_wait_is_separated_from_service",
   "import exp.r17_integration.models.groot.provider as m; del m.GrootMetaWorldAdapter.drain_queue_wait_ms"),
 "gr_default_table_removed": ("exp/r17_integration/tests/models/groot/test_round3_review_a.py::test_round3_default_config_carries_canonical_table_and_rejects_mismatched_task",
   "import exp.r17_integration.models.groot.serve as s; from exp.r17_integration.common.config import RecipeConfig; o=s.default_config\n"
   "def d(*a,**k):\n c=o(*a,**k); c.recipe=RecipeConfig(); return c\n"
   "s.default_config=d"),
 "analyzer_prices_host_wall": ("exp/r17_integration/tests/search/test_round3_review_a.py::test_round3_pricing_sums_service_time_not_lock_queue_wall",
   # Old analyzer: pricing total = sum(host-wall cost_ms) for any pricing_eligible row.
   "import exp.r17_integration.search.analyze as a; o=a.merge_decision_ledgers\n"
   "def m(*x,**k):\n rows=o(*x,**k)\n for r in rows: r['service_ms']=r['cost_ms']\n return rows\n"
   "a.merge_decision_ledgers=m"),
}
results = {}
for name, (test, patch) in PRE.items():
    for mode in ("red", "green"):
        code = (patch + "\n" if mode == "red" else "") + f"import pytest,sys; sys.exit(pytest.main([{test!r},'-q','-p','no:cacheprovider','--basetemp=/tmp/r17_review_A3/mut_{name}_{mode}']))"
        rc = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True).returncode
        results[f"{name}:{mode}"] = rc
print(json.dumps(results, indent=1))
