import json
from pathlib import Path
import subprocess
import sys


def test_default_off_matches_head_config_errors_metadata_ack_and_action_bytes():
    path=Path(__file__).with_name("legacy_probe.py")
    baseline=subprocess.run([sys.executable,str(path),"--baseline"],text=True,capture_output=True,check=True,timeout=40)
    current=subprocess.run([sys.executable,str(path)],text=True,capture_output=True,check=True,timeout=40)
    assert json.loads(current.stdout.splitlines()[-1])==json.loads(baseline.stdout.splitlines()[-1])


def test_importing_opt_in_modules_does_not_patch_stock_factories_or_wire_bytes():
    path=Path(__file__).with_name("legacy_probe.py")
    baseline=subprocess.run([sys.executable,str(path),"--baseline"],text=True,capture_output=True,check=True,timeout=40)
    code = ("import runpy; import exp.r17_integration.common.config; "
            "import exp.r17_integration.common.native; import exp.r17_integration.common.websocket_policy_server; "
            "runpy.run_path("+repr(str(path))+", run_name='__main__')")
    imported=subprocess.run([sys.executable,"-c",code],text=True,capture_output=True,check=True,timeout=40)
    assert json.loads(imported.stdout.splitlines()[-1])==json.loads(baseline.stdout.splitlines()[-1])
