"""P1-owned cross-package G1 witness, without provenance monkeypatches."""


def test_pi05_native_ws_conductor_capture_with_complete_probe_provenance(tmp_path):
    from exp.r17_integration.tests.models.pi05 import test_pi05_e2e as witness
    witness.test_first_pi_mw_real_ws_two_clients_two_yamls_conductor_and_capture(tmp_path)
