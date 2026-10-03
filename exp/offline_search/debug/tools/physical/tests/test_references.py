from copy import deepcopy
import json
import numpy as np

from exp.offline_search.debug.tools.physical.audits import grasps, grasp_references
from exp.offline_search.debug.tools.physical.blind_drift import Library


def test_grasp_policy_reference_requires_same_reset_seed(physical_episode):
    ep = physical_episode
    ep.controls["eef_pos"][3] = [0.0, 0.0, 0.12]
    reference = deepcopy(ep)
    reference.meta.update(arm="policy", episode_key="policy_a1", success=True)
    rows = grasps(ep) + grasps(reference)
    _, summary = grasp_references(rows, ["policy"])
    assert rows[0]["reference_status"] == "available"
    assert rows[0]["reference_xyz_error"] == 0
    reference.meta["env_seed"] = 603
    rows = grasps(ep) + grasps(reference)
    grasp_references(rows, ["policy"])
    assert rows[0]["reference_status"] == "unavailable"


def test_backfill_library_fingerprint_cannot_be_substituted(physical_episode, tmp_path):
    ep = physical_episode
    ep.meta["backfill"] = dict(admission="PASS", library_rows=[0], lib_sha="wrong")
    p = tmp_path / "episode"
    p.mkdir()
    (p / "episode.json").write_text(json.dumps(ep.meta))
    np.savez_compressed(
        p / "controls_0000.npz",
        **{k: v for k, v in ep.controls.items() if not k.startswith("contact_")},
    )
    catalog = [dict(row=0, task_id=0, episode=0, step=0, next=-1, lib_sha="frozen")]
    lib = Library(catalog=catalog, backfill_root=tmp_path)
    assert not lib.physics
    assert "fingerprint" in lib.rejected_backfills[0]
    ep.meta["backfill"]["lib_sha"] = "frozen"
    (p / "episode.json").write_text(json.dumps(ep.meta))
    assert Library(catalog=catalog, backfill_root=tmp_path).physics


def test_unavailable_episode_card_has_honest_html_png(physical_episode, tmp_path):
    from exp.offline_search.debug.tools.physical.cards import card

    physical_episode.controls.pop("predicates")
    row = card(physical_episode, tmp_path)
    assert row["status"] == "unavailable"
    assert (tmp_path / row["png"]).is_file()
    assert "Unavailable" in (tmp_path / row["html"]).read_text()
