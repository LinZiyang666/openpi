"""CPU test for exp/dp_nfe/analysis/diagnostics_table.py: per-file rows carry the cell identity and the four means, rows
are seed-averaged per (task, modality, variant, head, sampler) with min/max, files without a matrix identity are ignored
(not silently dropped), and the Markdown table has one line per group."""

import json

from exp.dp_nfe.analysis import diagnostics_table as T


def _diag(path, task, variant, head, seed, k, disp, bic):
    path.write_text(json.dumps({"cell": {"cell_id": f"{task}_lowdim_{variant}_{head}_s{seed}_B100k", "task_name": task, "modality": "lowdim",
                                         "variant": variant, "head": head, "train_seed": seed},
                                "sampler": {"sampler": "ddim", "k": k}, "n_histories": 4, "n_samples": 8, "std_source": "frozen",
                                "mean_dispersion": disp, "mean_delta_bic": bic, "mean_frac_out_of_range_unclipped": 0.5, "mean_nearest_demo": 1.0}))


def test_rows_groups_ignored_and_markdown(tmp_path):
    a = tmp_path / "a"; b = tmp_path / "b"; a.mkdir(); b.mkdir()
    _diag(a / "pusht_lowdim_U_epsilon_s42_B100k_ddim_1.json", "pusht", "U", "epsilon", 42, 1, 8.0, 100.0)
    _diag(b / "pusht_lowdim_U_epsilon_s43_B100k_ddim_1.json", "pusht", "U", "epsilon", 43, 1, 10.0, 50.0)
    _diag(a / "pusht_lowdim_U_sample_s42_B100k_ddim_1.json", "pusht", "U", "sample", 42, 1, 0.0, -20.0)
    (a / "official_square_image_p0diag.json").write_text(json.dumps({"cell": {"variant": "official"}, "mean_dispersion": 1.0}))
    (b / "broken.json").write_text("{not json")
    loaded = T.load([a, b])
    assert len(loaded["rows"]) == 3 and {i["reason"].split(":")[0] for i in loaded["ignored"]} == {"no matrix cell identity", "unreadable"}
    summary = T.summarise(loaded["rows"])
    eps = next(r for r in summary if r["head"] == "epsilon")
    assert eps["n_seeds"] == 2 and eps["seeds"] == [42, 43] and eps["sampler"] == "ddim_1"
    assert eps["mean_dispersion"] == {"mean": 9.0, "min": 8.0, "max": 10.0} and eps["mean_delta_bic"]["mean"] == 75.0
    md = T.markdown(summary)
    assert md.count("\n") == 2 + len(summary) and "| pusht | lowdim | U | epsilon | ddim_1 | 2 | 9.00 [8.00, 10.00] |" in md
