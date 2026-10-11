from exp.r17_integration.models.dp.samples.node_ram_preflight import evaluate


def test_old_cosmos_receipt_still_reserves_six_GiB_per_worker():
    gib = 1024**3
    fleets = [dict(workers=24, reserve_mib=4096, rss_bytes=91*gib, cosmos=True)]
    result = evaluate(251*gib, 139*gib, fleets, 12, 4096)
    assert result["other_pending_bytes"] == 53*gib
    assert result["admitted"]
    assert not evaluate(251*gib, 139*gib, fleets, 16, 4096)["admitted"]


def test_existing_fleet_RSS_is_not_double_counted():
    gib = 1024**3
    result = evaluate(251*gib, 139*gib, [dict(workers=24, reserve_mib=6144, rss_bytes=150*gib, cosmos=True)], 12, 4096)
    assert result["other_pending_bytes"] == 0 and result["admitted"]
