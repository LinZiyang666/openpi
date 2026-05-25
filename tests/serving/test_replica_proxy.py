"""Unit tests for the replica router's pure helpers (no GPU / no sockets)."""

from __future__ import annotations

import asyncio

from openpi_client import msgpack_numpy
from websockets.asyncio.client import connect as ws_connect
from websockets.asyncio.server import serve as ws_serve

from openpi.serving.replica_proxy import ReplicaProxy
from openpi.serving.replica_proxy import classify
from openpi.serving.replica_proxy import merge_metrics
from openpi.serving.replica_proxy import merge_throughput_summary

_pack = msgpack_numpy.Packer().pack
_unpack = msgpack_numpy.unpackb


def test_classify_infer_obs_is_sticky():
    # A raw inference obs has no __ctrl__ key.
    frame = _pack({"state": [0.0, 1.0], "image": {"cam": [[1, 2], [3, 4]]}})
    assert classify(frame) == ("sticky", None)


def test_classify_per_connection_ctrls_are_sticky():
    for ctrl in ("episode_start", "episode_end", "select_bundle",
                 "prefill_trajectory", "reset", "fetch_dump"):
        kind, name = classify(_pack({"__ctrl__": ctrl}))
        assert kind == "sticky", ctrl
        assert name == ctrl


def test_classify_broadcast_ctrls():
    for ctrl in ("load_cache_config", "preload_normalizer_buffer",
                 "unload_warmup_buffer", "set_batch_params",
                 "mem_history_start", "mem_history_stop"):
        assert classify(_pack({"__ctrl__": ctrl})) == ("broadcast", ctrl)


def test_classify_aggregate_ctrls():
    for ctrl in ("dump_metrics", "throughput_summary", "snapshot_mem"):
        assert classify(_pack({"__ctrl__": ctrl})) == ("aggregate", ctrl)


def test_classify_non_dict_and_garbage_are_sticky():
    assert classify(_pack([1, 2, 3])) == ("sticky", None)
    assert classify(_pack(42)) == ("sticky", None)
    assert classify(b"\xff\xff not msgpack \x00") == ("sticky", None)


def test_merge_metrics_empty():
    assert merge_metrics([]) == {}


def test_merge_metrics_concatenates_and_tags_replica():
    r0 = {
        "batch": [{"stage": 1, "size": 8}, {"stage": 3, "size": 5}],
        "util": [{"gpu_util_pct": 30}],
        "scalar": "keep-first",
    }
    r1 = {
        "batch": [{"stage": 1, "size": 8}],
        "util": [{"gpu_util_pct": 34}],
        "scalar": "ignored",
    }
    merged = merge_metrics([r0, r1])

    # list keys concatenated across replicas
    assert len(merged["batch"]) == 3
    assert len(merged["util"]) == 2
    # each record tagged with its source replica index
    replicas = sorted(rec["replica"] for rec in merged["batch"])
    assert replicas == [0, 0, 1]
    assert {rec["replica"] for rec in merged["util"]} == {0, 1}
    # original fields preserved
    assert {rec["size"] for rec in merged["batch"]} == {8, 5}
    # scalar key keeps the first replica's value
    assert merged["scalar"] == "keep-first"


def test_merge_metrics_handles_missing_key_in_some_replicas():
    r0 = {"batch": [{"stage": 1}], "mem": [{"alloc_mb": 100}]}
    r1 = {"batch": [{"stage": 2}]}  # no "mem"
    merged = merge_metrics([r0, r1])
    assert len(merged["batch"]) == 2
    assert len(merged["mem"]) == 1
    assert merged["mem"][0]["replica"] == 0


# --- end-to-end async wiring against fake (no-GPU) backends -----------------


async def _start_fake_backend(*, fail_lcc: bool = False):
    """A stand-in serve_policy backend: greets with metadata, then replies per
    ctrl. Tracks how many load_cache_config calls it received (broadcast check).
    ``fail_lcc=True`` makes it return an ``{"__ack__":"error"}`` payload on
    load_cache_config (to exercise partial-broadcast-failure handling)."""
    holder = {"port": None, "lcc": 0}

    async def handler(ws):
        await ws.send(_pack({"__meta__": True, "port": holder["port"]}))
        async for frame in ws:
            obj = _unpack(frame)
            ctrl = obj.get("__ctrl__") if isinstance(obj, dict) else None
            if ctrl == "dump_metrics":
                await ws.send(_pack({
                    "batch": [{"stage": 1, "size": 8}],
                    "util": [{"gpu_util_pct": holder["port"] % 100}],
                }))
            elif ctrl == "load_cache_config":
                holder["lcc"] += 1
                if fail_lcc:
                    await ws.send(_pack({"__ack__": "error", "msg": "simulated load failure"}))
                else:
                    await ws.send(_pack({"__ack__": "load_cache_config", "port": holder["port"]}))
            else:  # infer / per-connection ctrl -> echo + tag serving backend
                await ws.send(_pack({"echo": obj, "port": holder["port"]}))

    server = await ws_serve(handler, "127.0.0.1", 0, max_size=None, compression=None)
    holder["port"] = server.sockets[0].getsockname()[1]
    return server, holder


async def _scenario():
    s0, h0 = await _start_fake_backend()
    s1, h1 = await _start_fake_backend()
    proxy = ReplicaProxy("127.0.0.1", [h0["port"], h1["port"]])
    await proxy.prime_metadata()
    pserver = await ws_serve(proxy.handle, "127.0.0.1", 0, max_size=None, compression=None)
    uri = f"ws://127.0.0.1:{pserver.sockets[0].getsockname()[1]}"

    try:
        # A) one connection is pinned to a single backend for all its requests
        async with ws_connect(uri, max_size=None, compression=None) as ca:
            await ca.recv()  # metadata greeting
            ports = []
            for i in range(4):
                await ca.send(_pack({"state": [i]}))
                ports.append(_unpack(await ca.recv())["port"])
            assert len(set(ports)) == 1, f"not sticky: {ports}"
            a_port = ports[0]

            # B) a second concurrent connection lands on the OTHER backend
            #    (least-connections, since A still holds one)
            async with ws_connect(uri, max_size=None, compression=None) as cb:
                await cb.recv()
                await cb.send(_pack({"state": [0]}))
                b_port = _unpack(await cb.recv())["port"]
                assert b_port != a_port, f"least-conn failed: {a_port}=={b_port}"

        # C) dump_metrics is fanned out and merged into whole-host numbers
        async with ws_connect(uri, max_size=None, compression=None) as cc:
            await cc.recv()
            await cc.send(_pack({"__ctrl__": "dump_metrics", "clear": True}))
            merged = _unpack(await cc.recv())
            assert len(merged["batch"]) == 2, merged["batch"]
            assert {r["replica"] for r in merged["batch"]} == {0, 1}
            assert len(merged["util"]) == 2

        # D) load_cache_config is broadcast to EVERY backend
        async with ws_connect(uri, max_size=None, compression=None) as cd:
            await cd.recv()
            await cd.send(_pack({"__ctrl__": "load_cache_config", "yaml_content": "x"}))
            ack = _unpack(await cd.recv())
            assert ack["__ack__"] == "load_cache_config"
        assert h0["lcc"] == 1 and h1["lcc"] == 1, (h0["lcc"], h1["lcc"])
    finally:
        pserver.close()
        s0.close()
        s1.close()
        await pserver.wait_closed()
        await s0.wait_closed()
        await s1.wait_closed()


def test_proxy_end_to_end_routing():
    asyncio.run(_scenario())


# --- failure-mode coverage --------------------------------------------------


async def _scenario_backend_down():
    """A sticky connection whose backend died gets a text-frame error (so
    WebsocketClientPolicy.infer raises) instead of hanging in recv()."""
    s0, h0 = await _start_fake_backend()
    proxy = ReplicaProxy("127.0.0.1", [h0["port"]])
    await proxy.prime_metadata()  # cache the greeting while the backend is up
    pserver = await ws_serve(proxy.handle, "127.0.0.1", 0, max_size=None, compression=None)
    uri = f"ws://127.0.0.1:{pserver.sockets[0].getsockname()[1]}"
    s0.close()
    await s0.wait_closed()  # take the only backend down
    try:
        async with ws_connect(uri, max_size=None, compression=None) as c:
            await c.recv()  # cached metadata greeting still delivered
            await c.send(_pack({"state": [0]}))  # infer; backend is gone
            resp = await c.recv()
            assert isinstance(resp, str), f"expected text error frame, got {type(resp)}"
            assert "unavailable" in resp, resp
    finally:
        pserver.close()
        await pserver.wait_closed()


def test_proxy_sticky_backend_down_sends_error_frame():
    asyncio.run(_scenario_backend_down())


async def _scenario_broadcast_partial_failure():
    """If one replica fails a broadcast ctrl, the router surfaces an error ack
    (not the other replica's success) — but still fans out to every backend."""
    s0, h0 = await _start_fake_backend(fail_lcc=False)
    s1, h1 = await _start_fake_backend(fail_lcc=True)
    proxy = ReplicaProxy("127.0.0.1", [h0["port"], h1["port"]])
    await proxy.prime_metadata()
    pserver = await ws_serve(proxy.handle, "127.0.0.1", 0, max_size=None, compression=None)
    uri = f"ws://127.0.0.1:{pserver.sockets[0].getsockname()[1]}"
    try:
        async with ws_connect(uri, max_size=None, compression=None) as c:
            await c.recv()
            await c.send(_pack({"__ctrl__": "load_cache_config", "yaml_content": "x"}))
            ack = _unpack(await c.recv())
            assert ack["__ack__"] == "error", ack
            assert "broadcast failed" in ack["msg"], ack
        # broadcast still reached BOTH backends despite one failing
        assert h0["lcc"] == 1 and h1["lcc"] == 1, (h0["lcc"], h1["lcc"])
    finally:
        pserver.close()
        s0.close()
        s1.close()
        await pserver.wait_closed()
        await s0.wait_closed()
        await s1.wait_closed()


def test_proxy_broadcast_partial_failure_surfaces_error():
    asyncio.run(_scenario_broadcast_partial_failure())


def test_proxy_prime_metadata_gives_up_when_backend_down():
    """prime_metadata retries then raises RuntimeError (no hang/socket leak)
    when the backend never comes up."""
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    free_port = s.getsockname()[1]
    s.close()  # nothing listens here now -> connection refused

    async def _s():
        proxy = ReplicaProxy("127.0.0.1", [free_port])
        try:
            await proxy.prime_metadata(attempts=2, delay=0.01)
        except RuntimeError as exc:
            assert "could not fetch metadata" in str(exc)
            return
        raise AssertionError("expected RuntimeError when backend never comes up")

    asyncio.run(_s())


# --- G2 R1 fixes: throughput_summary aggregation + control->sticky handoff ---


def _summary(stage1_thr, count, inferences, avg_wait, max_size, gpu_avg, cpu_proc):
    return {
        "elapsed_s": 10.0,
        "stage1_throughput_inf_per_s": stage1_thr,
        "stages": {"stage1": {"count": count, "inferences": inferences,
                              "avg_size": inferences / count, "max_size": max_size,
                              "avg_wait_ms": avg_wait, "throughput_inf_per_s": stage1_thr}},
        "averages": {"gpu_util_pct": gpu_avg, "cpu_proc_pct": cpu_proc},
        "peaks": {"gpu_util_pct": gpu_avg + 20},
        "n_util_samples": 10, "n_batch_events": count,
    }


def test_merge_throughput_summary_preserves_schema_and_aggregates():
    s0 = _summary(10.0, 5, 40, 100.0, 8, 50.0, 200.0)
    s1 = _summary(12.0, 5, 60, 200.0, 16, 60.0, 220.0)
    m = merge_throughput_summary([s0, s1], n_expected=2)
    # single-server schema preserved (operator CLI indexes these), incl. the
    # __ack__ tag so routed mode is a drop-in for the single-server response.
    assert m["__ack__"] == "throughput_summary"
    assert set(m) >= {"elapsed_s", "stage1_throughput_inf_per_s", "stages",
                      "averages", "peaks", "n_util_samples", "n_batch_events"}
    assert m["stage1_throughput_inf_per_s"] == 22.0          # sum
    st = m["stages"]["stage1"]
    assert st["inferences"] == 100 and st["count"] == 10     # sum
    assert st["avg_size"] == 10.0                            # recomputed 100/10
    assert st["max_size"] == 16                              # max
    assert st["avg_wait_ms"] == 150.0                        # count-weighted mean
    assert st["throughput_inf_per_s"] == 22.0               # sum
    assert m["averages"]["gpu_util_pct"] == 55.0            # whole-host: mean
    assert m["averages"]["cpu_proc_pct"] == 420.0           # per-process: sum
    assert m["peaks"]["gpu_util_pct"] == 80                  # max
    assert m["n_util_samples"] == 20                         # sum
    assert m["n_contributing"] == 2 and "partial" not in m


def test_merge_throughput_summary_partial_surfaces_status():
    # only 1 of 2 expected replicas contributed (other errored / dropped)
    m = merge_throughput_summary([_summary(10.0, 5, 40, 100.0, 8, 50.0, 200.0)], n_expected=2)
    assert m["partial"] is True
    assert m["n_contributing"] == 1 and m["n_replicas"] == 2
    assert m["stage1_throughput_inf_per_s"] == 10.0  # still aggregates the survivor


def test_merge_throughput_summary_all_insufficient():
    m = merge_throughput_summary([{"insufficient_window": True}, {"empty": True}], n_expected=2)
    assert m["insufficient_window"] is True
    assert m["n_contributing"] == 0 and m["n_replicas"] == 2
    assert "__ack__" not in m  # genuine short-window, NOT an error


def test_merge_throughput_summary_all_errored_is_error_not_insufficient():
    # Every backend replied with an explicit error (e.g. monitor OFF). Reporting
    # this as insufficient_window would be a wrong control-plane response.
    m = merge_throughput_summary(
        [{"__ack__": "error", "msg": "monitor OFF"}, {"__ack__": "error", "msg": "x"}],
        n_expected=2)
    assert m["__ack__"] == "error"
    assert m["n_contributing"] == 0 and m["n_errored"] == 2 and m["n_unreachable"] == 0
    assert "monitor OFF" in m["msg"]


def test_merge_throughput_summary_dropped_reply_is_error():
    # Only 1 decoded reply arrived for 2 expected replicas -> the other reply was
    # an exception/dropped, which must surface as error (unreachable), not as a
    # short-window marker that would let the caller think the host is healthy.
    m = merge_throughput_summary([{"insufficient_window": True}], n_expected=2)
    assert m["__ack__"] == "error"
    assert m["n_unreachable"] == 1 and m["n_contributing"] == 0


def test_merge_throughput_summary_aggregates_cleared_totals():
    s0 = _summary(10.0, 5, 40, 100.0, 8, 50.0, 200.0); s0["cleared"] = {"util": 3, "batch": 2}
    s1 = _summary(12.0, 5, 60, 200.0, 16, 60.0, 220.0); s1["cleared"] = {"util": 4, "batch": 5}
    m = merge_throughput_summary([s0, s1], n_expected=2)
    assert m["cleared"] == {"util": 7, "batch": 7}  # summed per-buffer across replicas


async def _scenario_control_then_sticky():
    """A persistent client that sends a broadcast ctrl then switches to infer on
    the SAME socket must have the infer routed (handoff), not dropped."""
    s0, h0 = await _start_fake_backend()
    s1, h1 = await _start_fake_backend()
    proxy = ReplicaProxy("127.0.0.1", [h0["port"], h1["port"]])
    await proxy.prime_metadata()
    pserver = await ws_serve(proxy.handle, "127.0.0.1", 0, max_size=None, compression=None)
    uri = f"ws://127.0.0.1:{pserver.sockets[0].getsockname()[1]}"
    try:
        async with ws_connect(uri, max_size=None, compression=None) as c:
            await c.recv()  # metadata
            await c.send(_pack({"__ctrl__": "load_cache_config", "yaml_content": "x"}))
            assert _unpack(await c.recv())["__ack__"] == "load_cache_config"
            # switch to data on the same connection
            await c.send(_pack({"state": [1]}))
            r = _unpack(await c.recv())
            assert "echo" in r and "port" in r, r  # routed, not dropped
        assert h0["lcc"] == 1 and h1["lcc"] == 1  # broadcast reached both
    finally:
        pserver.close()
        s0.close()
        s1.close()
        await pserver.wait_closed()
        await s0.wait_closed()
        await s1.wait_closed()


def test_proxy_control_then_sticky_handoff():
    asyncio.run(_scenario_control_then_sticky())


def test_merge_throughput_summary_multistage_max_and_empty():
    s0 = {"elapsed_s": 5.0, "stage1_throughput_inf_per_s": 4.0,
          "stages": {
              "stage1": {"count": 2, "inferences": 16, "avg_size": 8.0, "max_size": 8,
                         "worst_wait_ms": 100.0, "avg_run_ms": 50.0, "throughput_inf_per_s": 4.0},
              "stage3": {"count": 1, "inferences": 6, "avg_size": 6.0, "max_size": 6,
                         "worst_wait_ms": 300.0, "avg_run_ms": 900.0, "throughput_inf_per_s": 1.2},
              "stage2": {"count": 0}},
          "averages": {}, "peaks": {}, "n_util_samples": 5, "n_batch_events": 3}
    s1 = {"elapsed_s": 5.0, "stage1_throughput_inf_per_s": 6.0,
          "stages": {
              "stage1": {"count": 2, "inferences": 24, "avg_size": 12.0, "max_size": 16,
                         "worst_wait_ms": 250.0, "avg_run_ms": 70.0, "throughput_inf_per_s": 6.0},
              "stage2": {"count": 0}},
          "averages": {}, "peaks": {}, "n_util_samples": 5, "n_batch_events": 2}
    m = merge_throughput_summary([s0, s1], n_expected=2)
    st1 = m["stages"]["stage1"]
    assert st1["count"] == 4 and st1["inferences"] == 40
    assert st1["max_size"] == 16          # MAX across replicas
    assert st1["worst_wait_ms"] == 250.0  # MAX
    assert st1["avg_run_ms"] == 60.0      # count-weighted (2*50+2*70)/4
    # stage3 present in only one replica is carried through
    assert m["stages"]["stage3"]["count"] == 1
    assert m["stages"]["stage3"]["worst_wait_ms"] == 300.0
    # an all-empty stage stays empty
    assert m["stages"]["stage2"] == {"count": 0}
