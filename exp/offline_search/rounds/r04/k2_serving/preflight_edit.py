from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/plugin.py');s=p.read_text()
s=s.replace('            if getattr(self, "_blind_extras", None):\n                row["blind_extras"] = self._blind_extras', '''            if not vision:
                row["blind_extras"] = ex
            elif getattr(self, "_blind_extras", None):
                row["blind_extras"] = self._blind_extras''')
s=s.replace('        raise ValueError("invalid blind action/members/weights; no action or history committed")', '''        log.warning("osplug: invalid blind candidate; requesting vision before commit")
        s._look_reason = 8
        invalidate = getattr(s.method, "invalidate_anchor", None)
        if callable(invalidate):
            invalidate()
        return None''')
s=s.replace('    output = adapter.output(action, state)\n    s._output_ms', '''    try:
        output = adapter.output(action, state)
    except Exception:
        log.exception("osplug: blind output preflight failed; requesting vision before commit")
        s._look_reason = 8
        invalidate = getattr(s.method, "invalidate_anchor", None)
        if callable(invalidate):
            invalidate()
        return None
    s._output_ms''')
p.write_text(s)
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/replay_client.py');s=p.read_text()
s=s.replace('    if a.wire_checkpoint:\n', '''        if "mixed" in rep:
            misses = sum(int((z["hit"] == 0).sum()) for z in logged.values())
            denoise = sum(float(np.nansum(z["miss_k"])) for z in logged.values())
            rep["mixed"]["ir_pi05_formula"] = (.152 * vision_n + .410 * misses + .438 * denoise / 10) / acc["n"]
    if a.wire_checkpoint:
''')
p.write_text(s)
