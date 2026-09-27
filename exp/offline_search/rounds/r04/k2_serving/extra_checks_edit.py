from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/plugin.py');s=p.read_text()
s=s.replace('        if rt.judge is not None:\n            self._dec.update(self._verdict(step, float(conf), ex))', '''        if rt.r4:
            anchor_owner = self.method
            for _ in range(8):
                anchor = getattr(anchor_owner, "_anchor", None)
                if isinstance(anchor, dict) and "rows" in anchor and "weights" in anchor:
                    self._dec.update(rows=np.array(anchor["rows"], np.int64, copy=True),
                                     weights=np.array(anchor["weights"], np.float32, copy=True))
                    break
                anchor_owner = getattr(anchor_owner, "base", None)
                if anchor_owner is None:
                    break
        if rt.judge is not None:
            self._dec.update(self._verdict(step, float(conf), ex))''')
s=s.replace('        self._blind_extras = {}\n        self._age_before', '        self._blind_extras = {}\n        self._prepare_ms = None\n        self._output_ms = None\n        self._age_before')
s=s.replace('                       last_vision_step=self.last_vision_step)', '''                       last_vision_step=self.last_vision_step,
                       blind_prepare_ms=getattr(self, "_prepare_ms", None),
                       blind_output_ms=getattr(self, "_output_ms", None))''')
s=s.replace('    rs, state = adapter.prepare(obs)', '    prepare_start = time.perf_counter_ns()\n    rs, state = adapter.prepare(obs)\n    s._prepare_ms = (time.perf_counter_ns() - prepare_start) / 1e6')
s=s.replace('    output = adapter.output(action, state)', '    output_start = time.perf_counter_ns()\n    output = adapter.output(action, state)\n    s._output_ms = (time.perf_counter_ns() - output_start) / 1e6')
p.write_text(s)
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/probe.py');s=p.read_text()
s=s.replace('        self.weights = np.full(len(o), 1 / len(o), np.float32)\n', '        self.weights = np.full(len(o), 1 / len(o), np.float32)\n        self._anchor = dict(rows=self.anchor_rows.copy(), weights=self.weights.copy())\n')
p.write_text(s)
