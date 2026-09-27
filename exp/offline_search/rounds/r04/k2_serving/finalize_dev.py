from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/plugin.py');s=p.read_text()
s=s.replace('        self._s1_ms = None\n        self._age_before', '        self._s1_ms = None\n        self._blind_extras = {}\n        self._age_before')
s=s.replace('    if isinstance(reason, LookReason):\n        s._look_reason', '    if isinstance(reason, LookReason):\n        s._blind_extras = copy.deepcopy(getattr(s.method, "last_blind_extras", {}))\n        s._look_reason')
s=s.replace('                d["policy"] = a\n', '''                d["policy"] = a
                if self.rt.blind and callable(getattr(self.method, "invalidate_anchor", None)):
                    self.method.invalidate_anchor()
''')
s=s.replace('            if "rows" in d:\n                row.update', '''            if getattr(self, "_blind_extras", None):
                row["blind_extras"] = self._blind_extras
            if "rows" in d:
                row.update''')
s=s.replace('        if self.rt.opts.os_tokens != "on" or self.kb is None:', '        if (self.has_vision and not self.has_vision[-1]) or self.rt.opts.os_tokens != "on" or self.kb is None:')
s=s.replace('        if self.rt.opts.os_tokens != "on" or self.cur_obs is None or key not in self.cur_obs:', '        if (self.has_vision and not self.has_vision[-1]) or self.rt.opts.os_tokens != "on" or self.cur_obs is None or key not in self.cur_obs:')
p.write_text(s)
