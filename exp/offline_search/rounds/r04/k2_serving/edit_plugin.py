from pathlib import Path
p = Path('exp/offline_search/rounds/r04/k2_serving/dev/plugin.py')
s = p.read_text()
def rep(a,b):
    global s
    assert a in s, a[:100]
    s=s.replace(a,b)
rep('import copy\n', 'import copy\nimport contextlib\n')
rep('    return ap\n', '''    ap.add_argument("--os-blind", action="store_true",
                    help="enable pre-inference blind_step serving and R4 decision logs")
    ap.add_argument("--os-log-r4", action="store_true",
                    help="emit R4 vision/source/served-head fields without enabling blind serving")
    return ap
''')
rep('    return opts, rest\n', '''    if opts.os_blind and opts.os_method == "native":
        raise SystemExit("--os-blind requires a method")
    return opts, rest
''')
rep('        self.opts = opts\n', '''        self.opts = opts
        self.blind = bool(getattr(opts, "os_blind", False))
        self.r4 = self.blind or bool(getattr(opts, "os_log_r4", False))
        self.decision_count = 0
        self.decision_lock = threading.RLock()
''')
rep('        self.emit(row)\n        log.info("osplug ready:', '''        if self.r4:
            row.update(blind=self.blind, r4=True, periodic_clock="server" if self.blind else "episode",
                       stage1_mode=getattr(opts, "os_stage1_mode", "full"),
                       miss_steps=getattr(opts, "os_miss_steps", None) or (10 if m == "pi05" else 8))
        self.emit(row)
        log.info("osplug ready:''')
rep('        self._tok_cache: dict = {}\n', '''        self._tok_cache: dict = {}
        self.blind_age = 0
        self.last_vision_step = -1
        self.has_vision = []
        self._look_reason = None
        self._prepared_rs = None
        self._s1_ms = None
        self.stage1_calls = 0
''')
rep('        self.hits: list = []\n', '''        self.hits: list = []
        self.has_vision = []
        self.blind_age = 0
        self.last_vision_step = -1
''')
rep('        if step == 0 and J.step0 != "judge":', '''        if rt.blind and J.mode == "periodic" and self._decision_index % J.k == J.k - 1:
            hit, why = False, "periodic"
        elif step == 0 and J.step0 != "judge":''')
rep('            hit, why = (step % J.k != J.k - 1), "periodic"', '''            clock = self._decision_index if rt.blind else step
            hit, why = (clock % J.k != J.k - 1), "periodic"''')
rep('        self.b_raw.append(self._raw_state())\n        self._tok_cache = {}', '''        self.b_raw.append(self._raw_state())
        self.has_vision.append(True)
        if self.rt.blind and self._prepared_rs is not None:
            if not np.array_equal(self._prepared_rs, self.b_rs.a[self.step]):
                raise RuntimeError("osplug: blind CPU state does not match the live key builder")
        self.last_vision_step = self.step
        self._tok_cache = {}''')
rep('            if self.rt.judge is not None:\n                d["t_exec"]', '            if self.rt.judge is not None or self.rt.r4:\n                d["t_exec"]')
rep('        self.t_obs = time.perf_counter_ns()\n', '''        self.t_obs = time.perf_counter_ns()
        self._look_reason = None
        self._prepared_rs = None
        self._s1_ms = None
        self._age_before = self.blind_age
''')
rep('        if err:\n            row["error"]', '''        if rt.r4:
            hit = bool(d.get("hit", True))
            vision = bool(d.get("vision", True))
            served = d.get("served") if hit else d.get("policy")
            row.update(vision=vision, src="cache_blind" if not vision else ("cache" if hit else "policy"),
                       hit=hit, blind_age=self._age_before, look_reason=self._look_reason,
                       miss_k=None if hit else getattr(self, "miss_steps", 10 if rt.model == "pi05" else 8),
                       s1_ms=self._s1_ms if vision else None,
                       s23_ms=round((d["t_exec"] - d["t_s1"]) / 1e6, 3)
                       if not hit and d.get("t_exec") is not None else None,
                       served_head=None if served is None else np.asarray(served)[:5, :7].tolist(),
                       searched=vision, source="cache_blind" if not vision else ("cache" if hit else "policy"),
                       decision_index=getattr(self, "_decision_index", d["step"]),
                       stage1_calls=self.stage1_calls, shadow_available=vision and self.shadow,
                       robot_state=self.b_rs.a[d["step"]].tolist(),
                       last_vision_step=self.last_vision_step)
            if "rows" in d:
                row.update(rows=d["rows"], weights=d["weights"])
            self.blind_age = 0 if vision else self._age_before + 1
        if err:
            row["error"]''')
rep('            self._recs.append(rec)\n', '''            if rt.r4:
                rec.update({k: row[k] for k in ("vision", "blind_age", "look_reason", "miss_k", "s1_ms", "s23_ms",
                                               "decision_index", "stage1_calls", "hit")})
                rec["wire_actions"] = d.get("wire_actions")
                rec["rows"], rec["weights"] = d.get("rows"), d.get("weights")
            self._recs.append(rec)
''')
rep('        if rt.judge is not None:\n            meta["judge"] = rt.judge.as_dict()', '''        if rt.r4:
            meta.update(blind=rt.blind, r4=True)
        if rt.judge is not None:
            meta["judge"] = rt.judge.as_dict()''')
rep('        keys = sorted({kk for r in recs', '''        if rt.r4:
            arrays.update(has_vision=np.array(self.has_vision[:n], bool),
                          wire_actions=np.asarray([r["wire_actions"] for r in recs]))
            for key in ("blind_age", "look_reason", "miss_k", "s1_ms", "s23_ms", "decision_index", "stage1_calls", "hit"):
                arrays[key] = np.asarray([math.nan if r[key] is None else r[key] for r in recs])
            arrays["vision"] = arrays["has_vision"]
            width = max([len(r["rows"]) if r["rows"] is not None else 0 for r in recs] + [0])
            arrays["blind_rows"] = np.full((n, width), -1, np.int64)
            arrays["blind_weights"] = np.full((n, width), np.nan, np.float32)
            for i, rec in enumerate(recs):
                if rec["rows"] is not None:
                    arrays["blind_rows"][i, :len(rec["rows"])] = rec["rows"]
                    arrays["blind_weights"][i, :len(rec["weights"])] = rec["weights"]
        keys = sorted({kk for r in recs''')
rep('    @property\n    def has_tok(self):\n        return self._s.rt.opts.os_tokens == "on"', '''    @property
    def has_vision(self):
        return bool(self._s.has_vision[self.step])

    @property
    def hist_has_vision(self):
        return _ro(np.asarray(self._s.has_vision[:self.step], bool))

    @property
    def last_vision_step(self):
        return self._s.last_vision_step

    @property
    def blind_age(self):
        return self._s._age_before

    @property
    def has_tok(self):
        return self.has_vision and self._s.rt.opts.os_tokens == "on"''')
rep('    def key_v0(self):\n        return', '    def key_v0(self):\n        if not self.has_vision:\n            raise self._s.rt.api.TokensUnavailable("no visual key on a blind decision")\n        return')
rep('    def key_v1(self):\n        return', '    def key_v1(self):\n        if not self.has_vision:\n            raise self._s.rt.api.TokensUnavailable("no visual key on a blind decision")\n        return')
p.write_text(s)
