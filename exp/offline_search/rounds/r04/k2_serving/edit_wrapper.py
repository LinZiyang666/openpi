from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/plugin.py')
s=p.read_text()
s=s.replace('class _ConnPolicy:',Path('exp/offline_search/rounds/r04/k2_serving/dev/serving_snippet.txt').read_text()+'class _ConnPolicy:')
s=s.replace('        if hasattr(inner, "on_episode_start"):', '''        self._osp_adapter = None
        enabled = [s for s in sessions if s.rt.r4]
        if enabled:
            if len(sessions) != 1:
                raise RuntimeError("osplug: R4 serving expects one CP1 session per connection")
            self._osp_adapter = _BlindAdapter(inner, enabled[0])
        if hasattr(inner, "on_episode_start"):''')
s=s.replace('            out = self._osp_inner.infer(obs, *a, **kw)\n            ok = True', '''            adapter = self._osp_adapter
            if adapter is None:
                out = self._osp_inner.infer(obs, *a, **kw)
            else:
                s = sessions[0]
                with s.rt.decision_lock:
                    s._decision_index = s.rt.decision_count
                    out = None
                    if s.rt.blind:
                        with adapter.lock:
                            out = _try_blind(s, adapter, obs)
                    if out is None:
                        out = self._osp_inner.infer(obs, *a, **kw)
                    if s._dec is not None:
                        s._dec["wire_actions"] = np.asarray(out["actions"]).copy()
                    s.rt.decision_count += 1
                    s._last_decision_id = getattr(s, "_pending_decision_id", None)
            ok = True''')
p.write_text(s)
