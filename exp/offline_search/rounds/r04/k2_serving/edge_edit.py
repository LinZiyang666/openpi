from pathlib import Path
p=Path('exp/offline_search/rounds/r04/k2_serving/dev/selftest.py');s=p.read_text()
s=s.replace('    served, expected, snapshots = [], [], []','    served, expected, snapshots = [], [], []\n    rejected_preflight = output_fallbacks = partial_looks = 0')
s=s.replace('                result = conn.infer(obs)\n', '''                if is_probe and step == 1 and not due:
                    from exp.offline_search.closed_loop.blind import BlindResult
                    adapter = conn._osp_adapter
                    before = (s.step, s.b_aex.n, len(s.has_vision), conn.orch._step_counter, rt.decision_count)
                    original_step = s.method.blind_step
                    s.method.blind_step = lambda bq: BlindResult(np.zeros((1, 32), np.float32),
                        np.array([0], np.int64), np.ones(1, np.float32), "current", {})
                    s.set_obs(obs)
                    s._decision_index = rt.decision_count
                    assert plugin._try_blind(s, adapter, obs) is None and s._look_reason == 8
                    rejected_preflight += 1
                    del s.method.blind_step  # restore class dispatch
                    assert before == (s.step, s.b_aex.n, len(s.has_vision), conn.orch._step_counter, rt.decision_count)
                    saved_method, _ = plugin.clone_method(s.method)
                    output = adapter.output
                    def fail_output(*args):
                        raise ValueError("intentional output-transform failure before commit")
                    adapter.output = fail_output
                    s.set_obs(obs)
                    assert plugin._try_blind(s, adapter, obs) is None and s._look_reason == 8
                    output_fallbacks += 1
                    adapter.output = output
                    s.method = saved_method
                    assert before == (s.step, s.b_aex.n, len(s.has_vision), conn.orch._step_counter, rt.decision_count)
                    partial = {**obs, "__extra__": {"decision_id": step, "executed_steps": 4}}
                    s.set_obs(partial)
                    assert plugin._try_blind(s, adapter, partial) is None and s._look_reason == 6
                    partial_looks += 1
                    assert before == (s.step, s.b_aex.n, len(s.has_vision), conn.orch._step_counter, rt.decision_count)
                result = conn.infer(obs)
''')
s=s.replace('                    assert s.hits[-1] == 1\n', '''                    assert s.hits[-1] == 1
                    q = plugin.OnlineQueryView(s, step, s.ep.task_id, s.ep)
                    assert not q.has_tok and not q.has_vision
                    for field in ("key_v0", "key_v1", "tok_v0", "tok_v1", "img0", "img1"):
                        try:
                            getattr(q, field)
                        except rt.api.TokensUnavailable:
                            pass
                        else:
                            raise AssertionError(f"blind query exposed {field}")
''')
s=s.replace('broadcasts=sum(c.broadcasts for c in conns), connections=2, episodes=4, duplicate_rejections=4)', '''broadcasts=sum(c.broadcasts for c in conns), connections=2, episodes=4, duplicate_rejections=4,
               rejected_preflight=rejected_preflight, output_fallbacks=output_fallbacks, partial_looks=partial_looks)''')
p.write_text(s)
