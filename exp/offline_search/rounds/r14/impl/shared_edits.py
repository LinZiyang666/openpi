"""Apply the small default-off diagnostic extension by atomic replacement."""
from .common import HERE, REPO, atomic, write_json, sha

def main():
    edits = {
      'plugin.py': [
        ('    ap.add_argument("--os-record-looks", action="store_true", help="record real looks on attested B dev roots only")',
         '    ap.add_argument("--os-r14-audit", action="store_true", help="versioned R14 side diagnostics; default off")\n    ap.add_argument("--os-record-looks", action="store_true", help="record real looks on attested B dev roots only")'),
        ('        self.method = method\n        self.method_name = method.name',
         '        if getattr(opts, "os_r14_audit", False):\n            from .devset import sha\n            self.r14_fit_sha256 = sha(art) if art is not None and art.exists() else None\n        self.method = method\n        self.method_name = method.name'),
        ('        if getattr(rt, "looks", None) is not None and getattr(rt, \'lookhalf\', None) is None:',
         '        if getattr(rt.opts, "os_r14_audit", False):\n            from exp.offline_search.rounds.r14.knob.recipe import reference_diagnostics\n            self._dec["r14"] = copy.deepcopy(getattr(self.method, "r14_diagnostics", None)\n                or reference_diagnostics(self.method, view, res))\n            self._dec["r14"].update(schema="offline_search.r14.v1", fit_sha256=rt.r14_fit_sha256,\n                retrieval_correction_ms=t_q / 1e6, step=step, task_id=int(view.task_id))\n        if getattr(rt, "looks", None) is not None and getattr(rt, \'lookhalf\', None) is None:'),
        ('        if err:\n            row["error"] = err[-2000:]',
         '        if getattr(rt.opts, "os_r14_audit", False):\n            before = int(d["step"])\n            audit = copy.deepcopy(d.get("r14", {}))\n            audit.update(schema="offline_search.r14.v1", fit_sha256=rt.r14_fit_sha256,\n                task_id=self.ep.task_id, N=before, V=sum(self.has_vision[:before]),\n                M=sum(v and h == 0 for v, h in zip(self.has_vision[:before], self.hits[:before])),\n                real_look=bool(d.get("vision", True)), look_reason=self._look_reason,\n                source=row.get("src"), lifecycle_fallback_counts=copy.deepcopy(\n                    getattr(self.method, "_fallback_counts", {})))\n            row["r14"] = audit\n        if err:\n            row["error"] = err[-2000:]'),
        ('    use_tail = bool(s.rt.policy_tail and s.step and',
         '    if getattr(s.rt.opts, "os_r14_audit", False) and (len(s.hits) != s.step\n            or len(s.has_vision) != s.step or s.b_aex.n != s.step):\n        raise s.rt.api.ContractError("R14 accepted buffers disagree with the decision count")\n    use_tail = bool(s.rt.policy_tail and s.step and')
      ],
      'look_recording.py': [
        ("        interval = getattr(self.opts, 'os_groot_path_check_every', 0)",
         '        if getattr(self.opts, "os_r14_audit", False):\n            arrays["r14"] = np.array(json.dumps(json_value(decision.get("r14", {})), sort_keys=True))\n' + "        interval = getattr(self.opts, 'os_groot_path_check_every', 0)")
      ]}
    records = []
    for name, replacements in edits.items():
        path = REPO / 'exp/offline_search/closed_loop' / name
        data = path.read_text()
        saved = HERE / 'before' / name
        if not saved.exists():
            atomic(saved, data.encode())
        elif (HERE / 'shared_edits.json').exists():
            expected = next(r['after_sha256'] for r in __import__('json').loads(
                (HERE / 'shared_edits.json').read_text()) if r['path'] == str(path))
            if sha(path) != expected:
                raise ValueError(f'shared file changed independently: {path}')
            data = saved.read_text()
        for old, new in replacements:
            if new in data:
                continue
            if data.count(old) != 1:
                raise ValueError(f'expected one insertion point in {name}')
            data = data.replace(old, new)
        compile(data, str(path), 'exec')
        atomic(path, data.encode())
        records.append(dict(path=str(path), before_sha256=sha(saved), after_sha256=sha(path)))
    write_json(HERE / 'shared_edits.json', records)

if __name__ == '__main__':
    main()
