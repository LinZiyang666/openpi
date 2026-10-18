"""Generate review-only integration diff. Never writes any shared source file."""
from pathlib import Path
import difflib
import hashlib
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[4]
MOD = 'exp.offline_search.rounds.r17.autoscale_1017'
PREFIX = 'exp/offline_search/rounds/r17/autoscale_1017/'


def changes(root=ROOT):
    before, after = {}, {}
    def edit(path, old, new, count=1):
        if path not in before:
            before[path] = (root / path).read_text()
            after[path] = before[path]
        n = after[path].count(old)
        if n != count:
            raise ValueError(f'{path}: expected {count} anchors, found {n}: {old[:100]!r}')
        after[path] = after[path].replace(old, new)
    p = 'src/openpi/conductor/worker.py'
    edit(p, '        self._probe = probe\n        self._probe_sent = False\n', '''        self._probe = probe
        self._probe_sent = False
        self._observer = None
        import importlib, os
        observer = os.environ.get('OPENPI_WORKER_OBSERVER')
        if observer:
            module, name = observer.split(':', 1)
            self._observer = getattr(importlib.import_module(module), name)(worker_id, runner)
''', count=1)
    edit(p, '        _proto.send_message(sock, _proto.MSG_PULL, payload)', '''        if self._observer is not None:
            payload['_autoscale'] = self._observer.payload()
        _proto.send_message(sock, _proto.MSG_PULL, payload)''')
    edit(p, '        if msg_type == _proto.MSG_SHUTDOWN:', '''        if self._observer is not None:
            self._observer.response(payload)
        if msg_type == _proto.MSG_SHUTDOWN:''')
    edit(p, '        _proto.send_message(sock, _proto.MSG_REPORT_RESULT, _proto.result_to_wire(result))', '''        payload = _proto.result_to_wire(result)
        if self._observer is not None:
            payload['_autoscale_episode'] = self._observer.result()
        _proto.send_message(sock, _proto.MSG_REPORT_RESULT, payload)''')
    edit(p, '                        "worker_id": self._worker_id,', '''                        "_autoscale": self._observer.payload() if self._observer is not None else {},
                        "worker_id": self._worker_id,''')
    edit(p, '        started = time.monotonic()\n        try:\n            result =', '''        started = time.monotonic()
        if self._observer is not None:
            self._observer.begin(task)
        try:
            result =''')
    edit(p, '        self._report_result(sock, result)\n', '''        self._report_result(sock, result)
        if self._observer is not None:
            self._observer.recycle()
''')

    edit(p, '        should_stop = stop or (lambda: False)', '''        should_stop = lambda: bool(stop and stop()) or bool(self._observer and self._observer.stop_requested())''')

    p = 'src/openpi/conductor/driver.py'
    edit(p, '        self._run_id = uuid.uuid4().hex[:12]', '''        self._run_id = uuid.uuid4().hex[:12]
        # Runtime policy is injectable; core has no experiment imports.
        import importlib, os
        module = os.environ.get('OPENPI_AUTOSCALE_DRIVER_MODULE')
        self._autoscaler = importlib.import_module(module).from_environment(self) if module else None''')
    edit(p, '            self.drive_stages_once()\n', '''            self.drive_stages_once()
            if self._autoscaler is not None:
                self._autoscaler.tick()
''')
    edit(p, '        if self._monitor is not None:\n            self._monitor.on_result(result)', '''        if self._autoscaler is not None:
            self._autoscaler.result(payload, accepted and terminal and evidence_safe)
        if self._monitor is not None:
            self._monitor.on_result(result)''')
    edit(p, '    def handle_progress(self, payload: dict[str, Any]) -> None:\n', '''    def handle_progress(self, payload: dict[str, Any]) -> None:
        if self._autoscaler is not None:
            with self._autoscaler.lock:
                self._autoscaler.ingest(payload.get('_autoscale', {}))
                self._autoscaler.tick()
''')
    edit(p, '                    out_type, assign_payload = self.handle_pull(key)', '''                    retire = bool(payload.get('_autoscale', {}).get('retire'))
                    control = None
                    if self._autoscaler is not None:
                        retire, control = self._autoscaler.pull(payload)
                    if retire:
                        out_type, assign_payload = _proto.MSG_SHUTDOWN, {}
                        local = getattr(self, '_shutdown_local', None)
                        if local is not None:
                            local.sent_shutdown = True
                    else:
                        out_type, assign_payload = self.handle_pull(key)
                    if control is not None:
                        assign_payload['_autoscale_control'] = control''')

    p = 'src/openpi/conductor/agent.py'
    edit(p, '        self._stop = threading.Event()', '''        self._stop = threading.Event()
        import importlib
        module = os.environ.get('OPENPI_WORKER_GOVERNOR')
        self._governor = importlib.import_module(module).from_environment(self) if module else None''')
    edit(p, '        self._handles[spec.worker_id] = self._spawn_fn(spec, self._driver_host, self._driver_port)', '''        handle = (self._governor.spawn(spec, self._spawn_fn, self._driver_host, self._driver_port)
                  if self._governor is not None else self._spawn_fn(spec, self._driver_host, self._driver_port))
        if handle is not None:
            self._handles[spec.worker_id] = handle
        return handle is not None''')
    edit(p, '        for spec in self._specs:\n            handle =', '''        if self._governor is not None:
            self._governor.tick()
        for spec in self._specs:
            handle =''')

    edit(p, '    def _spawn(self, spec: WorkerSpec) -> None:', '    def _spawn(self, spec: WorkerSpec) -> bool:')
    edit(p, '''                self._restart_counts[spec.worker_id] = self._restart_counts.get(spec.worker_id, 0) + 1
                logger.warning(
                    "agent: worker %s died; restart #%d", spec.worker_id, self._restart_counts[spec.worker_id]
                )
                self._spawn(spec)''', '''                if self._spawn(spec):
                    self._restart_counts[spec.worker_id] = self._restart_counts.get(spec.worker_id, 0) + 1
                    logger.warning("agent: worker %s restart #%d", spec.worker_id, self._restart_counts[spec.worker_id])''')

    p = 'exp/r17_integration/qualification/driver.py'
    edit(p, "                if len(workers)==self.expected_workers and not active:", "                if not active and (getattr(self, '_autoscaler', None) is not None or len(workers)==self.expected_workers):")
    edit(p, "            while self.scheduler.all_done() and not self._shutdown_external_stop() and time.monotonic()<until:\n", "            while self.scheduler.all_done() and not self._shutdown_external_stop() and time.monotonic()<until:\n                if getattr(self, '_autoscaler', None) is not None:self._autoscaler.tick()\n")

    p = 'exp/r17_integration/conductor/worker_entry.py'
    edit(p, '        probe = attest()', f'''        import os
        if os.environ.get('OPENPI_AUTOSCALE_NODE'):
            from {MOD}.worker import creation_lock
            with creation_lock():probe = attest()
        else:
            probe = attest()''')

    p = 'exp/r17_integration/ops/formal_process.py'
    edit(p, "    result['supervisor_alive'] = alive(result['supervisor'])", f"    from {MOD}.driver import status_result\n    status_result(request, result)\n    result['supervisor_alive'] = alive(result['supervisor'])")
    edit(p, "        request = entry['request']\n", f"        from {MOD}.driver import control_entry\n        control_entry(entry)\n        request = entry['request']\n")

    p = 'exp/r17_integration/ops/worker_lifetime.py'
    edit(p, 'import os\n', 'import os\nimport time\n')
    edit(p, 'def limited_loop(base, maximum, marker):\n', 'def limited_loop(base, maximum, marker, seconds=1800):\n')
    edit(p, '            super().run_forever(stop=lambda: self.completed >= maximum or bool(stop and stop()))\n            if self.completed >= maximum:', '''            started = time.monotonic()
            expired = lambda: self.completed > 0 and time.monotonic() - started >= seconds
            super().run_forever(stop=lambda: self.completed >= maximum or expired() or bool(stop and stop()))
            if self.completed >= maximum or expired():''')
    edit(p, "                    maximum_episodes=maximum, reason='episode_boundary'))", "                    maximum_episodes=maximum, maximum_seconds=seconds, elapsed_seconds=time.monotonic()-started, reason='episode_boundary'))")
    edit(p, "            marker.get('completed_episodes') == marker['maximum_episodes'])", "            (marker.get('completed_episodes') == marker['maximum_episodes'] or\n             0 < marker.get('completed_episodes', 0) <= marker['maximum_episodes'] and\n             marker.get('elapsed_seconds', 0) >= marker.get('maximum_seconds', float('inf'))))")

    p = 'exp/r17_integration/ops/formal_workers.py'
    edit(p, "    if not report['memory_admission'].get('admitted'):\n        pass",
         "    from .worker_memory import capacity, RAM_HEADROOM_MIB, MIB\n    _, available = capacity()\n    if available < RAM_HEADROOM_MIB*MIB:\n        return dict(admitted=False, reason='below RAM floor; do not refill recycled slot')")
    edit(p, "    restarts, pending = {}, {}", f"    restarts, pending = {{}}, {{}}\n    autoscaler = None\n    if campaign.get('autoscale'):\n        from {MOD}.node import LocalGovernor\n        autoscaler = LocalGovernor(campaign, run)")
    edit(p, "        log = (run / f'worker_{index}_generation_{generation}.log').open('wb')", "        if autoscaler is not None:\n            env.update(autoscaler.worker_env(worker_id))\n            if assignment:env['OPENPI_AUTOSCALE_GPU_UUID'] = assignment['uuid']\n        log = (run / f'worker_{index}_generation_{generation}.log').open('wb')")
    edit(p, '        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,\n            preexec_fn=parent_death, cwd=workspace, env=env)', f'''        if autoscaler is not None:
            env['OPENPI_AUTOSCALE_PARENT_PID'] = str(os.getpid())
            argv = [sys.executable, '-B', '-m', '{MOD}.bootstrap', '--', *argv]
        child = subprocess.Popen(argv, stdout=log, stderr=subprocess.STDOUT,
            preexec_fn=None if autoscaler is not None else parent_death, cwd=workspace, env=env)''')
    # Move the old fixed loop behind an explicit branch; preserve original text.
    if p not in after:raise AssertionError
    start = after[p].index('        from .worker_gpu import acquire\n')
    end = after[p].index('    except DrainedBeforeStart:\n', start)
    old = after[p][start:end]
    new = f'''        if autoscaler is not None:
            from {MOD}.fleet import run_fleet
            run_fleet(campaign, run, report, children, logs, launch,
                      lambda: stopped, drain_requested, autoscaler)
        else:
''' + ''.join('    ' + line if line.strip() else line for line in old.splitlines(True))
    edit(p, old, new)
    edit(p, "        for log in logs.values():\n", "        if autoscaler is not None and hasattr(autoscaler, 'slots'):\n            autoscaler.slots.close()\n        for log in logs.values():\n")

    edit(p, "        if autoscaler is not None and hasattr(autoscaler, 'slots'):\n", f"        if autoscaler is not None:\n            from {MOD}.kernel_evidence import collect\n            collect(run, autoscaler.started_wall)\n        if autoscaler is not None and hasattr(autoscaler, 'slots'):\n")

    p = 'exp/r17_integration/ops/worker_memory.py'
    edit(p, 'RAM_HEADROOM_MIB = 8192', 'RAM_HEADROOM_MIB = 20480')
    edit(p, "worker_memory_mib=per_worker//MIB, admitted=True)  # owner 2026-10-10: no RAM admission",
         "worker_memory_mib=per_worker//MIB, admitted=available-requested-pending >= headroom)")

    p = 'exp/r17_integration/ops/worker_gpu.py'
    edit(p, "            devices = [(e['index'], e['device']) for e in assignments] if assignments else [", "            devices = [(e['index'], e['device']) for e in assignments] if assignments or reservation.get('autoscale') else [")

    p = 'exp/r17_integration/ops/local_workers.py'
    edit(p, "('timan107','timan108','local','timan1')", "('timan107','local','timan1')")
    edit(p, " or (node == 'timan1' and int(number) > 40)", "")
    edit(p, "    worker['policy_endpoint'] =", "    if campaign.get('autoscale') and campaign['domain'] == 'metaworld':\n        worker.update(worker_max_episodes=32, worker_max_seconds=1800)\n    worker['policy_endpoint'] =")
    edit(p, '    with tempfile.TemporaryDirectory(prefix=', f"    if campaign.get('autoscale'):\n        from {MOD}.bundle import files\n        overlay += files() + ['src/openpi/conductor/worker.py', 'exp/r17_integration/conductor/worker_entry.py', 'exp/r17_integration/ops/worker_lifetime.py',\n                              'exp/r17_integration/ops/formal_process.py', 'exp/r17_integration/ops/worker_memory.py']\n    with tempfile.TemporaryDirectory(prefix=")

    p = 'exp/offline_search/rounds/r17/filter_failed/timan1_gpu2.py'
    edit(p, "not 1 <= n <= CAPS[domain]", "n < 1")
    edit(p, "    if not lease or worker['node_workers'] != lease['workers']:", "    if lease and campaign.get('autoscale'):lease = dict(lease, workers=worker['node_workers'])\n    if not lease or worker['node_workers'] != lease['workers']:")
    edit(p, "worker_max_episodes=4 if domain == 'libero' else 0", "worker_max_episodes=4 if domain == 'libero' else 32")
    edit(p, "    worker.update(worker_egl_device=DEVICE,", "    if campaign.get('autoscale'):worker['worker_max_seconds'] = 1800\n    worker.update(worker_egl_device=DEVICE,")

    p = 'exp/offline_search/rounds/r17/filter_failed/worker_resilience.py'
    edit(p, "NODES = ('timan107', 'timan108', 'timan1', 'local')", "NODES = ('timan107', 'timan1', 'local')")
    edit(p, "    import os\n    from pathlib import Path", "    if node == 'timan108':\n        raise ValueError('timan108 is disabled; no probe allowed')\n    import os\n    from pathlib import Path")
    edit(p, "                row.update(probe_status='MEASURED', probe_attempts=attempt, probe_errors=errors)", "                if campaign.get('autoscale'):\n                    row['capacity'] = campaign.get('autoscale_node_maxima', campaign.get('execution_worker_nodes', campaign['worker_nodes'])).get(node, 0)\n                    row['autoscale'] = True\n                row.update(probe_status='MEASURED', probe_attempts=attempt, probe_errors=errors)")
    edit(p, 'return dict(pool.map(probe, NODES))', "return dict(pool.map(probe, tuple(n for n in NODES if not campaign.get('autoscale') or n in campaign.get('autoscale_node_maxima', campaign.get('execution_worker_nodes', campaign['worker_nodes'])))))")
    edit(p, "    return dict(capacity=cap, gpu_slots=gpu, ram_slots=memory, cards=cards, server_budget_mib=growth,", "    return dict(capacity=cap, gpu_slots=gpu, ram_slots=memory, mem_available_bytes=mem['MemAvailable']*1024**2, cards=cards, server_budget_mib=growth,")
    old = after[p][after[p].index('    caps = {n:'):after[p].index('\n\ndef record(')]
    edit(p, old, f'''    from {MOD}.policy import place
    if any(r.get('autoscale') for r in measured.values()):
        return {{n: min(v, measured.get(n, {{}}).get('capacity') or 0) for n, v in requested.items()
                if n in NODES and (measured.get(n, {{}}).get('capacity') or 0) > 0}}
    caps = {{n: dict(cap=r['capacity'], headroom=r.get('mem_available_bytes', r['capacity']), valid=True)
            for n, r in measured.items() if n in NODES and r['capacity'] is not None and r['capacity'] > 0}}
    if not caps:
        raise RuntimeError('no usable worker node remains')
    return {{n: count for n, count in place(sum(requested.values()), caps).items() if count}}
''')
    edit(p, "('timan107','timan108','timan1','local')", "('timan107','timan1','local')")
    edit(p, "                driver._node_retired=set(state['retired'])", "                driver._node_retired=set(state['retired'])\n                driver._node_worker_retired=set(state.get('retired_workers', []))")
    edit(p, "self._shutdown_local.node_rejected=node_of(worker) not in self._node_counts", "self._shutdown_local.node_rejected=(node_of(worker) not in self._node_counts or worker in self._node_worker_retired)")
    edit(p, "            connections=[conn for conn,worker in driver._node_connections.items()", "            # Per-worker memory/throughput retirement is carried in pull metadata\n            # and handled by the core transport only at the next episode boundary.\n            connections=[conn for conn,worker in driver._node_connections.items()")

    p = 'exp/r17_integration/ops/formal_eval.py'
    edit(p, "def remote(node, argv, seconds=60, *, replay_safe=False):\n", "def remote(node, argv, seconds=60, *, replay_safe=False):\n    if node == 'timan108':raise ValueError('timan108 is disabled; no probe allowed')\n")
    edit(p, 'def prepare_workers(campaign, directory, *, payload_override=None):',
         'from exp.offline_search.rounds.r17.filter_failed import timan1_gpu2 as _autoscale_gpu2\n\n\n@_autoscale_gpu2.coordinator()\ndef prepare_workers(campaign, directory, *, payload_override=None):')
    edit(p, '@campaign_retries\ndef run_campaign', '@campaign_retries\n@_autoscale_gpu2.coordinator()\ndef run_campaign')
    edit(p, '        if ownership:\n            from .live_fleets import owned_spawn',
         "        if node == 'timan1' and label == 'workers' and campaign.get('autoscale'):\n            argv = _autoscale_gpu2.bootstrap(argv[0], ['exp.r17_integration.ops.formal_workers', *argv[4:]], 'fleet')\n        if ownership:\n            from .live_fleets import owned_spawn")
    edit(p, "    names = worker_payload_files(ROOT)", f"    names = worker_payload_files(ROOT)\n    if campaign.get('autoscale'):\n        from {MOD}.bundle import files\n        names += files()\n        if payload_override is not None and not campaign.get('autoscale_overlay'):\n            raise ValueError('use autoscale formal_trial prepare to seal a new worker orchestration overlay')")
    edit(p, "        if campaign.get('cache_capacity_authorization'):\n            from .cache_capacity import renderer_devices", "        if campaign.get('autoscale'):\n            env.update(OPENPI_AUTOSCALE_NODE=paths['run']+'/AUTOSCALE_NODE.json',\n                       OPENPI_AUTOSCALE_CONTROL=paths['run']+'/AUTOSCALE_CONTROL.json')\n            worker['autoscale_initial'] = count\n        if campaign.get('cache_capacity_authorization'):\n            from .cache_capacity import renderer_devices")
    edit(p, "        deferred=can_defer_admission(campaign)", "        deferred=bool(campaign.get('autoscale')) or can_defer_admission(campaign)")
    edit(p, "def prepare(campaign, directory):\n", f"def prepare(campaign, directory):\n    campaign.setdefault('autoscale', dict(mode='dynamic'))\n    campaign.setdefault('autoscale_maximum', campaign['workers'])\n    from {MOD}.policy import Config\n    Config(**campaign['autoscale'])\n    if campaign.get('autoscale') and any(campaign.get(k) for k in ('runtime_from', 'resume_from', 'cache_control')):\n        raise ValueError('autoscale requires a newly prepared release')\n    if campaign.get('domain') == 'metaworld' and not campaign.get('worker_max_episodes'):\n        campaign.update(worker_max_episodes=32, worker_max_seconds=1800)\n    if 'timan108' in campaign.get('execution_worker_nodes', campaign.get('worker_nodes', {{}})):\n        raise ValueError('timan108 is disabled; choose timan107, timan1 GPU2, or local')\n")
    edit(p, "    with phase(directory, 'runtime_build_verify_transfer'):", f"    from {MOD}.bundle import runtime_sources\n    with runtime_sources(campaign), phase(directory, 'runtime_build_verify_transfer'):")
    edit(p, "        entries=[dict(label=o['label'],request=o['request']) for o in remote_owned if o['node']==node]", f"        entries=[dict(label=o['label'],request=o['request']) for o in remote_owned if o['node']==node]\n        from {MOD}.driver import status_entries\n        entries=status_entries(entries, directory)")
    # Changing command documents must not grow a new staged file every30s.
    edit(p, "        if len(payload.encode())>8192 and not (globals().get('worker_resilience') and", "        if len(payload.encode())>8192 and not any(e.get('autoscale_control') for e in entries) and not (globals().get('worker_resilience') and")
    edit(p, "        for label,value in results.items():status_cache[node,label]=value\n", f"        for label,value in results.items():status_cache[node,label]=value\n    from {MOD}.driver import collect_status\n    collect_status(directory, status_cache)\n")
    edit(p, "        child_env = {**os.environ, **env}", f"        child_env = {{**os.environ, **env}}\n        if label == 'driver' and campaign.get('autoscale'):\n            child_env.update(OPENPI_AUTOSCALE_CONFIG=str(directory/'campaign.json'),\n                             OPENPI_AUTOSCALE_DRIVER_MODULE='{MOD}.driver')")
    edit(p, "choices=('timan107', 'timan108','local','timan1')", "choices=('timan107','local','timan1')")
    edit(p, "    p.add_argument('--campaign')", "    p.add_argument('--campaign')\n    p.add_argument('--semantic-campaign', help='Stable scientific namespace across paired runtime trials')")
    edit(p, "    campaign = dict(run_name=name, model=model, domain=domain,", "    campaign = dict(run_name=name, semantic_campaign=args.semantic_campaign or name, model=model, domain=domain,")
    edit(p, "Explicit per-node counts, e.g. timan107=8,local=16; overrides --workers/--worker-node", "Desired per-node maxima, e.g. timan1=150,timan107=100; runtime chooses active counts")
    edit(p, "    p.add_argument('--extra-workers',", "    p.add_argument('--worker-autoscale', choices=('dynamic','fixed'), default='dynamic')\n    p.add_argument('--worker-memory-floor-gib', type=float, default=20)\n    p.add_argument('--worker-memory-critical-gib', type=float, default=10)\n    p.add_argument('--extra-workers',")
    edit(p, "args.worker_node = args.worker_node or ('timan108' if cosmos else 'timan107')", "args.worker_node = args.worker_node or 'timan107'")
    edit(p, "args.worker_max_episodes = 4 if args.model.endswith('libero') else 0", "args.worker_max_episodes = 4 if args.model.endswith('libero') else 32")
    edit(p, "        worker_nodes['timan108' if args.worker_node == 'timan107' else 'timan107'] = args.extra_workers", "        worker_nodes['timan1' if args.worker_node == 'timan107' else 'timan107'] = args.extra_workers")
    edit(p, "        worker_max_episodes=args.worker_max_episodes,", "        worker_max_episodes=args.worker_max_episodes, worker_max_seconds=1800,\n        autoscale=dict(mode=args.worker_autoscale, floor=int(args.worker_memory_floor_gib*1024**3),\n                       critical=int(args.worker_memory_critical_gib*1024**3)),\n        autoscale_maximum=sum(worker_nodes.values()),")

    p = 'exp/r17_integration/ops/driver_release.py'
    edit(p, "    if overlay:\n", f"    if campaign.get('autoscale'):\n        from {MOD}.release import driver_files as autoscale_files\n        overlay = (*overlay, *autoscale_files())\n    if overlay:\n")
    edit(p, "    return {n:files[n] for n in sorted(names)}", f"    names.update(n for n in files if n.startswith('{PREFIX}'))\n    return {{n:files[n] for n in sorted(names)}}")

    p = 'exp/offline_search/rounds/r16/conductor.py'
    edit(p, "    if not 1<=args.workers<=64: raise ValueError('Worker cap is 64')", f"    if args.workers < 1: raise ValueError('Positive desired worker maximum required')\n    os.environ.update(OPENPI_WORKER_MEMORY_GUARD='1',\n                      OPENPI_WORKER_GOVERNOR='{MOD}.legacy',\n                      OPENPI_AUTOSCALE_MODEL=args.model,\n                      OPENPI_AUTOSCALE_RUN=str(Path('/scratch/zixuans8/mw_r17/autoscale_legacy')/str(os.getpid())))")
    # Preserve the historical desired maximum; only the runtime chooses active count.
    edit(p, "p.add_argument('--workers',type=int,default=64)", "p.add_argument('--workers',type=int,default=64,help='Desired maximum; node memory governor chooses active count')")

    p = 'exp/offline_search/closed_loop/ops/h100/fleet.py'
    edit(p, '    "timan108": (Path("/scratch/zixuans8/openpi_trace"), tuple(range(4)), 40),\n', '')
    edit(p, 'os.environ.get("WORKER_HOST", "timan108")', 'os.environ.get("WORKER_HOST", "timan107")')
    edit(p, 'WORKER_HOST must be timan108 or timan107', 'WORKER_HOST must be timan107; timan108 is disabled')

    p = 'exp/offline_search/closed_loop/ops/h100/control.py'
    edit(p, 'REMOTE_NODE = {"h100": str(BASE / "node.py"), "timan108": str(ISLAND / "os_cl/node.py")}',
         'REMOTE_NODE = {"h100": str(BASE / "node.py")}')
    edit(p, 'def remote(node, cmd, timeout=300, attempts=None):\n',
         "def remote(node, cmd, timeout=300, attempts=None):\n    if node == 'timan108':raise ValueError('timan108 is disabled; no probe allowed')\n")

    p = 'exp/offline_search/closed_loop/ops/h100/run_arm.sh'
    edit(p, 'DEFAULT_WORKER_HOST=timan108', 'DEFAULT_WORKER_HOST=timan107')
    edit(p, '  timan108) GPU_IDS=0,1,2,3; GPU_COUNT=4; WORKER_CAP=40 ;;\n', '')
    edit(p, 'WORKER_HOST must be timan108 or timan107', 'WORKER_HOST must be timan107; timan108 is disabled')
    edit(p, 'export OMP_NUM_THREADS=1', f"export OPENPI_WORKER_MEMORY_GUARD=1\nexport OPENPI_WORKER_GOVERNOR={MOD}.legacy\nexport OPENPI_AUTOSCALE_RUN=\"$OUT/autoscale\"\nexport OPENPI_AUTOSCALE_CAMPAIGN=\"$ARM\"\nexport OMP_NUM_THREADS=1")
    return before, after


def main():
    before, after = changes()
    patch = ''.join(''.join(difflib.unified_diff(before[p].splitlines(True), after[p].splitlines(True),
                fromfile='a/' + p, tofile='b/' + p)) for p in before if before[p] != after[p])
    (HERE / 'INTEGRATION.patch').write_text(patch)
    (HERE / 'PATCH_BASE.json').write_text(json.dumps({p: dict(sha256=hashlib.sha256(before[p].encode()).hexdigest(),
        proposed_sha256=hashlib.sha256(after[p].encode()).hexdigest()) for p in before}, indent=2) + '\n')
    for p, text in after.items():
        if p.endswith('.py'):
            compile(text, p, 'exec')
    print('Prepared patch for %d shared files; applied none.' % len(before))


if __name__ == '__main__':
    main()
