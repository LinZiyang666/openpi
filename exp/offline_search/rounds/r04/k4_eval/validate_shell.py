"""Exercise only extracted selection functions and a locally stubbed remote launcher."""
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile

HERE=Path(__file__).resolve().parent
REPO=HERE.parents[4]
BASE=REPO/'exp/offline_search/closed_loop/ops' if '--installed' in sys.argv else HERE/'dev'
PREFIX=['taskset','-c','30-33,74-77',sys.executable]
with tempfile.TemporaryDirectory(prefix='k4_shell_') as td:
    p=Path(td); state=p/'state'; state.mkdir(); cfg=p/'os_cl/cfg'; cfg.mkdir(parents=True)
    small=p/'small.json'; small.write_text('[[0,0],[0,0],[1,3]]')
    large=p/'large.json'; large.write_text('[[0,0],[1,3],[9,49]]')
    chain=(BASE/'chain.sh').read_text()
    select=chain[chain.index('selection_for()'):chain.index('prepare_selection()')]
    source='\n'.join([
      f'HERE={shlex.quote(str(BASE))}', f'STATE={shlex.quote(str(state))}',
      'ISL=/unused; TAG=test; LEGACY_EXPECT=500',
      'field_or() { echo ""; }',
      'python3() { '+shlex.join(PREFIX)+' "$@"; }',select,
      f'OSCL_MANIFEST={shlex.quote(str(small))}; selection_for a || exit 3',
      '[ "$EXPECT" = 2 ] || exit 4; SMALL_DONE=$DONE; touch "$DONE"',
      f'OSCL_MANIFEST={shlex.quote(str(large))}; selection_for a || exit 5',
      '[ "$EXPECT" = 3 ] && [ "$DONE" != "$SMALL_DONE" ] && [ ! -f "$DONE" ] || exit 6',
      'OSCL_MANIFEST=; selection_for a; [ "$EXPECT" = 500 ] && [ "$DONE" = "$STATE/a.DONE" ] && [ ! -f "$DONE" ] || exit 7',
      'LEGACY_EXPECT=100; selection_for a; [ "$EXPECT" = 100 ] || exit 8',
      'echo PASS_selection_markers'])
    subprocess.run(['bash','-c',source],check=True)
    # Extract servers_up into a fake launcher environment. No real GPU/port/tmux tool runs.
    stub=p/'launcher'; stub.mkdir(); (p/'runs').mkdir()
    (p/'arms.json').write_text(json.dumps([dict(arm='a',model='groot',suite='libero_10',yaml='unused',mode='plugin',
        cell='groot_l10_cache',method='seeded:Inference',kwargs={},full_model=True,server_seed=1,
        server_env={'GROOT_DENOISING_STEPS':2},plugin_args=['--os-judge','periodic:1'])]))
    (stub/'start_server.sh').write_text('#!/bin/bash\nprintf "%s\\n" "STAGE1_ONLY=$STAGE1_ONLY" "K=$GROOT_DENOISING_STEPS" "$@" > "$STUB_ROOT/call_$3"\ntouch "$STUB_ROOT/up_$3"\n')
    fields=chain[chain.index('field()'):chain.index('if [ -n "${OSCL_EPISODES')]
    servers=chain[chain.index('server_args()'):chain.index('servers_down()')]
    source='\n'.join([f'RUN={shlex.quote(str(p))}', f'HERE={shlex.quote(str(stub))}',
        f'export STUB_ROOT={shlex.quote(str(p))}', 'PORTS=23180,23181',
        'python3() { '+shlex.join(PREFIX)+' "$@"; }',
        'nvidia-smi() { echo 50000; }',
        # ss receives "sport = :PORT" as its last arg.
        'ss() { local p="${*: -1}"; test -f "$STUB_ROOT/up_${p##*:}" && echo LISTEN; }',
        'sleep() { :; }', 'note() { :; }', 'ev() { echo "$*" >&2; }',fields,servers,'servers_up a'])
    subprocess.run(['bash','-c',source],check=True)
    seeds=[]
    for port in [23180,23181]:
        args=(p/f'call_{port}').read_text().splitlines()
        assert args[:2]==['STAGE1_ONLY=0','K=2']
        seeds.append(int(args[args.index('--os-seed')+1]))
    assert seeds==[65536+23180,65536+23181]
    print('PASS_server_argument_stubs 2 process seeds; full model; per-arm GR00T K2 env')
    startup=p/'shell_env'
    startup.write_text('python3() { '+shlex.join(PREFIX)+' "$@"; }\n')
    common={**os.environ,'BASH_ENV':str(startup),'PORTS':'23180','PILOT_DRY':'1'}
    pilot_out=[]
    for base in [HERE/'before',BASE]:
        out=subprocess.check_output(['bash',str(base/'pilot.sh'),str(p),'l10','a'],env=common,text=True)
        pilot_out.append(out.replace(str(base),'<OPS>'))
    assert pilot_out[0]==pilot_out[1]
    out=subprocess.check_output(['bash',str(BASE/'pilot.sh'),str(p),'l10','a'],
                                env={**common,'OSCL_MANIFEST':str(small)},text=True)
    assert 'expect=2/arm' in out and '--manifest' in out
    print('PASS_pilot_dry legacy stdout identical; manifest EXPECT=2')
    # run_arm's PY executes real Python only for matrix/count parsing; actual worker entry is a recorder.
    (p/'os_cl/count.py').write_text((BASE/'remote/count.py').read_text())
    (p/'os_cl/run_gtp_subset.py').write_text((BASE/'remote/run_gtp_subset.py').read_text())
    fake=p/'fakepy'; fake.write_text('#!/bin/bash\nif [ "$1" = - ] || [[ "$1" == */count.py ]]; then exec '+shlex.join(PREFIX)+' "$@"; fi\nprintf "%s\\n" "$@" > "$ARG_LOG"\n')
    fake.chmod(0o755)
    script=(BASE/'remote/run_arm.sh').read_text().replace('R=/scratch/zixuans8/openpi_trace',f'R={shlex.quote(str(p))}').replace('PY=/scratch/zixuans8/openpi/.venv/bin/python',f'PY={shlex.quote(str(fake))}')
    launcher=p/'run_arm.sh'; launcher.write_text(script)
    (cfg/'a.yaml').write_text('model: pi05\n')
    for i,(client,extra,expected) in enumerate([({},[],5),({'replan_steps':10},[],10),({'replan_steps':10},['--replan-steps','7'],7)]):
        (cfg/'matrix_a.yaml').write_text('arms:\n- arm: a\n  yaml: x\n  suite: libero_10\n  client_overrides: '+json.dumps(client)+'\n')
        argsfile=p/f'args{i}'; env={**os.environ,'ARG_LOG':str(argsfile)}
        subprocess.run(['bash',str(launcher),'libero_10','a','unused:1','1',str(p/'out'),'--manifest',str(small),*extra],env=env,check=True,stdout=subprocess.DEVNULL)
        args=argsfile.read_text().splitlines(); assert args[args.index('--replan-steps')+1]==str(expected),args
        assert args[0].endswith('run_gtp_subset.py') and '--manifest' not in args
        assert args[args.index('--trials')+1]=='50'
    (cfg/'matrix_a.yaml').write_text('arms:\n- arm: a\n  yaml: x\n  suite: libero_10\n')
    argsfile=p/'oldargs'; env={**os.environ,'ARG_LOG':str(argsfile)}
    subprocess.run(['bash',str(launcher),'libero_10','a','unused:1','1',str(p/'out')],env=env,check=True,stdout=subprocess.DEVNULL)
    args=argsfile.read_text().splitlines(); assert args[:2]==['-m','exp.gate_threshold_pareto.run_gtp']
    old=(HERE/'before/remote/run_arm.sh').read_text().replace('R=/scratch/zixuans8/openpi_trace',f'R={shlex.quote(str(p))}').replace('PY=/scratch/zixuans8/openpi/.venv/bin/python',f'PY={shlex.quote(str(fake))}')
    old_launcher=p/'old_run_arm.sh'; old_launcher.write_text(old)
    old_args=p/'original_args'; env={**os.environ,'ARG_LOG':str(old_args)}
    subprocess.run(['bash',str(old_launcher),'libero_10','a','unused:1','1',str(p/'out')],env=env,check=True,stdout=subprocess.DEVNULL)
    assert old_args.read_bytes()==argsfile.read_bytes()
    print('PASS_legacy_remote_argv byte-identical')
    print('PASS_remote_launcher 4 cases: default5/matrix10/CLI7/exact-manifest; full trials50 retained')
    for script in ['chain.sh','pilot.sh','remote/run_arm.sh']:
        subprocess.run(['bash','-n',str(BASE/script)],check=True)
    print('PASS_bash_syntax')
