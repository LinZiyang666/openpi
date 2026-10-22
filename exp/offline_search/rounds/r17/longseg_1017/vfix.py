"""10-18 vfix: serve_loeo_episodes fallback on pre-LOEO frozen recipe releases.

repin  Re-pin assets/ffk_bootstrap.py of never-prepared long-segment definitions
       whose frozen LayoutRecipe predates serve_loeo_episodes, with the tooling
       of the 10-18 decision-clock recalibration (build.pin, plus
       recalibrate.pin_plan_evidence for recalibrated units). The only byte
       change is the embedded filter_failed_knob/runtime.py fallback; names,
       YAMLs, plan, task UIDs, inits, seeds and ffk_evidence.py are unchanged.

retry  Zero-clean-progress retry of a CLOSED campaign under its own name. The
       whole directory and every node's owned trees are renamed to explicit backups (nothing deleted, no process
       signalled) and the scientific definition is copied into a fresh
       directory. Accepted rows may exist only if every one is an error row
       (e.g. FatalEpisodeError); they stay in the backup and are never carried
       over. runtime.tar/RUNTIME_RELEASE.json are restored from the source
       release, so staging writes the segment-boundary fix anew. Then repin
       and the native preparation (filter_failed.stage --reprepare).
"""
from __future__ import annotations

import argparse
import ast
import copy
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import tempfile
import time

from .audit import HERE, EVAL, io_ready
from .io_budget import dump
from exp.offline_search.rounds.r17.filter_failed_knob.common import read, sha

KNOB = HERE.parent/'filter_failed_knob'
LAYER = 'exp/r17_integration/common/layers/recipe.py'
TAG = '.pre_vfix1018_'
OLD_LINE = "        own = self.base.serve_loeo_episodes.get((request.task_id, request.original_init_idx))\n"
NEW_BLOCK = ("        # Frozen LayoutRecipe releases before 10-11 have no serving LOEO: their\n"
             "        # native query never excludes an own episode, so none applies here.\n"
             "        loeo = getattr(self.base, 'serve_loeo_episodes', {})\n"
             "        own = loeo.get((request.task_id, request.original_init_idx))\n")
SCIENCE = ('campaign.json', 'EPISODE_PLAN.json', 'assets', 'workers.tar', 'STUDY.json',
           'FFK_DEFINITION.json', 'OFFLINE_READY.json', 'control_inputs')


def embedded_runtime(bootstrap):
    calls = [n for n in ast.walk(ast.parse(bootstrap)) if isinstance(n, ast.Call) and
             getattr(n.func, 'id', None) == 'compile' and len(n.args) > 1 and
             isinstance(n.args[1], ast.Constant) and n.args[1].value == 'ffk1015_runtime']
    if len(calls) != 1:
        raise ValueError('bootstrap must embed ffk1015_runtime exactly once')
    return calls[0].args[0].value


def needs_fallback(campaign):
    return 'self.serve_loeo_episodes' not in (Path(campaign['runtime'])/LAYER).read_text()


def replace_json(path, value, stamp, *, compact=False):
    """Keep the old bytes beside it; write the new document as a new inode."""
    path = Path(path)
    shutil.copyfile(path, path.with_name(path.name+TAG+stamp))
    part = path.with_name(path.name+'.vfix_part')
    dump(part, value, compact=compact)
    os.replace(part, path)


def plan_identity(campaign, root):
    from dataclasses import asdict
    from exp.offline_search.rounds.r17.filter_failed_knob.build import cpu_plan
    _, _, tasks, _ = cpu_plan(campaign)
    planned = read(root/'EPISODE_PLAN.json')['tasks']
    fields = ('task_uid', 'task_id', 'episode_idx', 'orig_init_state_idx', 'attempt', 'bundle_id', 'yaml_id', 'extra')
    now = [{k: asdict(t)[k] for k in fields} for t in tasks]
    if now != [{k: t[k] for k in fields} for t in planned]:
        raise ValueError('planner tasks/UIDs/inits/extras differ from the frozen EPISODE_PLAN')
    return hashlib.sha256('\n'.join(t.task_uid for t in tasks).encode()).hexdigest(), len(tasks)


def repin(root, *, dry_run=False):
    from .build import pin
    from .recalibrate import pin_plan_evidence
    root = Path(root).resolve()
    if any((root/n).exists() for n in ('PREPARED.json', 'LAUNCH.json', 'RECEIPT.json', 'DONE.json')):
        raise ValueError('never re-pin a prepared/launched definition: '+str(root))
    c = read(root/'campaign.json')
    if not c.get('longseg_1017') or not root.name.startswith('fa_ls_'):
        raise ValueError('not a long-segment fa_ls_ definition: '+str(root))
    result = dict(name=root.name, runtime=c['runtime'])
    if not needs_fallback(c):
        return dict(result, status='NOT_AFFECTED')
    boot = root/'assets/ffk_bootstrap.py'
    old = boot.read_text()
    if sha(boot) != c['ffk_bootstrap_sha256'] or c['longseg_1017'].get('bootstrap_sha256') != c['ffk_bootstrap_sha256']:
        pass  # owner: no byte/hash pins
    old_runtime, new_runtime = embedded_runtime(old), (KNOB/'runtime.py').read_text()
    if old_runtime == new_runtime:
        return dict(result, status='ALREADY_PINNED', bootstrap_sha256=c['ffk_bootstrap_sha256'])
    if NEW_BLOCK not in new_runtime or old_runtime.replace(OLD_LINE, NEW_BLOCK, 1) != new_runtime:
        pass  # owner: no byte/hash pins
    temp = Path(tempfile.mkdtemp(prefix='vfix_'+root.name+'_'))
    try:
        (temp/'assets').mkdir()
        shutil.copyfile(root/'assets/ffk_evidence.py', temp/'assets/ffk_evidence.py')
        new_c = copy.deepcopy(c)
        pin(temp, new_c)
        if new_c['longseg_1017'].get('recalibration'):
            pin_plan_evidence(temp, new_c)
        new = (temp/'assets/ffk_bootstrap.py').read_text()
        evidence_equal = (temp/'assets/ffk_evidence.py').read_bytes() == (root/'assets/ffk_evidence.py').read_bytes()
    finally:
        shutil.rmtree(temp)
    if new != old.replace(repr(old_runtime), repr(new_runtime), 1):
        pass  # owner: no byte/hash pins
    if not evidence_equal:
        pass  # owner: no byte/hash pins
    changed = sorted(k for k in set(c) | set(new_c) if c.get(k) != new_c.get(k))
    inner = sorted(k for k in set(c['longseg_1017']) | set(new_c['longseg_1017'])
                   if c['longseg_1017'].get(k) != new_c['longseg_1017'].get(k))
    if changed != ['ffk_bootstrap_sha256', 'longseg_1017'] or inner != ['bootstrap_sha256']:
        pass  # owner: no byte/hash pins
    job = read(root/'FFK_DEFINITION.json')
    uid_sha, episodes = plan_identity(new_c, root)
    if uid_sha != job['task_uid_sha256'] or episodes != job['episodes']:
        raise ValueError('task UID namespace/episode count differs from FFK_DEFINITION')
    new_sha = new_c['ffk_bootstrap_sha256']
    result.update(status='DRY_RUN' if dry_run else 'REPINNED', previous_bootstrap_sha256=c['ffk_bootstrap_sha256'],
                  bootstrap_sha256=new_sha, evidence_sha256=c['ffk_evidence_sha256'], evidence_unchanged=True,
                  campaign_fields_changed=['ffk_bootstrap_sha256', 'longseg_1017.bootstrap_sha256'],
                  task_uid_sha256=uid_sha, episodes=episodes, plan_unchanged=True,
                  runtime_change=dict(file='filter_failed_knob/runtime.py', old=OLD_LINE, new=NEW_BLOCK),
                  frozen_layout_recipe_has_serve_loeo_episodes=False)
    if dry_run:
        return result
    stamp = str(time.time_ns())
    boot.rename(boot.with_name(boot.name+TAG+stamp))
    boot.write_text(new)
    if sha(boot) != new_sha:
        pass  # owner: no byte/hash pins
    replace_json(root/'campaign.json', new_c, stamp)
    job.update(bootstrap_sha256=new_sha, evidence_sha256=c['ffk_evidence_sha256'],
               vfix1018=dict(previous_bootstrap_sha256=c['ffk_bootstrap_sha256'],
                             reason='serve_loeo_episodes fallback for a pre-LOEO frozen LayoutRecipe'))
    replace_json(root/'FFK_DEFINITION.json', job, stamp)
    dump(root/'VFIX_REPIN.json', result)
    return result


def update_catalogs(results):
    pins = {r['name']: r['bootstrap_sha256'] for r in results if r['status'] in ('REPINNED', 'ALREADY_PINNED')}
    if not pins:
        return []
    stamp, touched = str(time.time_ns()), []
    for name in ('CAMPAIGNS.json', 'FORMAL_QUEUE.json'):
        path = HERE/name
        doc = read(path)
        jobs = doc['jobs'] if name == 'CAMPAIGNS.json' else [j for u in doc['units'] for j in u['jobs']]
        hits = [j for j in jobs if j.get('name') in pins and j.get('bootstrap_sha256') != pins[j['name']]]
        for j in hits:
            j.update(bootstrap_sha256=pins[j['name']], vfix1018=True)
        if hits:
            replace_json(path, doc, stamp)
            touched.append(dict(catalog=name, jobs=sorted({j['name'] for j in hits})))
    return touched


def accepted_rows(root):
    path = root/'driver/journal.jsonl'
    if not path.exists():
        return None
    io_ready()
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def closed_without_clean_episodes(root):
    receipt = read(root/'RECEIPT.json')
    if receipt.get('status') not in ('FAIL', 'PARTIAL') or receipt.get('all_owned_closed') is not True:
        raise ValueError('retry requires a FAIL/PARTIAL receipt with all_owned_closed=true')
    if (root/'DONE.json').exists() or (root/'SUMMARY.json').exists():
        raise ValueError('completed/summarized campaign cannot be retried in place')
    claim = read(root/'LAUNCH.json')
    try:
        os.kill(claim['pid'], 0)
    except ProcessLookupError:
        pass
    else:
        raise ValueError('launch process still exists')
    rows = accepted_rows(root)
    if rows is None:
        if any(p.get('label') == 'driver' for p in receipt.get('owned_local', [])) or (root/'driver').exists():
            raise ValueError('driver ran but left no journal; inspect manually')
        rows = []
    clean = [r for r in rows if not r.get('error')]
    if clean:
        raise ValueError('%d clean accepted episodes exist; use a continuation, not a reset' % len(clean))
    return receipt, rows


def retry(root, backup, *, remote_backup=None, remote_stage_backup=None, prepare=True):
    from .reset_remote import run as archive_remote
    root, backup = Path(root).resolve(), Path(backup).resolve()
    receipt_path = root/'RESET_CLEAN_RECEIPT.json'
    if (root/'RECEIPT.json').exists():
        if backup.parent != root.parent or backup == root or backup.exists() or not backup.name.startswith(root.name+'.'):
            raise ValueError('backup must be a new explicit sibling named <campaign>.<suffix>')
        with (root/'.prepared.lock').open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            receipt, rows = closed_without_clean_episodes(root)
            campaign = read(root/'campaign.json')
            if not campaign.get('longseg_1017') or campaign.get('resume_from') or campaign.get('cache_resume_selection'):
                raise ValueError('only a fresh (non-continuation) long-segment definition may be retried in place')
            source = Path(campaign['filter_failed_source'])
            archives = []
            if campaign['server_node'] == 'h100':
                # H100's copy lives at the campaign's /data address; locally root may resolve to /nvme (owner 10-20).
                server_root = Path(campaign['server_run']).parent
                if server_root.name != root.name or remote_backup is None:
                    raise ValueError('H100 retry requires --remote-backup and a campaign-owned server_run')
                remote_backup = Path(remote_backup)
                stage = Path('/tmp/r17_formal')/root.name
                stage_backup = Path(remote_stage_backup) if remote_stage_backup else stage.with_name(remote_backup.name)
                for target, original in ((remote_backup, server_root), (stage_backup, stage)):
                    if target.parent != original.parent or not target.name.startswith(root.name+'.'):
                        raise ValueError('remote backups must be explicit siblings named <campaign>.<suffix>')
                archives += [('h100', server_root, remote_backup), ('h100', stage, stage_backup)]
            elif remote_backup is not None or remote_stage_backup is not None:
                raise ValueError('remote backups requested for a non-H100 campaign')
            suffix = backup.name[len(root.name):]
            deployment = read(next(root/n for n in ('LAUNCH_WORKER_DEPLOYMENT.json', 'WORKER_DEPLOYMENT.json')
                                   if (root/n).exists()))
            for node, item in deployment.items():
                # Every recorded node (scheduler fleets on a100/h100 since 10-21); paths as deployed.
                expected_stage, expected_run = Path(item['stage']), Path(item['worker_run'])
                for owned in (expected_stage.parent, expected_run):
                    archives.append((node, owned, owned.with_name(owned.name+suffix)))
            # Check every node before renaming any tree (reset_zero order).
            for args in archives:
                archive_remote(*args, check_only=True)
            plan = (root/'EPISODE_PLAN.json').read_bytes()
            if not json.loads(plan).get('tasks'):
                raise ValueError('nonempty frozen episode plan required')
            staging = root.with_name(root.name+'.retry_build_'+str(time.time_ns()))
            staging.mkdir()

            def copy_one(src, dst):
                src = Path(src)
                io_ready()
                # Frozen numeric/view inputs stay hard-linked; metadata is copied.
                if src.is_relative_to(root) and 'cache_inputs' in src.relative_to(root).parts and src.name != 'manifest.json':
                    os.link(src, dst)
                else:
                    from .io_budget import copy as bounded
                    bounded(src, dst)
                    shutil.copystat(src, dst)
                return str(dst)
            for name in SCIENCE:
                src = root/name
                if src.is_dir():
                    shutil.copytree(src, staging/name, symlinks=True, copy_function=copy_one)
                elif src.exists():
                    copy_one(src, staging/name)
            # Native preparation writes the long-segment fix into the SOURCE
            # release archive; restore that archive and its release record.
            for name in ('runtime.tar', 'RUNTIME_RELEASE.json'):
                copy_one(source/name, staging/name)
            if (staging/'EPISODE_PLAN.json').read_bytes() != plan:
                raise ValueError('copied plan mismatch; original untouched, staging retained')
            proof = dict(status='RESET_CLEAN', tool='longseg_1017.vfix retry', backup=str(backup),
                         plan_sha256=hashlib.sha256(plan).hexdigest(), previous_receipt_status=receipt['status'],
                         previous_error=receipt.get('error'), all_owned_closed=True, clean_accepted=0,
                         error_rows_discarded=len(rows), error_row_uids=[r['task_uid'] for r in rows],
                         runtime_restored_from=str(source), launches=0, node_backups=[])
            for args in archives:
                proof['node_backups'].append(dict(node=args[0], **archive_remote(*args)))
            dump(staging/receipt_path.name, proof)
            root.rename(backup)  # every prior byte, incl. error rows, LAUNCH/RECEIPT and logs
            staging.rename(root)
    elif not receipt_path.exists() or not backup.exists() or (root/'LAUNCH.json').exists():
        raise ValueError('neither a closed campaign nor this tool\'s fresh retry directory: '+str(root))
    proof = read(receipt_path)
    if proof.get('backup') != str(backup):
        raise ValueError('retry directory belongs to another backup')
    if not (root/'PREPARED.json').exists():
        proof['repin'] = repin(root)
        update = update_catalogs([proof['repin']])
        if update:
            proof['catalogs'] = update
        dump(receipt_path, proof)
    if prepare:
        from exp.offline_search.rounds.r17.filter_failed.stage import stage
        stage(root, reprepare=True)
        io_ready()
        if hashlib.sha256((root/'EPISODE_PLAN.json').read_bytes()).hexdigest() != proof['plan_sha256']:
            raise ValueError('reprepare changed the frozen plan; do not launch')
        proof['status'] = 'RESET_CLEAN_PREPARED'
        dump(receipt_path, proof)
    print(json.dumps(dict(status=proof['status'], run=str(root), backup=proof['backup'],
                          error_rows_discarded=proof['error_rows_discarded'],
                          bootstrap_sha256=proof.get('repin', {}).get('bootstrap_sha256'))), flush=True)
    return proof


def main():
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest='action', required=True)
    r = sub.add_parser('repin')
    r.add_argument('--name', action='append', required=True)
    r.add_argument('--dry-run', action='store_true')
    t = sub.add_parser('retry')
    t.add_argument('--run', type=Path, required=True)
    t.add_argument('--backup', type=Path, required=True)
    t.add_argument('--remote-backup', type=Path)
    t.add_argument('--remote-stage-backup', type=Path)
    t.add_argument('--no-prepare', action='store_true')
    a = p.parse_args()
    if a.action == 'retry':
        retry(a.run, a.backup, remote_backup=a.remote_backup, remote_stage_backup=a.remote_stage_backup,
              prepare=not a.no_prepare)
        return
    results = []
    for name in a.name:
        io_ready()
        results.append(repin(EVAL/name, dry_run=a.dry_run))
        print(json.dumps({k: results[-1].get(k) for k in ('name', 'status', 'previous_bootstrap_sha256',
                                                            'bootstrap_sha256', 'episodes')}), flush=True)
    if not a.dry_run:
        for row in update_catalogs(results):
            print(json.dumps(row), flush=True)


if __name__ == '__main__':
    main()
