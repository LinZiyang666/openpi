"""Read-only CPU audit of existing MT50 journals; outputs only beside this script.

This is a NEW descriptive aggregation, not a rerun of registered statistics.
No repository imports, simulator, model, GPU, or network calls.
"""
import collections
import hashlib
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent
ROOT = Path('/data/mw_finetune/eval')
RUNS = [
    ('pi05', 10, 'tpi7_0536'),
    ('pi05', 1, 'tpi7k1_0922'), ('pi05', 2, 'tpi7k2_1135'),
    ('pi05', 3, 'tpi7k3_1232'), ('pi05', 4, 'tpi7k4_1335'),
    ('pi05', 5, 'tpi7k5_1440'),
    ('groot', 4, 'tgr7_0502'),
    ('groot', 1, 'tgr7k1_1021'), ('groot', 2, 'tgr7k2_1044'),
    ('groot', 3, 'tgr7k3_1107'),
    ('fastwam', 10, 'tfw7full_0944'),
    ('fastwam', 1, 'tfw7k1_0908'), ('fastwam', 2, 'tfw7k2_1024'),
    ('fastwam', 3, 'tfw7k3_1051'), ('fastwam', 4, 'tfw7k4_1121'),
    ('fastwam', 5, 'tfw7k5_1153'),
    ('dp', 100, 'tdp7_0729'), ('dp', 10, 'tdp7d10_0851'),
    ('groot_H100', 4, 'tgr7h1_full'), ('groot_H100', 2, 'tgr7h1_k2'),
    ('dp_H100', 100, 'tdp7h1_full2'),
]


def read_json(path):
    return json.loads(path.read_text())


def cite(path, needle=None):
    lines = path.read_text().splitlines()
    line = next((i for i, s in enumerate(lines, 1) if needle and needle in s), 1)
    return f'[{path.parent.name}/{path.name}:{line}]({path}:{line})'


def audit(model, k, name):
    run = ROOT / name
    plan = read_json(run / 'EPISODE_PLAN.json')
    spec = read_json(run / 'assets/benchmark.json')
    campaign = read_json(run / 'campaign.json')
    summary = read_json(run / 'SUMMARY.json')
    tasks = {t['task_uid']: t for t in plan['tasks']}
    difficulty = {row[0]: row[2] for row in spec['tasks']}
    accepted = {}
    rejected = collections.Counter()
    path = run / 'driver/journal.jsonl'
    nrows = duplicates = 0
    for lineno, line in enumerate(path.open(), 1):
        if not line.strip():
            continue
        nrows += 1
        row = json.loads(line)
        if row.get('accepted') is not True:
            rejected['not_accepted'] += 1
            continue
        if row.get('error') or row.get('status') == 'error':
            rejected['error'] += 1
            continue
        assert row['status'] in {'done', 'failed'} and isinstance(row['success'], bool), (name, lineno)
        task = tasks[row['task_uid']]
        extra = task['extra']
        assert extra['num_steps'] == k and extra['seed'] == 7
        assert extra['commit_horizon'] == 5
        key = (extra['task_name'], extra['seed'], task['orig_init_state_idx'])
        if key in accepted:
            duplicates += 1
            assert accepted[key]['success'] == row['success'], (name, key)
        accepted[key] = {'success': row['success'], 'line': lineno, 'attempt': row['attempt']}
    assert len(accepted) == 2500, (name, len(accepted))
    by_task = collections.defaultdict(list)
    for (task, seed, idx), row in accepted.items():
        by_task[task].append((idx, row['success']))
    assert len(by_task) == 50
    for task, rows in by_task.items():
        assert sorted(idx for idx, _ in rows) == list(range(50)), (name, task)
    task_sr = {t: sum(s for _, s in v) / len(v) for t, v in by_task.items()}
    groups = collections.defaultdict(list)
    for task, sr in task_sr.items():
        groups[difficulty[task]].append(sr)
    group_sr = {g: 100 * sum(v) / len(v) for g, v in groups.items()}
    counts = {g: len(v) for g, v in groups.items()}
    hashes = {str(run / f): hashlib.sha256((run / f).read_bytes()).hexdigest()
              for f in ['driver/journal.jsonl', 'EPISODE_PLAN.json', 'campaign.json',
                        'assets/benchmark.json', 'SUMMARY.json']}
    return {
        'model': model, 'k': k, 'run': name, 'n': len(accepted),
        'successes': sum(r['success'] for r in accepted.values()),
        'group_mean': sum(group_sr.values()) / len(group_sr),
        'task_mean': 100 * sum(task_sr.values()) / len(task_sr),
        'groups': group_sr, 'group_task_counts': counts, 'task_sr': task_sr,
        'journal_rows': nrows, 'rejected_rows': dict(rejected), 'duplicates': duplicates,
        'seed': spec['seed'], 'commit_horizon': 5,
        'server_node': campaign['server_node'], 'replicas': campaign['replicas'],
        'checkpoint': campaign['checkpoint'],
        'sampler': summary.get('execution_sampler'), 'summary_status': summary['status'],
        'benchmark_protocol': spec['protocol_identity'],
        'hashes': hashes, 'outcomes': accepted,
    }


def main():
    results = [audit(*args) for args in RUNS]
    controls = {}
    for r in results:
        if r['model'] not in controls:
            controls[r['model']] = r
        c = controls[r['model']]
        assert r['benchmark_protocol'] == c['benchmark_protocol']
        assert set(r['outcomes']) == set(c['outcomes'])
        r['paired_control'] = c['run']
        r['gains'] = sum(v['success'] and not c['outcomes'][key]['success']
                         for key, v in r['outcomes'].items())
        r['losses'] = sum(not v['success'] and c['outcomes'][key]['success']
                          for key, v in r['outcomes'].items())
    lines = [
        '# MT50 原始 journal 审核（本次新增描述统计）', '',
        '只读 CPU；仅 accepted=true、无非空 error 的 done/failed 终局；正常任务失败保留。按 task_uid 对照 EPISODE_PLAN，',
        '再按 (task_name, seed, original_init_idx) 去重；重复有效终局须结局一致。',
        '每臂均核验完整 50 任务×50 初态、seed=7、H_exec=5；四难度组任务数为 28/11/6/5。',
        '组均值为四组 SR 等权；按任务均值为全部任务等权。数字为百分数。',
        '同协议身份可配对，但本次没有跨渲染 GPU 像素复核，也不声称模型噪声或整条轨迹相同。',
        'g/l 是相对同 model 表内首个 full 臂的配对新增成功/失败，纯描述；不计算新显著性或非劣判决。',
        'h1 命名的复测是 H100 主机，不是 H_exec 的变化。DP 的原始 full 与 DDIM10 分属不同服务主机。',
        '逐臂 SHA256、全部纳入行号、逐任务 SR 和采样器语义见 audit.json；原日志不变。', '',
        '| 模型 | K | run | accepted / 成功 | 组均值 | 按任务均值 | easy/medium/hard/very-hard | g/l | 主机 | 排除/重复 | 原始来源 |',
        '|---|---:|---|---:|---:|---:|---|---:|---|---|---|',
    ]
    for r in results:
        p = ROOT / r['run']
        vals = '/'.join(f"{r['groups'][g]:.2f}" for g in ['easy','medium','hard','very_hard'])
        lines.append(f"| {r['model']} | {r['k']} | {r['run']} | {r['n']} / {r['successes']} | "
                     f"{r['group_mean']:.2f} | {r['task_mean']:.2f} | {vals} | {r['gains']}/{r['losses']} | "
                     f"{r['server_node']} | {r['rejected_rows']} / {r['duplicates']} | "
                     f"{cite(p/'driver/journal.jsonl')}；{cite(p/'EPISODE_PLAN.json', 'num_steps')}；"
                     f"{cite(p/'assets/benchmark.json', 'seed')} |")
    lines.extend(['', '没有将早期每任务较少局数的结果与本表池化。没有重新运行任何预注册分析。', ''])
    (OUT / 'MT50_AUDIT.md').write_text('\n'.join(lines))
    serial = []
    for r in results:
        r = dict(r)
        r['outcomes'] = [{'task': key[0], 'seed': key[1], 'idx': key[2], **value}
                         for key, value in sorted(r['outcomes'].items())]
        serial.append(r)
    (OUT / 'audit.json').write_text(json.dumps(serial, ensure_ascii=False, indent=2) + '\n')
    print('\n'.join(lines[:13] + [line.split('| [')[0] for line in lines[13:]]))


if __name__ == '__main__':
    main()
