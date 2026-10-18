"""Validate real frozen-planner pilot manifests without starting any worker."""
import json
import copy
from pathlib import Path
import unittest
from artifacts import digest, sha
from normalize_formal import index_plan


WORK = Path(__file__).parent
OLD = {'pi05':'pi05_v2', 'groot':'groot_longseg_recal1018',
       'dp':'dp_longseg_recal1018','fastwam':'fastwam_4090_reduced','cosmos':'cosmos'}


class ClosedloopPilotTest(unittest.TestCase):
    def test_all_tasks_official_inits_seeds_and_arms_preserved(self):
        for model, old in OLD.items():
            root = WORK/'closedloop_pilot_1018'/model
            registration=json.loads((root/'PREREGISTRATION.json').read_text())
            ninit=10 if model in ('pi05','groot') else 50
            paired=[]
            for arm in ('pure','cache'):
                with self.subTest(model=model,arm=arm):
                    path=root/arm/'PLAN.json';plan=json.loads(path.read_text())
                    scientific={k:v for k,v in plan.items() if k!='identity'}
                    self.assertEqual(digest(scientific),plan['identity'])
                    self.assertEqual(sha(path),registration['arms'][arm]['plan_sha256'])
                    reference=json.loads((WORK/'closedloop'/old/arm/'PLAN.json').read_text())
                    expected=[t for t in reference['tasks'] if t['orig_init_state_idx']<ninit]
                    def episode_identity(task):
                        task=copy.deepcopy(task)
                        # The frozen planner binds the selected index set into
                        # scheduling IDs and records its length. MT50 make_env
                        # still selects MT1(task, seed).train_tasks[original_idx].
                        task.pop('task_uid');task.pop('yaml_id')
                        task['extra'].pop('num_trials_per_task')
                        return task
                    self.assertEqual([episode_identity(t) for t in plan['tasks']],
                                     [episode_identity(t) for t in expected])
                    self.assertEqual(len(plan['tasks']),500)
                    self.assertEqual(len({t['task_uid'] for t in plan['tasks']}),500)
                    benchmark=json.loads((root/arm/'assets/benchmark.json').read_text())
                    indexed=index_plan(plan,benchmark)
                    cells={(k['task_id'],k['init_idx'],k['env_seed'],k['policy_seed']) for _,k in indexed.values()}
                    self.assertEqual(len(cells),500)
                    self.assertEqual({t['orig_init_state_idx'] for t in plan['tasks']},set(range(ninit)))
                    self.assertTrue(all(t['extra']['num_trials_per_task'] in (ninit,50) for t in plan['tasks']))
                    paired.append(cells)
            if len(paired)==2:self.assertEqual(paired[0],paired[1])

    def test_formal_worker_count_is_not_stress_client_override(self):
        for model,workers in [('pi05',112),('groot',40),('dp',48),('fastwam',12),('cosmos',44)]:
            for arm in ('pure','cache'):
                campaign=json.loads((WORK/'closedloop_pilot_1018'/model/arm/'campaign.json').read_text())
                self.assertEqual(campaign['execution_workers'],workers)
                self.assertEqual(campaign['workers'],workers)


if __name__=='__main__':unittest.main()
