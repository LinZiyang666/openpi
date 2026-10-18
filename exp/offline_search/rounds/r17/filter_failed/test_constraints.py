"""CPU reproduction of the existing fit/serving constraints; no new gates."""
from types import SimpleNamespace

import numpy as np
import pytest

from exp.offline_search.rounds.r17.filter_failed.audit_counts import summarize
from exp.r17_integration.common.contracts import ActionStateLayout
from exp.r17_integration.common.layers import LibraryData, fit_library
from exp.r17_integration.common.layers.recipe import LayoutRecipe
from exp.r17_integration.common.layers.math import fit_metric
from exp.offline_search.rounds.r17.filter_failed.prove_retrieval import compare


def test_filter_does_not_top_up_or_borrow_outside_the_size():
    episodes = [dict(id=i, task=0, success=i != 1) for i in range(4)]
    result = summarize(episodes, [0, 1, 2], id_key='id', task_key='task')
    assert (result['collected'], result['failed'], result['kept']) == (3, 1, 2)
    assert result['kept_episode_ids'] == [0, 2]


def test_empty_and_singleton_tasks_are_counted():
    episodes = [dict(id=0, task=0, success=False), dict(id=1, task=1, success=True)]
    result = summarize(episodes, [0, 1], id_key='id', task_key='task')
    assert result['empty_tasks'] == [0]
    assert result['singleton_tasks'] == [1]


@pytest.mark.parametrize('corrector,episodes', [(False, 1), (True, 2)])
def test_standard_fitter_rejects_insufficient_complete_episodes(corrector, episodes):
    layout = ActionStateLayout(horizon=1, replan_steps=1, valid_action_dim=1,
                               valid_state_indices=(0,), motion_channels=(0,),
                               gripper_channel=None)
    rng = np.random.default_rng(0)
    n = 8 * episodes
    library = LibraryData(features={'vision': rng.normal(size=(n, 2)).astype(np.float32)},
                          state=rng.normal(size=(n, 1)).astype(np.float32),
                          action=rng.normal(size=(n, 1, 1)).astype(np.float32),
                          task_id=np.zeros(n, np.int64), episode=np.repeat(np.arange(episodes), 8),
                          step=np.tile(np.arange(8), episodes), source_digest='cpu-test')
    with pytest.raises(ValueError, match='raw retrieval requires >=2'):
        fit_library(library, layout=layout, task_count=1, corrector=corrector)


def test_standard_retrieval_has_no_empty_task_fallback():
    recipe = LayoutRecipe.__new__(LayoutRecipe)
    recipe.tasks = {}
    with pytest.raises(KeyError):
        recipe._retrieve(SimpleNamespace(task_id=7))


def test_single_episode_metric_itself_has_no_cross_episode_scatter():
    # This calls the actual estimator, bypassing the recipe's episode-count check.
    # Removing that check or disabling correction does not define this metric.
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match='zero-scatter library metric'):
        fit_metric(rng.normal(size=(8, 3)), rng.normal(size=(8, 2)), np.zeros(8, np.int64))


def test_proof_requires_byte_equality_instead_of_numerical_closeness():
    original = np.array([1.0], np.float32)
    changed = np.nextafter(original, np.float32(2.0))
    assert np.allclose(original, changed)
    assert not compare(original, changed)[0]['equal']
