"""Tests for the Phase-6 projection label/mask/fold core (plan §A/§C)."""

import numpy as np

from exp.zixuan_proposal.projection_labels import (
    Entry,
    assign_folds,
    build_masks,
    compat_matrix,
    represented_tasks,
    valid_anchor_fraction,
    whiten_flatten,
)


def test_whiten_flatten_shape_and_scale():
    H, adim = 4, 10
    chunk = np.ones((H, adim))
    active = np.array([True] * 7 + [False] * 3)
    sigma = np.array([2.0] * adim)
    out = whiten_flatten(chunk, active, sigma)
    assert out.shape == (7 * H,)  # only active dims kept, H read from array
    assert np.allclose(out, 1.0 / (2.0 + 1e-8))


def _entry(ident, outcome, a):
    return Entry(ident=ident, outcome=outcome, action_flat=np.array(a, dtype=float), snap_flat={})


def test_compat_matrix_eta1_properties():
    ents = [_entry((0, 0), 1, [0, 0]), _entry((0, 2), 1, [0.1, 0]), _entry((1, 0), 1, [5, 5])]
    c = compat_matrix(ents, {"sigma_A_sq": 0.5, "eta": 1.0})
    assert np.allclose(np.diag(c), 1.0)  # self-compat = 1
    assert np.allclose(c, c.T)  # symmetric
    assert c[0, 1] > c[0, 2]  # closer action -> higher compat


def test_masks_symmetry_disjoint_and_action_compatible_dminus_excluded():
    # e2 is an action-compatible D- (close to successes) -> must NEVER be a negative.
    ents = [
        _entry((0, 0), 1, [0.0, 0]),
        _entry((0, 2), 1, [0.1, 0]),
        _entry((0, 4), -1, [0.05, 0]),  # action-compatible D- (high c to successes)
        _entry((0, 6), -1, [9.0, 9]),  # action-dissimilar D- (low c)
    ]
    c = compat_matrix(ents, {"sigma_A_sq": 0.3, "eta": 1.0})
    pos, neg = build_masks(ents, c, rho_plus=0.8, rho_minus=0.3)
    assert np.allclose(pos, pos.T) and np.allclose(neg, neg.T)  # symmetric
    assert not (pos & neg).any()  # disjoint
    assert pos[0, 1] and pos[1, 0]  # the two close successes are mutual positives
    assert not pos[0, 2]  # e2 is a failure -> never a positive
    assert not neg[0, 2] and not neg[2, 0]  # action-compatible D- is NOT a negative
    assert neg[0, 3]  # action-dissimilar D- IS a negative (outcome-agnostic)


def test_valid_anchor_fraction():
    pos = np.zeros((3, 3), dtype=bool)
    neg = np.zeros((3, 3), dtype=bool)
    pos[0, 1] = pos[1, 0] = True
    neg[0, 2] = neg[2, 0] = True
    # anchor 0 has both a positive and a negative; 1 and 2 do not
    assert valid_anchor_fraction(pos, neg) == 1 / 3


def test_assign_folds_exact_rule():
    # task 0 has 4 even inits -> last=test, second-last=val, rest=train.
    # task 1 has 2 even inits -> all train (n<3), not represented.
    idents = [(0, 0), (0, 2), (0, 4), (0, 6), (1, 0), (1, 2)]
    fold = assign_folds(idents)
    assert fold[(0, 6)] == "test"
    assert fold[(0, 4)] == "val"
    assert fold[(0, 0)] == "train" and fold[(0, 2)] == "train"
    assert fold[(1, 0)] == "train" and fold[(1, 2)] == "train"
    assert represented_tasks(fold) == {0}
