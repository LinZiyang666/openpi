"""NumPy-only serving math. No task/init identity, dispatch, or per-task scale."""
import numpy as np


def dense_predict(head, x):
    if not head:
        return np.zeros((len(x), 60), np.float32)
    z = np.clip((x - head['mean']) / head['std'], -8, 8)
    f = np.concatenate([z, np.cos(z @ head['w'] + head['bias']) * np.sqrt(2)], 1)
    return f @ head['coef'].T + head['intercept']


def residual(head, table, x, rows, weights):
    w = np.asarray(weights, np.float64)
    w = w / w.sum(axis=-1, keepdims=True)
    local = np.einsum('nk,nko->no', w, table[np.asarray(rows, np.int64)])
    return (dense_predict(head, x) + local).reshape(-1, 10, 6)
