"""Portable diagnostic at an anchor, consuming the deployed retriever's rows/weights.

No simulator, policy calls, action-coordinate convention, cameras or benchmark constants.
The calibration arrays must be indexed in the same row order as the deployed library.
This is a reconstruction/support diagnostic, NOT a success probability or MISS utility.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class Calibration:
    action_error: np.ndarray
    successor_error: np.ndarray
    task: np.ndarray
    episode: np.ndarray
    loeo_distance: np.ndarray
    pair_scale: dict
    provenance: str = 'conditional LOEO in frozen deployed representation'
    action_factor: float = 1.0
    successor_factor: float = 1.0

    def at_anchor(self, task, rows, weights, nearest_distance):
        rows=np.asarray(rows,dtype=int);w=np.asarray(weights,dtype=float)
        if len(rows)!=len(w) or not len(w) or np.any(w<0) or w.sum()<=0:
            raise ValueError('Need nonempty nonnegative deployed kernel weights.')
        w=w/w.sum();sel=self.task==task
        if not sel.any() or np.any(self.task[rows]!=task):
            raise ValueError('Calibration and retrieval task/candidate identities differ.')
        d=self.loeo_distance[sel];d=d[np.isfinite(d)]
        scale=float(self.pair_scale[task])
        if not len(d) or not scale>0:raise ValueError('No valid cross-episode distance calibration.')
        action=float(self.action_factor*(w@self.action_error[rows]));succ=self.successor_error[rows]
        known=np.isfinite(succ);mass=float(w[known].sum())
        prior=self.successor_error[sel];prior=prior[np.isfinite(prior)]
        successor=float(self.successor_factor*(w@np.where(known,succ,prior.mean()))) if len(prior) else None
        ratio=float(nearest_distance/scale)
        # Missing residual labels use the task residual mean; mass exposes how much is imputed.
        quality=None if successor is None else float(1/(1+ratio+action+successor))
        return dict(quality=quality,distance_ratio=ratio,expected_action_error=action,
                    expected_successor_error=successor,successor_evidence_mass=mass,
                    internal_support_p=float((1+np.sum(d>=nearest_distance))/(len(d)+1)),
                    internal_95pct_covered=bool(nearest_distance<=np.quantile(d,.95)),
                    calibration_provenance=self.provenance)


def load_audit_calibration(directory,tag):
    """Adapter for this report's artifacts; the Calibration class is benchmark independent."""
    import json,pathlib
    p=pathlib.Path(directory);meta=json.loads((p/(tag+'.json')).read_text())
    a=np.load(p/(tag+'_loeo.npz'))
    return Calibration(a['err10'],a['succ2'],a['task'],a['ep'],a['d1'],
                       {int(t):s['pair'] for t,s in meta['scales'].items()})
