"""Pure reference arithmetic for the pre-look atlas; no server, I/O or launches.

Actual camera/state histories and action commits belong to the serving plugin.
All proposal operations here are pure. Missing current-camera data is never read.
"""
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True)
class AtlasConfig:
    label_rms: float = .15
    required_mass: float = .75
    support_mass: float = .99
    state_valve: float = .2
    lag: int = 2
    warmup: int = 4

def label_safe(full_action, half_action, sigma, threshold):
    f,h=np.asarray(full_action),np.asarray(half_action)
    rms=np.sqrt(np.mean(((f[...,:6]-h[...,:6])/np.asarray(sigma)[:6])**2,axis=(-2,-1)))
    grip=((f[...,6]>=0)==(h[...,6]>=0)).all(-1)
    return (rms<=threshold)&grip

def plan(config, *, step, last_full_step, current_rs, last_full_rs,
         rows, weights, next2, library_rs, safe, state_scale,
         pending_no_progress=False, periodic_due=False, lifecycle_valid=True):
    """No current keys/actions/ground truth are arguments to this planner."""
    if (step<config.warmup or step-last_full_step!=config.lag or pending_no_progress
            or periodic_due or not lifecycle_valid):
        return dict(half=False,reason='lifecycle_or_due')
    rows=np.asarray(rows);w=np.asarray(weights,np.float32)
    if (rows.ndim!=1 or w.shape!=rows.shape or len(rows)==0 or not np.isfinite(w).all()
            or np.any(w<0) or w.sum()<=0 or not np.isfinite(current_rs).all()
            or not np.isfinite(last_full_rs).all()):
        return dict(half=False,reason='invalid')
    w=w/w.sum();nxt=next2[rows];ok=nxt>=0;idx=np.maximum(nxt,0)
    support=float((w*ok).sum());mass=float((w*ok*safe[idx]).sum())
    dw=w*ok/max(support,1e-9)
    expected=np.einsum('k,kd->d',dw,library_rs[idx]-library_rs[rows])
    residual=float(np.sqrt(np.mean(((current_rs-last_full_rs-expected)/state_scale)**2)))
    take=bool(support>=config.support_mass and mass>=config.required_mass and residual<=config.state_valve)
    return dict(half=take,reason='safe' if take else 'atlas_or_state',support=support,safe_mass=mass,state_error=residual)

def half_features(*, last_full_v0, current_wrist, current_rs, last_full_rs):
    """Projected camera vectors: 64+64+8+8; current v0 is intentionally absent."""
    vals=[np.asarray(x,np.float32) for x in (last_full_v0,current_wrist,current_rs,last_full_rs)]
    if [x.shape for x in vals]!=[(64,),(64,),(8,),(8,)]:raise ValueError('feature shape')
    if not all(np.isfinite(x).all() for x in vals):raise ValueError('nonfinite half features')
    p0,p1,rs,rs0=vals
    return np.concatenate([p0,p1,rs,rs-rs0]).astype(np.float32)
