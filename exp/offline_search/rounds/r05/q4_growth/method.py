"""AWM CL2 over an immutable grown store library, with refit or frozen-50 fit."""
import copy
import numpy as np
from exp.offline_search.harness import api
from exp.offline_search.rounds.r02.g1_awm.awm import AWM, _Task
from exp.offline_search.rounds.r05.q4_growth.common import (
    base_artifact, features, load_base, readonly, sha256,
)


class GrowthAWM(AWM):
    family = 'q4_growth'

    def __init__(self, library='grow250', variant='frozen', kref=5):
        if variant not in ('refit', 'frozen'):
            raise ValueError('variant must be refit or frozen')
        if int(kref) != 5:
            raise ValueError('Q4 holds the deployed 50-library CL2 kref=5 fixed')
        super().__init__(lib='current', kref=5)
        self.library, self.variant = library, variant
        self.name = f'AWM_growth_{library}_{variant}_kr5'

    def fit(self, lib, ctx):
        grown = ctx.open_library(self.library)
        if grown.ids[:lib.L] != lib.ids:
            raise api.ContractError('growth must retain every current ID at its original row')
        if self.variant == 'refit':
            # AWM's current branch fits the supplied LibraryView. Its PCA cache
            # fingerprint will not match grow250, so it runs the normal recipe.
            super().fit(grown, ctx)
        else:
            self._append_frozen(lib, grown, ctx)
        self.cand_name = self.os_library = self.library
        self.os_fit_library = self.library if self.variant == 'refit' else 'current'
        self.name = f'AWM_growth_{self.library}_{self.variant}_kr5'
        self.growth_manifest_sha256 = sha256(grown.dir / 'manifest.json')
        readonly(self)

    def _append_frozen(self, lib, grown, ctx):
        base = load_base(ctx.suite)
        if not np.array_equal(base.act, lib.action):
            raise api.ContractError('deployed fit actions differ from current library')
        settings = {k: getattr(self, k) for k in ('library', 'variant', 'name')}
        self.__dict__.update(copy.deepcopy(vars(base)))
        self.__dict__.update(settings)
        for task, old in list(self.tasks.items()):
            self.tasks[task] = _Task()
            self.tasks[task].__dict__.update(vars(old))
        self.frozen_artifact = str(base_artifact(ctx.suite))
        self.frozen_artifact_sha256 = sha256(self.frozen_artifact)
        x = features(self, grown)
        sig64 = np.asarray(ctx.action_sigma, np.float64)
        for task, T in self.tasks.items():
            added = grown.rows_of_task(task)
            added = added[added >= lib.L]
            xx = x[added]
            z = (xx @ T.Wf - T.shift).astype(np.float32)
            # Do not reproject old codes: exact old candidate bytes are retained.
            old_rows = T.rows.copy()
            T.rows = np.concatenate((T.rows, added))
            T.Z = np.concatenate((T.Z, z))
            T.z2 = np.concatenate((T.z2, np.sum(z.astype(np.float64)**2, axis=1).astype(np.float32)))
            hd = (np.asarray(grown.action[added, :5, :7], np.float64) / sig64).reshape(-1, 35)
            rs = np.asarray(grown.rs[added, :8], np.float64)
            for field, val in (('HD', hd.astype(np.float32)), ('h2', (hd**2).sum(1).astype(np.float32)),
                               ('RS', rs.astype(np.float32)), ('rs2', (rs**2).sum(1).astype(np.float32))):
                setattr(T, field, np.concatenate((getattr(T, field), val)))
            if T.Z0 is not None or T.As0 is not None:
                raise api.ContractError('Q4 frozen CL2 requires full-rank joint codes')
            # Match the deployed implicit early-code layout using its frozen A0.
            y = z.astype(np.float64) @ T.A0.astype(np.float64)
            T.n20 = np.concatenate((T.n20, np.sum(y*y, axis=1).astype(np.float32)))
            for fi, sl in ((0, slice(0, 64)), (1, slice(64, 128))):
                # Historical R2 CL2 artifacts predate G3's cached visual aux.
                # Reconstruct those unused-by-CL2 centers from current only.
                center = getattr(T, f'Vm{fi}', x[old_rows, sl].astype(np.float64).mean(0).astype(np.float32))
                setattr(T, f'Vm{fi}', center)
                v = x[T.rows, sl].astype(np.float64) - center
                v /= np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)
                setattr(T, f'V{fi}', v.astype(np.float32))
        self.act = np.array(grown.action, copy=True)
        self.lib_ep = np.array(grown.episode, dtype=np.int32, copy=True)
        self.lib_step = np.array(grown.step, dtype=np.int32, copy=True)
        self.n_cand = {t: len(T.rows) for t, T in self.tasks.items()}
        self.prof = api.NULL_PROFILER
