"""Every grid's pooling against a spatial-block reference on recorded tokens."""
import json
import numpy as np
from exp.offline_search.harness import store
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE, STORE
from exp.offline_search.rounds.r06.p2_ablations.token_pca import pool_tokens, TokenPCAAWM


def main():
    cases=[]
    for model in ('pi05','groot'):
        q=store.QueryCell(STORE,f'{model}_spatial_cache')
        for camera in ('v0','v1'):
            t=np.array(q.tok(camera)[0],np.float32).reshape(16,16,2048)
            for grid in (1,2,4,8,16):
                b=16//grid
                ref=np.stack([t[i*b:(i+1)*b,j*b:(j+1)*b].mean(axis=(0,1))
                              for i in range(grid) for j in range(grid)]).reshape(-1)
                got=pool_tokens(t.reshape(256,2048),grid)
                assert got.shape==(grid*grid*2048,) and got.dtype==np.float32
                error=float(np.max(np.abs(got-ref)))
                assert np.allclose(got,ref,atol=2e-5,rtol=2e-6),(grid,error)
                if grid==16: assert np.array_equal(got,t.reshape(-1))
                m=TokenPCAAWM(pooling_grid=grid,lib='current',serving='anchor_tail',budget=1,gates='budget_only')
                assert m.k==16 and m.pooling_grid==grid
                cases.append(dict(model=model,camera=camera,grid=grid,dims=len(got),reference_max_abs_error=error))
    for bad in (0,3,5,32):
        try: TokenPCAAWM(pooling_grid=bad)
        except ValueError: pass
        else: raise AssertionError(bad)
    out=dict(PASS=True,pooling_cases=cases,invalid_grid_refusals=4)
    (HERE/'results/pca/ladder.json').write_text(json.dumps(out,indent=1)+'\n')
    print(json.dumps(out))


if __name__=='__main__':main()
