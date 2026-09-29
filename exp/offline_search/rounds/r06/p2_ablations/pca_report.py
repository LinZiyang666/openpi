"""Aggregate completed direct-PCA evidence; no experiment results are inferred."""
import hashlib
import json
from pathlib import Path
from exp.offline_search.rounds.r06.p2_ablations.make_arms import HERE


def main():
    d=HERE/'results/pca'
    load=lambda p:json.loads(p.read_text())
    validation=load(d/'validation.json'); tokens=load(d/'token_equality.json'); resources=load(d/'resources.json')
    ladder=load(d/'ladder.json')
    grid4=load(d/'grid4_exact_payloads.json')
    checks=sorted([load(p) for p in d.glob('r6p2_direct_*_check.json')],key=lambda x:x['arm'])
    smokes=[load(d/f'smoke_{m}.json') for m in ('pi05','groot')]
    assert len(checks)==8 and all(r['PASS'] for r in checks+smokes+[validation,tokens,ladder,grid4]) and not resources['failed']
    result=dict(PASS=True,token_equality=tokens,ladder=ladder,grid4_exact_payloads=grid4,resources=resources,artifacts=validation['artifacts'],checks=checks,smokes=smokes,
                totals=dict(recorded_queries=sum(r['recorded_queries'] for r in checks),
                            learned_metric_refit_checks=sum(r['learned_metric_refit_checks'] for r in checks),
                            blind_tail_checks=sum(r['blind_tail_checks'] for r in checks),
                            smoke_episodes=sum(r['episodes'] for r in smokes),smoke_decisions=sum(r['decisions'] for r in smokes),
                            fit_bytes=sum(r['bytes'] for r in validation['artifacts'])))
    (d/'summary.json').write_text(json.dumps(result,indent=1)+'\n')
    fits={r['arm']:r for r in validation['artifacts']}
    lines=['# Direct-token PCA: observed results','',
      '| arm | total fit s | PCA s v0 / v1 | explained variance v0 / v1 | process peak GiB | projection median ms (2 cameras, 3 threads) | A median ms |',
      '|---|---:|---:|---:|---:|---:|---:|']
    for c in checks:
        f=fits[c['arm']];v0,v1=f['pca']['v0'],f['pca']['v1'];tm=c['cpu_timings']['3']
        lines.append(f"| {c['arm']} | {f['fit_seconds']:.1f} | {v0['wall_s']:.1f} / {v1['wall_s']:.1f} | {v0['explained_variance_ratio_sum']:.4f} / {v1['explained_variance_ratio_sum']:.4f} | {max(v0['process_peak_rss_bytes'],v1['process_peak_rss_bytes'])/2**30:.2f} | {tm['two_camera_projection_median_ms']:.3f} | {tm['A_two_camera_projection_median_ms']:.3f} |")
    lines+=['',f"Peak combined fit-worker RSS (sampled every 2 s): {resources['peak_children_rss_bytes']/2**30:.2f} GiB. Four workers × three BLAS threads plus the one-thread coordinator; the single three-thread check process fits within 16 total. Each projection matrix is 134,217,728 bytes (128 MiB), or 256 MiB for two cameras; means add 4 MiB.",'',
      'CPU timings are observed under shared load, with recorded token arrays already on CPU. They exclude stage-1 inference and live GPU-to-CPU transfer. Full-query timings and projection p95 are in summary.json. GPU was not run under the CPU-only constraint.','',
      '## Held-out recorded-query sanity checks (descriptive)','',
      '| arm | stream | queries | mean top-16 overlap with A | mean sigma-RMS action error: direct | A |',
      '|---|---|---:|---:|---:|---:|']
    for c in checks:
        for r in c['descriptive']:
            lines.append(f"| {c['arm']} | {r['stream']} | {r['queries']} | {r['mean_top16_overlap']:.4f} | {r['mean_sigma_RMS_action_error_direct']:.4f} | {r['mean_sigma_RMS_action_error_A']:.4f} |")
    totals=result['totals']
    lines+=['',f"PASS: {totals['recorded_queries']:,} query comparisons with stock BlindAWM on the new visual features and with exact grid4 A control; unchanged state block; {totals['learned_metric_refit_checks']} independent main/early metric refits; {totals['blind_tail_checks']:,} anchor-tail checks.",
      '',f"The grid4 control separately matches all payload fields, including the optional still diagnostic, on {grid4['queries']:,} recorded queries. The direct-grid comparison excludes only that unused diagnostic when prior full tokens are unavailable.",
      '',f"Real-plugin smoke PASS: 2 arms × 4 complete recorded episodes, {totals['smoke_decisions']} decisions, no MISS, one blind tail maximum. Token extraction/serialization PASS on {tokens['cases']} recorded query rows and {tokens['library_cases']} library rows, both cameras; exact HDF5/store/serializer/plugin values. Library and query source-file sets are disjoint. See token_equality.json for the existing pi05 bf16-pooling vs f32-pooling difference.",'',
      '## Final prefits','', '| file | bytes | SHA256 |','|---|---:|---|']
    for f in validation['artifacts']:
        lines.append(f"| {Path(f['path']).name} | {f['bytes']} | `{f['sha256']}` |")
    (d/'TABLES.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(totals),flush=True)


if __name__=='__main__':main()
