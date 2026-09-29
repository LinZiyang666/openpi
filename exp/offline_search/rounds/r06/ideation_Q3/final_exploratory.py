"""Post-hoc closed-loop audit. Never alters the frozen pilot estimator or sources."""
from __future__ import annotations
import itertools
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import binomtest, spearmanr, t

HERE = Path(__file__).resolve().parent
OUT = HERE / "final_exploratory"
ROOT = Path("/home/weiland/trace_runs/os_closed_loop")
sys.path.insert(0, str(HERE))
from analyze_pilot import clean, sha
from exp.offline_search.rounds.r06.p1_groot_commit.make_arms import source_rows


def save(name, obj):
    (OUT/name).write_text(json.dumps(clean(obj), indent=2, allow_nan=False)+"\n")


def effect(d):
    """One record per task/init; repeated reference seeds are averaged inside init."""
    x = pd.Series(d).astype(float)
    delta = x.mean()
    influence = (x-delta)/len(x)
    by_task = list(influence.groupby(level=0))
    variance = sum(len(g)/(len(g)-1)*((g-g.mean())**2).sum() for _,g in by_task)
    se = float(np.sqrt(variance))
    df = len(x)-len(by_task)
    z=t.ppf(.975,df)
    return dict(delta=float(delta), se=se, lo95=float(delta-z*se), hi95=float(delta+z*se), clusters=len(x), df=df)


def load(run, name, cell, kind, rep=1):
    path=ROOT/run/"runs"/name
    summary_path=path/"summary.json"
    summary=json.loads(summary_path.read_text())
    assert summary["complete"]==500, (name,summary["complete"])
    accepted={}
    journal=path/"client/journal.jsonl"
    for line in journal.open():
        r=json.loads(line)
        if r.get("accepted") and r.get("status") in ("done","failed") and not r.get("error"):
            task,init=map(int,r["task_uid"].rsplit(":",2)[-2:])
            key=(task,init)
            if key in accepted:
                assert accepted[key]==r, (name,key,"conflicting accepted result")
            accepted[key]=r
    assert set(accepted)=={(t,i) for t in range(10) for i in range(50)}, name
    y=pd.Series({k:int(r["success"]) for k,r in accepted.items()}).sort_index()
    y.index.names=["task_id","init"]
    assert y.sum()==summary["success"] and abs(y.mean()-summary["sr"])<1e-12
    ledger=summary["cost_ledger"]
    n,v,m=[float(ledger[k]) for k in ("decisions","vision_decisions","misses")]
    assert np.isclose(ledger["v"],v/n) and np.isclose(ledger["m"],m/n)
    c=.152 if summary["model"]=="pi05" else .148
    nominal_controls=float(ledger["controls"])
    rec=dict(cell=cell,kind=kind,rep=rep,arm=name,run=run,summary=str(summary_path),journal=str(journal),
             summary_sha256=sha(summary_path),journal_sha256=sha(journal),sr=float(y.mean()),n=500,
             vision=v,misses=m,decisions=n,v=v/n,m=m/n,
             owner_IR_request=c*v/n+(1-c)*m/n,owner_IR=(c*v+(1-c)*m)/(nominal_controls/5),
             controls_nominal=nominal_controls,actual_controls=ledger.get("actual_controls"),
             method=summary.get("method"),kwargs=summary.get("kwargs"),collected_at=summary.get("collected_at"))
    return rec,y


def pair(left, right, label, repeated=False):
    lm,ly=left;rm,ry=right
    assert ly.index.equals(ry.index)
    d=ly-ry
    rec=dict(cell=lm["cell"],contrast=label,left=lm["arm"],right=rm["arm"],**effect(d))
    if not repeated:
        plus=int(((ly==1)&(ry==0)).sum());minus=int(((ly==0)&(ry==1)).sum())
        rec.update(plus=plus,minus=minus,discordance=(plus+minus)/len(d),
                   mcnemar_p=binomtest(plus,plus+minus).pvalue if plus+minus else 1.)
    rec["IR_delta"]=lm["owner_IR"]-rm["owner_IR"]
    return rec


def mean_ref(arms):
    first=arms[0][0]
    return dict(cell=first["cell"],arm="mean3:"+first["arm"],sr=np.mean([x[0]["sr"] for x in arms]),
                owner_IR=np.mean([x[0]["owner_IR"] for x in arms])), sum(x[1] for x in arms)/len(arms)


def main():
    OUT.mkdir(exist_ok=False)
    rows=[];pairs=[];objects={};ab={}
    for (model,kind,short),entry in sorted(source_rows().items()):
        cell=model+"_"+short;name=entry["row"]["name"]
        root=entry["source"].split("/")[0] if entry["source"].startswith("r0") else "r06_paper"
        for rep in (1,2,3):
            obj=load(root if rep==1 else "r06_paper",name if rep==1 else name+f"_rep{rep}",cell,kind,rep)
            objects[obj[0]["arm"]]=obj;rows.append(obj[0]);ab[cell,kind,rep]=obj
    ab_table=[]
    for cell in sorted({k[0] for k in ab}):
        aa=[ab[cell,"A",i] for i in (1,2,3)];bb=[ab[cell,"B",i] for i in (1,2,3)]
        aavg,bavg=mean_ref(aa),mean_ref(bb)
        p=pair(bavg,aavg,"Bmean3-Amean3",True);pairs.append(p)
        per=[pair(bb[i],aa[i],f"B-A_rep{i+1}") for i in range(3)];pairs.extend(per)
        ab_table.append(dict(A_sr=[a[0]["sr"] for a in aa],B_sr=[a[0]["sr"] for a in bb],
                             A_mean=aavg[0]["sr"],B_mean=bavg[0]["sr"],A_IR=aavg[0]["owner_IR"],B_IR=bavg[0]["owner_IR"],
                             pooled_plus=sum(x["plus"] for x in per),pooled_minus=sum(x["minus"] for x in per),**p))
    pure={}
    for model in ("pi05","groot"):
        for suite in ("l10","sp"):
            run="r04_cost" if model=="pi05" else "r05_q2"
            name=f"r4f_p_{suite}_inf_k10_L10" if model=="pi05" else f"r5q2_g_{'spatial' if suite=='sp' else suite}_policy_L10"
            pure[model,suite]=load(run,name,f"{model}_{suite}","P10")
            rows.append(pure[model,suite][0])
    # The native pi05 L5 references have two seed runs; GR00T source is located explicitly below if present.
    for suite in ("l10","sp"):
        for seed in (1001,2001):
            obj=load("r04_cost",f"r4f_p_{suite}_inf_s{seed}",f"pi05_{suite}","P5",seed)
            rows.append(obj[0])
    ablation=[]
    for typ,file in (("guard_loo","arms_trigger_loo.json"),("direct_PCA","arms_pca.json")):
        specs=json.loads((HERE.parent/"p2_ablations"/file).read_text())
        for spec in specs:
            match=re.search(r"_([pg])_(l10|sp)_(50|500)$",spec["name"])
            assert match,spec["name"]
            cell=("pi05" if match[1]=="p" else "groot")+f"_{match[2]}_{match[3]}"
            obj=load("r06_abl",spec["name"],cell,typ)
            objects[obj[0]["arm"]]=obj;rows.append(obj[0])
            refs=[ab[cell,"B" if typ=="guard_loo" else "A",r] for r in (1,2,3)]
            avg=mean_ref(refs)
            contrast=(spec["kwargs"].get("disabled_guard",typ))+"-reference"
            p=pair(obj,avg,contrast,True);pairs.append(p)
            per=[pair(obj,r,contrast+f"_rep{i+1}") for i,r in enumerate(refs)];pairs.extend(per)
            aa=mean_ref([ab[cell,"A",r] for r in (1,2,3)])
            vs_a=pair(obj,aa,"ablation-Amean3",True)
            pairs.append(vs_a)
            ablation.append(dict(kind=typ,disabled_guard=spec["kwargs"].get("disabled_guard"),arm=obj[0]["arm"],
                                 sr=obj[0]["sr"],IR=obj[0]["owner_IR"],delta_A=vs_a["delta"],
                                 per_rep_plus=[x["plus"] for x in per],per_rep_minus=[x["minus"] for x in per],
                                 per_rep_p=[x["mcnemar_p"] for x in per],**p))
    frontier=[];skipped=[]
    specs=json.loads((HERE.parent/"ideation_Q2/frontier/adapters/emit_arms_all43.json").read_text())
    for spec in specs:
        name=spec["name"];p=ROOT/"r06_frontier/runs"/name/"summary.json"
        if not p.exists():
            skipped.append(dict(arm=name,reason="no summary at snapshot"));continue
        if json.loads(p.read_text()).get("complete")!=500:
            skipped.append(dict(arm=name,reason="not 500 complete at snapshot"));continue
        match=re.search(r"_(pi05|groot)_(l10|spatial)_(50|500)_",name)
        assert match,name
        cell=f"{match[1]}_{'sp' if match[2]=='spatial' else match[2]}_{match[3]}"
        obj=load("r06_frontier",name,cell,"frontier");rows.append(obj[0]);objects[name]=obj
        ref=pure[match[1],"sp" if match[2]=="spatial" else match[2]]
        pi=pair(obj,ref,"frontier-P10");pairs.append(pi)
        pb=pair(obj,mean_ref([ab[cell,"B",r] for r in (1,2,3)]),"frontier-Bmean3",True);pairs.append(pb)
        frontier.append(dict(arm=name,cell=cell,sr=obj[0]["sr"],IR=obj[0]["owner_IR"],P10_sr=ref[0]["sr"],
                             P10_delta=pi["delta"],P10_lo95=pi["lo95"],P10_hi95=pi["hi95"],P10_plus=pi["plus"],P10_minus=pi["minus"],
                             P10_p=pi["mcnemar_p"],B_delta=pb["delta"],B_lo95=pb["lo95"],B_hi95=pb["hi95"]))
    # State-budget matched comparisons named in the task, retaining task/init pairing.
    explicit=[("r6q2_pi05_l10_50_risk_rho0p35","r6q2_pi05_l10_50_B_dose0p5"),
              ("r6q2_groot_l10_50_risk_rho0p35","r6q2_groot_l10_50_B_dose0p5"),
              ("r6q2_pi05_l10_500_risk_rho0p24","r6q2_pi05_l10_500_B_dose0p25"),
              ("r6q2_groot_spatial_50_B_dose0p25","r6p2_no_progress_g_sp_50")]
    for l,r in explicit:
        if l in objects and r in objects:pairs.append(pair(objects[l],objects[r],"named_frontier_pair"))
    # Test available offline library summaries against the existing whole-policy guard ablations.
    librows=[]
    for cell in sorted({k[0] for k in ab}):
        lib=json.loads((HERE.parent/"p3_profiling/calibration_v2"/(cell+".json")).read_text())
        if not cell.endswith("_50"):continue
        model,suite,scale=cell.split("_")
        no=objects[f"r6p2_no_progress_{'p' if model=='pi05' else 'g'}_{suite}_{scale}"]
        br=mean_ref([ab[cell,"B",r] for r in (1,2,3)])
        for task in range(10):
            rr=[r for r in lib["records"] if r["task_id"]==task]
            librows.append(dict(cell=cell,task_id=task,library_episodes=len(rr),
                                library_noprog_alert=np.mean([r["fired_bits"][3] for r in rr]),
                                library_chunk_rms=np.mean([r["loeo_chunk_rms_mean"] for r in rr]),
                                library_progress_max=np.mean([r["maxima"]["progress"] for r in rr]),
                                benefit_keep_noprog=float(br[1].loc[task].mean()-no[1].loc[task].mean())))
    proxy=pd.DataFrame(librows)
    cellproxy=proxy.groupby("cell")[["library_noprog_alert","library_chunk_rms","library_progress_max","benefit_keep_noprog"]].mean()
    correlations=[]
    for feature in ("library_noprog_alert","library_chunk_rms","library_progress_max"):
        a,b=cellproxy[feature].values,cellproxy.benefit_keep_noprog.values
        rho=float(spearmanr(a,b).statistic)
        permutations=[abs(spearmanr(a,list(p)).statistic) for p in itertools.permutations(b)]
        correlations.append(dict(feature=feature,cell_spearman=rho,exact_cell_permutation_p=float(np.mean(np.array(permutations)>=abs(rho)-1e-12)),
                                 n_cells=4,task_spearman=float(spearmanr(proxy[feature],proxy.benefit_keep_noprog).statistic),
                                 scope="post-hoc descriptive proxy check; four banks, not independent bank validation"))
    pd.DataFrame(rows).to_csv(OUT/"arm_audit.csv",index=False)
    pd.DataFrame(pairs).to_csv(OUT/"paired_contrasts.csv",index=False)
    pd.DataFrame(ab_table).to_csv(OUT/"AB_three_replicates.csv",index=False)
    pd.DataFrame(ablation).to_csv(OUT/"ablations.csv",index=False)
    pd.DataFrame(frontier).to_csv(OUT/"frontier.csv",index=False)
    proxy.to_csv(OUT/"library_proxy_task.csv",index=False)
    cellproxy.to_csv(OUT/"library_proxy_cell.csv")
    save("library_proxy_correlations.json",correlations)
    save("inventory.json",dict(arms_audited=len(rows),episodes=sum(r["n"] for r in rows),frontier_complete=len(frontier),
                                frontier_skipped=skipped,analysis_sha256=sha(__file__),exploratory=True,
                                inference="per-pair exact McNemar; mean3 differences cluster all reused outcomes within task/init",
                                cost="owner coefficients, normalized to nominal five controls; terminal partial controls unavailable"))
    print(json.dumps(dict(arms_audited=len(rows),frontier_complete=len(frontier),pair_comparisons=len(pairs),exploratory=True)))


if __name__=="__main__":main()
