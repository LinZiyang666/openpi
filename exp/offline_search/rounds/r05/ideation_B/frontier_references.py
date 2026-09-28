"""Keep the best known controllers beside matched selector controls; no runs launched."""
import pathlib,json
R=pathlib.Path('/home/weiland/trace_runs/os_closed_loop');O=pathlib.Path(__file__).parent
sources=[('r03_full','r3f_p_sp_a05'),('r03_mx','r3mx_p_sp_awm_h70'),('r03_full','r3f_g_sp_a05'),('r02_g50','oscl50_g_l10_cl3'),('r02_g500','oscl500_g_sp_cl3'),('r02_g500','oscl500_g_l10_cl3')]
out=[]
for root,name in sources:
 r=next(r for r in json.loads((R/root/'arms.json').read_text()) if r['arm']==name)
 args=[];it=iter(r.get('plugin_args',[]))
 for x in it:
  if x=='--os-fit-artifact':next(it);continue
  args.append(x)
 row={k:r[k] for k in ['model','suite','cell','method','kwargs'] if k in r}
 row.update(arm='r5b_best_'+name,mode='plugin',source_run=root,source_arm=name,plugin_args=args,full_model=r.get('full_model',False),provenance='borrowed big-library information' if 'a05' in name else 'existing fixed-library controller',note='Frontier reference, separately labeled from the matched one-knob contrast; coordinator must prefit/emit.')
 out.append(row)
(O/'frontier_reference_arms.json').write_text(json.dumps(out,indent=2));print(len(out))
