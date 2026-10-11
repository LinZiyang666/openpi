"""CPU preparation and formal resident YAML campaign for the R17 cache study.

Defaults never start a GPU. --run delegates resource ownership, replicas/MPS,
proxy, worker admission and cleanup to formal_eval. Original B stores are read
only. Cosmos's long labels are aliases, never independent duplicate outcomes.
"""
from __future__ import annotations

from dataclasses import asdict, fields, replace
import argparse
import copy
import hashlib
import json
from pathlib import Path
import shutil
import uuid
from collections import defaultdict

from exp.r17_integration.common.contracts import digest
from exp.r17_integration.search.store import atomic_json

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).with_suffix("")
MODELS = ("dp_libero", "fastwam_libero", "cosmos_libero")
SUITES = ("libero_spatial", "libero_10")
GROUPS = ("c", "cr", "ch", "chr", "cg", "crg", "chg", "chrg")
ALIASES = {"ch": "c", "chr": "cr", "chg": "cg", "chrg": "crg"}
CONTROLS = {'dp_libero':'hs5dp_ctl_142812','fastwam_libero':'hs6fw_res_152931',
            'cosmos_libero':'formal_cosmos_libero_official2000_20261010_130639_a8ca2043'}
PRE_REG = ROOT / "exp/offline_search/rounds/r17/cache/CACHE_STUDY.md"
RULINGS = PRE_REG.with_name("OWNER_RULINGS.md")


def read(path):
    return json.loads(Path(path).read_text())


def file_sha(path):
    from .build_client_bundle import sha256
    return sha256(path)


def model_spec(value):
    from exp.r17_integration.common.contracts import ActionStateLayout, FeatureSpec, SamplerSpec, ModelSpec, Capabilities, Capability
    value = copy.deepcopy(value)
    value["layout"] = ActionStateLayout(**value["layout"])
    value["feature_spec"] = FeatureSpec(**value["feature_spec"])
    value["default_sampler"] = SamplerSpec(**value["default_sampler"])
    value["capabilities"] = Capabilities({k: Capability(**v) for k,v in value["capabilities"]["features"].items()})
    return ModelSpec(**value)


def profiles(bundle, parent):
    from .formal_eval import benchmark
    from exp.r17_integration.models.cosmos.protocols import benchmark_profile
    source, serving = benchmark(), benchmark(True)
    if bundle == "cosmos_libero":
        source, serving = benchmark_profile(source), benchmark_profile(serving, True)
    if source.identity != parent["binding"]["benchmark_protocol"]:
        raise ValueError("teacher source protocol cannot be reconstructed exactly")
    from exp.r17_integration.common.benchmarks import register_benchmark
    register_benchmark(serving)
    return source, serving


def group_settings(bundle, group, long):
    base, _, level = group.partition('_')
    native = 16 if bundle == "cosmos_libero" else 5
    return dict(corrector="r" in base, gated="g" in base,
                commit=long if "h" in base else native,
                alias=ALIASES.get(base) if bundle == "cosmos_libero" and not level else None)


def config_document(provider, serving, source, parent, selected, *, store_path, fit_path, fit_digest,
                    stall_path, stall_digest, settings, periodic_gap=None, serve_loeo=False, num_steps=None):
    from exp.r17_integration.common.qualification import config_for_probe
    from exp.r17_integration.common.provider import binding_for
    from exp.r17_integration.common.config import RecipeConfig
    from exp.r17_integration.common.recipe import fit_identity
    k = provider.model_spec.default_sampler.num_steps if num_steps is None else num_steps
    config = config_for_probe(provider, k, settings["commit"])
    config.enabled = True
    config.checkpoints = {}
    config.adapter_binding = binding_for(provider, serving.identity,
        num_steps=k, commit_horizon=settings["commit"], allow_partial_commit=bool(settings["commit"] % provider.model_spec.layout.replan_steps))
    config.recipe = RecipeConfig(enabled=True, store_path=str(store_path),
        store_digest=selected["content_digest"], teacher_sampler_identity=digest(parent["binding"]["sampler"]),
        fitted_path=str(fit_path), fitted_digest=fit_digest,
        miss_mode="stall_only" if settings["gated"] else "never", corrector=settings["corrector"],
        guard=False, periodic_gap=periodic_gap, serve_loeo=serve_loeo, target_ir=None, k=16, kref=8, early=True,
        stall_path=str(stall_path) if settings["gated"] else None,
        stall_digest=stall_digest if settings["gated"] else None,
        ordered_tasks=list(serving.ordered_tasks), task_table_identity=serving.task_table_identity,
        library_benchmark=dict(source=source.manifest(), serving=serving.manifest()))
    config.recipe.fit_identity = fit_identity(config)
    return asdict(config)


def library_size(row):
    return row.get("library_size", 500)


def view_path(row):
    return Path(row["library"])/("subsets/size_"+str(library_size(row)))


def size_label(row):
    # Preserve historical size-500 task/bundle IDs exactly.
    return ("n"+str(library_size(row))+"_" if "library_size" in row else "")


def generate(output, *, library_root="/data/r17_library", long_dp=10, long_fast=10, sizes=None):
    """Emit a complete cohort per size; absent prerequisite fits are explicit."""
    import yaml
    from exp.r17_integration.common.provider import AdapterProvider
    sizes = tuple(sizes) if sizes is not None else (500,)
    if not sizes or len(set(sizes)) != len(sizes) or any(type(n)!=int or n not in (50,100,200,300,400,500) for n in sizes):
        raise ValueError("unique supported library sizes required")
    sized = sizes != (500,)
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema="r17.cache_study.v1", models={}, eval_pool="A", inits=50,
        suites=list(SUITES), groups=list(GROUPS), prereg_sha256=file_sha(PRE_REG),
        rulings_sha256=file_sha(RULINGS), ordinary_miss=False, knobs=False,
        controls=CONTROLS,horizon_selection=dict(dp=long_dp, fastwam=long_fast, cosmos=16,
            status="fallback_C10_pending_horizon_NI_freeze"), artifacts_pending=[])
    if sized:manifest["library_sizes"]=list(sizes)
    for bundle in MODELS:
        long = {"dp_libero":long_dp,"fastwam_libero":long_fast,"cosmos_libero":16}[bundle]
        model_rows = []
        for n in sizes:
            for suite in SUITES:
                library = Path(library_root) / (bundle+"_"+suite) / "bpool"
                parent, selected = read(library/"READY.json"), read(library/f"subsets/size_{n}/VIEW.json")
                source, serving = profiles(bundle, parent)
                spec = replace(model_spec(parent["binding"]["model"]), allowed_benchmark_protocols=(serving.identity,))
                # Reviewable teacher-bound template; seal() rebinds current provider
                # and requires its explicit compatibility receipt before dispatch.
                provider = AdapterProvider("teacher_template_"+bundle, spec, lambda: None, lambda k,s=spec:s.default_sampler)
                for group in GROUPS:
                    settings = group_settings(bundle, group, long)
                    original_fit = library/f"fits/size_{n}"
                    old = read(original_fit/"READY.json") if (original_fit/"READY.json").exists() else None
                    c = settings["commit"]
                    fit = (original_fit if old and old["receipt"]["config"]["commit_horizon"] == c else
                           (output/"artifacts"/("n"+str(n)) if sized else output/"artifacts")/(bundle+"_"+suite)/("C"+str(c))/"fit")
                    fit_digest = read(fit/"READY.json")["digest"] if (fit/"READY.json").exists() else "PENDING_C_SPECIFIC_FIT"
                    stall = (output/"artifacts"/("n"+str(n)) if sized else output/"artifacts")/(bundle+"_"+suite)/("C"+str(c))/"stall"
                    stall_digest = read(stall/"READY.json")["digest"] if (stall/"READY.json").exists() else "PENDING_B_Q3_CALIBRATION"
                    dest = output/bundle/("sp" if suite=="libero_spatial" else "l10")
                    dest.mkdir(parents=True, exist_ok=True)
                    path = dest/(("n"+str(n)+"_" if sized else "")+group+".yaml")
                    doc = config_document(provider,serving,source,parent,selected,
                        store_path=library/f"subsets/size_{n}",fit_path=fit,fit_digest=fit_digest,
                        stall_path=stall,stall_digest=stall_digest,settings=settings)
                    path.write_text(yaml.safe_dump(doc, sort_keys=False))
                    row = dict(suite=suite, group=group, file=str(path.relative_to(output)),
                        sha256=file_sha(path), settings=settings, library=str(library),
                        fit=str(fit), fit_digest=fit_digest, stall=str(stall), stall_digest=stall_digest,
                        status="PENDING_MODEL_COMPATIBILITY_AND_CALIBRATION")
                    if sized:row["library_size"]=n
                    model_rows.append(row)
                    if fit_digest.startswith("PENDING") or settings["gated"] and stall_digest.startswith("PENDING"):
                        manifest["artifacts_pending"].append(bundle+"/"+str(n)+"/"+suite+"/"+group)
        manifest["models"][bundle] = model_rows
    manifest["digest"] = digest(manifest)
    atomic_json(output/"STUDY.json",manifest)
    return manifest


def validate_manifest(path):
    value = read(path)
    knobs = value.get('knobs', False)
    if (value.get("schema") != "r17.cache_study.v1" or value.get("digest") !=
            digest({k:v for k,v in value.items() if k!="digest"}) or value.get("eval_pool", "A") not in ("A", "B") or
            value.get("ordinary_miss") is not False or
            value.get("inits")!=50 or value.get("suites")!=list(SUITES) or value.get("groups")!=list(GROUPS)):
        pass
    root = Path(path).parent
    sizes=value.get("library_sizes",[500])
    if (not sizes or len(set(sizes))!=len(sizes) or any(type(n)!=int or n not in (50,100,200,300,400,500) for n in sizes)
            or not value['models'] or not set(value['models']) <= set(MODELS)
            or not knobs and set(value["models"])!=set(MODELS)):
        raise ValueError("invalid cache-study size/model cohort")
    import yaml
    views={}
    for bundle, rows in value["models"].items():
        cells = {(library_size(r),r['suite'],r['group']) for r in rows}
        if knobs:
            allowed = {('crg' if bundle=='cosmos_libero' else 'chrg')+'_'+level
                       for level in ('k50','k40','k30','k20','k10','koff','pure')}
            if not rows or len(cells)!=len(rows) or any(
                    n not in sizes or s not in SUITES or g not in allowed for n,s,g in cells):
                raise ValueError('invalid cache-knob rows')
        elif cells != {(n,s,g) for n in sizes for s in SUITES for g in GROUPS} or len(rows)!=16*len(sizes):
            raise ValueError("cache-study cohort is not the complete eight groups/two suites")
        for row in rows:
            if row["settings"]!=group_settings(bundle,row["group"],value["horizon_selection"][{"dp_libero":"dp","fastwam_libero":"fastwam","cosmos_libero":"cosmos"}[bundle]]):
                pass
            if file_sha(root/row["file"]) != row["sha256"]:
                pass
            document=yaml.safe_load((root/row['file']).read_text())
            recipe=document['recipe'];settings=row['settings']
            view=view_path(row)
            if str(view) not in views:views[str(view)]=read(view/'VIEW.json')
            if (Path(recipe['store_path']).resolve()!=view.resolve() or
                    recipe['store_digest']!=views[str(view)]['content_digest'] or
                    views[str(view)]['size']!=library_size(row) or
                    recipe['fitted_path']!=row['fit'] or recipe['fitted_digest']!=row['fit_digest'] or
                    recipe['corrector']!=settings['corrector'] or
                    recipe['miss_mode']!=('stall_only' if settings['gated'] else 'never') or
                    recipe['guard'] is not False or not knobs and recipe['periodic_gap'] is not None or recipe['target_ir'] is not None or
                    document['adapter_binding']['commit_horizon']!=settings['commit'] or
                    settings['gated'] and (recipe['stall_path']!=row['stall'] or recipe['stall_digest']!=row['stall_digest'])):
                pass
    return value


def fit_inputs(path, *, bundle=None):
    """Create C-specific fits + Q3 LOEO B calibration; originals remain intact."""
    import numpy as np
    from exp.r17_integration.search.fit import fit_recipe, load_fitted_recipe, save_fitted_recipe
    from exp.r17_integration.search.cache_recipe import sealed_store_descriptor, metric_mapping
    from exp.offline_search.rounds.r06.ideation_Q3.stall.stall import StallModel
    from exp.r17_integration.search.teacher_library import IndexedArray
    from exp.r17_integration.search.cache_recipe import CacheStudyRecipe
    manifest = validate_manifest(path)
    for name, rows in manifest["models"].items():
        if bundle and name != bundle: continue
        for n in manifest.get("library_sizes",[500]):
            for suite in SUITES:
                group = [r for r in rows if r["suite"]==suite and library_size(r)==n]
                library = Path(group[0]["library"])
                descriptor = sealed_store_descriptor(view_path(group[0]), read(view_path(group[0])/"VIEW.json")["content_digest"])
                parent = descriptor.parent.manifest
                # Verify only the fit allowlist. Both uint8 cameras stay in the
                # immutable teacher parent; serving does not ship/read those arrays.
                indices = np.load(descriptor.directory/"row_indices.npy", allow_pickle=False)
                arrays = {}
                for key, entry in parent["arrays"].items():
                    if key.startswith("key_") or key in ("state","normalized_actions"):
                        p = library/entry["file"]
                        if file_sha(p) != entry["sha256"]: pass
                        arrays[key] = IndexedArray(np.load(p,mmap_mode="r",allow_pickle=False),indices)
                view = type("SelectedFitView", (), {"manifest":descriptor.manifest,"rows":descriptor.rows,
                    "fields":tuple(arrays),"array":lambda self,key:arrays[key]})()
                _, serving = profiles(name,parent)
                for c in sorted({r["settings"]["commit"] for r in group}):
                    row = next(r for r in group if r["settings"]["commit"]==c)
                    fit_path = Path(row["fit"])
                    print(f"B-only fit/calibration {name} {suite} N={n} C={c}",flush=True)
                    if not (fit_path/"READY.json").exists():
                        fitted, receipt = fit_recipe(view,task_names=serving.ordered_tasks,corrector=True,
                            compact_loeo=True,commit_horizon=c,allow_partial_commit=bool(c%parent["binding"]["model"]["layout"]["replan_steps"]),
                            k=16,kref=8,early=True,guard=False,periodic_gap=None)
                        receipt.update(library_size=n, selection_digest=descriptor.manifest["content_digest"],
                            parent_digest=parent["content_digest"], calibration_scope="B library only; C-specific head")
                        receipt["digest"] = digest({k:v for k,v in receipt.items() if k!="digest"})
                        save_fitted_recipe(fitted,receipt,fit_path)
                    fitted = load_fitted_recipe(fit_path,expected_binding=parent["binding"])
                    stall_path = Path(row["stall"])
                    if not (stall_path/"READY.json").exists():
                        # Reuse Q3's exact estimator/quantiles. No π/GR thresholds
                        # and no A outcomes are transplanted into these artifacts.
                        data = dict(manifest=dict(H=fitted.layout.horizon,exec_steps=fitted.layout.replan_steps),
                            task_id=fitted.task,episode=fitted.episode,step=fitted.step,
                            success=np.array([r["success"] for r in descriptor.rows],bool))
                        model = StallModel.fit(data,metric=metric_mapping(fitted),
                            commit_controls=min(fitted.layout.horizon,2*fitted.layout.replan_steps))
                        model.save(stall_path)
                        receipt = dict(schema="r17.cache_stall.v1",store_digest=descriptor.manifest["content_digest"],
                            fitted_digest=read(fit_path/"READY.json")["digest"],
                            calibration_partition="B_library_successful_LOEO",
                            calibration_source=f"selected teacher B{n}; Q3 successful-episode LOEO empirical residuals",
                            library_size=n,commit_horizon=c,
                            independent_B_validation_recording=False,
                            files={p.name:file_sha(p) for p in stall_path.iterdir() if p.is_file()},
                            unavailable_tasks=[k for k,v in model.tasks.items() if len(v["references"])<4])
                        receipt["digest"]=digest(receipt);atomic_json(stall_path/"READY.json",receipt)
    # Refresh only this NEW study generation; never any original B fit/READY.
    refresh(path)


def refresh(path):
    import yaml
    from exp.r17_integration.common.provider import AdapterProvider, get_provider, register_provider
    manifest = validate_manifest(path)
    pending=[]
    parents={}
    for bundle,rows in manifest["models"].items():
        # Fitting is a separate CPU process from generation. Reconstruct the
        # metadata-only provider before hashing the refreshed binding; never
        # call a resident/model loader here.
        library=rows[0]['library']
        if library not in parents:parents[library]=read(Path(library)/'READY.json')
        parent=parents[library]
        _,serving=profiles(bundle,parent)
        spec=replace(model_spec(parent['binding']['model']),allowed_benchmark_protocols=(serving.identity,))
        provider=AdapterProvider('teacher_template_'+bundle,spec,lambda:None,lambda k,s=spec:s.default_sampler)
        try:existing=get_provider(provider.provider_id)
        except ValueError:register_provider(provider)
        else:
            if existing.model_spec!=provider.model_spec:pass
        for row in rows:
            document = yaml.safe_load((Path(path).parent/row["file"]).read_text())
            if 'library_size' in row:
                for kind in ('fit','stall'):row[kind]=str(Path(row[kind]).resolve())
            document['recipe']['fitted_path']=row['fit']
            document['adapter_binding']['commit_horizon']=row['settings']['commit']
            library=row['library']
            if library not in parents:parents[library]=read(Path(library)/'READY.json')
            layout=parents[library]['binding']['model']['layout']
            document['adapter_binding']['allow_partial_commit']=bool(row['settings']['commit']%layout['replan_steps'])
            if row['settings']['gated']:document['recipe']['stall_path']=row['stall']
            for kind in ("fit","stall"):
                ready = Path(row[kind])/"READY.json"
                if ready.exists():
                    row[kind+"_digest"] = read(ready)["digest"]
                    key = "fitted_digest" if kind=="fit" else "stall_digest"
                    if kind=="fit" or row["settings"]["gated"]:
                        document["recipe"][key] = row[kind+"_digest"]
            if row["fit_digest"].startswith("PENDING") or row["settings"]["gated"] and row["stall_digest"].startswith("PENDING"):
                pending.append(bundle+"/"+row["suite"]+"/"+row["group"])
                row['status']='PENDING_FIT_OR_CALIBRATION'
            else:
                row['status']='ARTIFACTS_READY_MODEL_ADMISSION_PENDING'
            from exp.r17_integration.common.config import config_from_dict
            from exp.r17_integration.common.recipe import fit_identity
            document['recipe']['fit_identity']=fit_identity(config_from_dict(document))
            p=Path(path).parent/row["file"];p.write_text(yaml.safe_dump(document,sort_keys=False));row["sha256"]=file_sha(p)
    manifest["artifacts_pending"]=pending
    manifest['controls']=CONTROLS
    manifest["digest"]=digest({k:v for k,v in manifest.items() if k!="digest"});atomic_json(path,manifest)


def stage_inputs(campaign, directory, launch):
    """SHA-checked /tmp transport of fits + metadata only, never teacher RGB."""
    import yaml
    from .sync_back import snapshot_run, install_snapshot
    manifest = validate_manifest(campaign['cache_study'])
    bundle = campaign['model']+'_libero'
    pool = manifest.get('eval_pool', 'A')
    if pool == 'B':
        import tarfile
        archives = sorted(Path('/data/r17_library').glob(bundle+'_libero_spatial/campaigns/teacher_*/bpool_inputs.tar'))
        if not archives:
            raise FileNotFoundError('no teacher B init archive for '+bundle)
        archive = archives[0]
        with tarfile.open(archive) as tar:
            init_manifest = json.load(tar.extractfile('INIT_FILES.json'))
        if init_manifest.get('init_pool')!='B' or any(
                suite not in init_manifest['alignment_digests'] or
                sum(name.startswith(suite+'/') for name in init_manifest['files'])!=10 for suite in SUITES):
            raise ValueError('teacher B init archive must cover both ten-task suites')
        campaign.update(eval_pool='B', bpool_archive=str(archive),
                        bpool_transfer=dict(archive_sha256=file_sha(archive), manifest=init_manifest))
    selected = [r for r in manifest['models'][bundle]
                if r['suite'] in campaign.get('suites', SUITES)]
    if any(r['fit_digest'].startswith('PENDING') or r['settings']['gated'] and
           r['stall_digest'].startswith('PENDING') for r in selected):
        raise ValueError('cache study missing C-specific fits/B Q3 calibration; run cache_study fit before prepare')
    root = Path(directory)/'study_inputs'
    root.mkdir(exist_ok=False)
    inputs = dict(schema='r17.cache_inputs.v1', source_study_digest=manifest['digest'],
        model=bundle, groups=[], eval_pool=pool, prereg_sha256=manifest['prereg_sha256'],
        rulings_sha256=manifest['rulings_sha256'], horizon_selection=manifest['horizon_selection'])
    copied = {}
    def copy_file(source, relative):
        target = root/relative
        target.parent.mkdir(parents=True,exist_ok=True)
        if relative in copied:
            if copied[relative]!=file_sha(source):raise ValueError('study input collision')
            return
        shutil.copyfile(source,target)
        copied[relative]=file_sha(source)
        if file_sha(target)!=copied[relative]:pass
    for row in selected:
        suite,commit=row['suite'],row['settings']['commit']
        label=size_label(row)
        prefix=suite+'/parent'
        library=Path(row['library'])
        copy_file(library/'READY.json',prefix+'/READY.json')
        for name in ('VIEW.json','row_indices.npy'):
            copy_file(view_path(row)/name,prefix+'/subsets/size_'+str(library_size(row))+'/'+name)
        fit_rel=suite+'/'+label+'C'+str(commit)+'/fit'
        for name in ('READY.json','fit.npz'):
            copy_file(Path(row['fit'])/name,fit_rel+'/'+name)
        stall_rel=suite+'/'+label+'C'+str(commit)+'/stall'
        if row['settings']['gated']:
            ready=read(Path(row['stall'])/'READY.json')
            for name in ('READY.json', *ready['files']):
                copy_file(Path(row['stall'])/name,stall_rel+'/'+name)
        document = yaml.safe_load((Path(campaign['cache_study']).parent/row['file']).read_text())
        recipe = document['recipe']
        options = {key:recipe[key] for key in ('periodic_gap','serve_loeo') if key in recipe}
        # Historical full-step staged inputs remain byte-identical. Reduced
        # YAMLs must carry their K through the resident-provider rebind.
        k = document['adapter_binding']['num_steps']
        if k != read(library/'READY.json')['binding']['model']['default_sampler']['num_steps']:
            options['num_steps'] = k
        inputs['groups'].append(dict(row,view_rel=prefix+'/subsets/size_'+str(library_size(row)),
                                    fit_rel=fit_rel,stall_rel=stall_rel, **options))
    inputs['files']=copied
    inputs['digest']=digest(inputs)
    atomic_json(root/'manifest.json',inputs)
    dest=str(Path(campaign['assets'])/'cache_inputs')
    if campaign['server_node']=='local':
        shutil.copytree(root,dest)
    else:
        archive=Path(directory)/'study_inputs.tar'
        receipt=snapshot_run(root,archive)
        stage='/tmp/r17_formal/'+campaign['run_name']+'/server/study_inputs.tar'
        launch.transfer(['tether','push',str(archive),'h100:'+stage])
        command=('from exp.r17_integration.ops.sync_back import install_snapshot; import json; '
            'print(json.dumps(install_snapshot('+repr(stage)+','+repr(dest)+
            ',expected_sha256='+repr(receipt['archive_sha256'])+')))')
        installed=json.loads(launch.remote('h100',['env','PYTHONPATH='+launch.pythonpath(campaign['runtime']),
            launch.host_python(campaign),'-c',command],seconds=180,replay_safe=True))
        atomic_json(Path(directory)/'STUDY_INPUT_TRANSFER.json',installed)


def stall_runtime_files():
    return ('exp/offline_search/rounds/r06/ideation_Q3/stall/stall.py',
            'exp/offline_search/rounds/r06/ideation_Q3/stall/stall_reference.py',
            'exp/offline_search/rounds/r06/ideation_Q1/method_c/stall_bridge.py')


def append_stall_runtime(archive, root):
    import tarfile
    # These existing math modules are read-only dependencies, not forked R6
    # implementations. Do not archive old results, fitted pickles or harness.
    names=stall_runtime_files()
    with tarfile.open(archive,'a') as tar:
        present=set(tar.getnames())
        for name in names:
            if name not in present:tar.add(Path(root)/name,arcname=name,recursive=False)


def audit_batches(campaign, directory):
    config=campaign.get('fastwam_batching') or campaign.get('dp_batching')
    counts={name:[] for name in ('stage1','stage2','stage3')}
    decisions=0
    for path in sorted(Path(directory).glob('replica_*/serving.jsonl')):
        for line in path.read_text().splitlines():
            row=json.loads(line);batch=row.get('batching',{})
            if batch.get('config')!=config:pass
            stages=batch.get('stages',{})
            expected=({'stage1','stage2','stage3'} if row['source']=='policy' else
                      {'stage1'} if row['source']=='library' else set())
            if set(stages)!=expected or row['source']!='policy' and row['nfe']!=0:
                raise ValueError('cache hit/tail/gate executed wrong native stages')
            for name,entry in stages.items():
                members=entry['members']
                if (entry['stage']!=name or entry['config_identity']!=config['identity'] or
                    not 1<=entry['size']<=config['max_batch'] or entry['size']!=len(members) or
                    not 0<=entry['row']<len(members) or len({m['request_id'] for m in members})!=len(members)):
                    pass
                member=members[entry['row']]['request']
                if member['decision_seq']!=row['decision_seq'] or member['task_name']!=row['task_name']:
                    pass
                counts[name].append(entry['size'])
            decisions+=1
    if not decisions:raise ValueError('cache-study batching evidence absent')
    return dict(status='PASS',decisions=decisions,mean_actual_batch_size={
        k:sum(v)/len(v) if v else 0 for k,v in counts.items()})


def seal(provider, campaign, benchmark):
    """Bind every arm to the actual resident; every prerequisite fails on CPU."""
    import yaml
    from exp.r17_integration.common.config import config_from_dict, build_shared_storage
    from exp.r17_integration.common.serving import make_bundle_validator
    from exp.r17_integration.common.recipe import bundle_identity, recipe_identity,make_recipe_factory
    from exp.r17_integration.search.cache_recipe import sealed_store_descriptor
    # CPU asset sealers may synthesize a provider without the live loader's
    # validator. Reuse model-owned hooks so preparation cannot promise recipes
    # that the same model rejects at resident publication.
    import importlib
    module=importlib.import_module('exp.r17_integration.models.'+campaign['model']+'.provider')
    checked=provider
    if checked.config_validator is None and hasattr(module,'validate_config'):
        checked=replace(checked,config_validator=module.validate_config)
    configure=getattr(module,'configure_library_provider',None)
    if configure is not None:checked=configure(checked)
    if checked is None or checked.provider_id!=provider.provider_id or checked.model_spec!=provider.model_spec:
        pass
    root=Path(campaign['assets'])/'cache_inputs';inputs=read(root/'manifest.json')
    if inputs['digest']!=digest({k:v for k,v in inputs.items() if k!='digest'}) or any(
            file_sha(root/name)!=pin for name,pin in inputs['files'].items()):
        pass
    output=Path(campaign['assets'])/'cache_yamls';output.mkdir(exist_ok=False)
    groups=[]
    for row in inputs['groups']:
        descriptor=sealed_store_descriptor(root/row['view_rel'],read(root/row['view_rel']/'VIEW.json')['content_digest'])
        source,serving=profiles(inputs['model'],descriptor.parent.manifest)
        if serving.identity!=benchmark.identity:pass
        expected_names={task[0] for task in serving.tasks if task[2]==row['suite']}
        if ({r['task_name'] for r in descriptor.rows}!=expected_names or descriptor.manifest['size']!=library_size(row)):
            raise ValueError('study suite requires all ten tasks from its complete selected B view')
        document=config_document(provider,serving,source,descriptor.parent.manifest,descriptor.manifest,
            store_path=root/row['view_rel'],fit_path=root/row['fit_rel'],fit_digest=row['fit_digest'],
            stall_path=root/row['stall_rel'],stall_digest=row['stall_digest'],settings=row['settings'],
            periodic_gap=row.get('periodic_gap'), serve_loeo=row.get('serve_loeo',False), num_steps=row.get('num_steps'))
        config=config_from_dict(document)
        if checked.config_validator is not None:checked.config_validator(config)
        make_bundle_validator(provider,recipe_factory=make_recipe_factory(checked))(config,build_shared_storage(config))
        identity=bundle_identity(config)
        short=size_label(row)+('sp' if row['suite']=='libero_spatial' else 'l10')+'_'+row['group']
        path=output/(short+'.yaml');path.write_text(yaml.safe_dump(document,sort_keys=False))
        groups.append(dict(row,file='cache_yamls/'+path.name,sha256=file_sha(path),
            bundle_id='r17cs_'+short+'_'+identity[:12],bundle_identity=identity,
            recipe_identity=recipe_identity(config)))
    sealed=dict(schema='r17.cache_sealed.v1',inputs_digest=inputs['digest'],model=asdict(provider.model_spec),
                groups=groups,benchmark_protocol=benchmark.identity,eval_pool=inputs.get('eval_pool','A'),debug_capture=False,
                source_study_digest=inputs['source_study_digest'],horizon_selection=inputs['horizon_selection'])
    sealed['digest']=digest(sealed);atomic_json(Path(campaign['assets'])/'CACHE_STUDY.json',sealed)


def selected_groups(campaign):
    seal=read(Path(campaign['local_assets'])/'CACHE_STUDY.json')
    if seal['digest']!=digest({k:v for k,v in seal.items() if k!='digest'}):
        pass
    wanted=campaign.get('suites') or SUITES
    rows=[r for r in seal['groups'] if r['suite'] in wanted]
    if not campaign.get('smoke'):
        rows=[r for r in rows if not r['settings']['alias'] or '_' in r['group']]
    return seal,rows


def publish(control, campaign):
    from exp.r17_integration.conductor.manifest import load_plan_config
    seal,rows=selected_groups(campaign)
    acknowledgments=[]
    for row in rows:
        path=Path(campaign['local_assets'])/row['file']
        ack=control.load_cache_config(str(path),yaml_id=row['bundle_id'],bundle_id=row['bundle_id'])
        if ack.get('bundle_identity')!=row['bundle_identity']:
            pass
        acknowledgments.append(dict(group=row['group'],suite=row['suite'],ack=ack,
                                    bundle_id=row['bundle_id'],bundle_identity=row['bundle_identity']))
    return dict(schema='r17.cache_bundle_acks.v1',seal_digest=seal['digest'],rows=acknowledgments)


def plan(campaign):
    from exp.r17_integration.conductor.manifest import load_benchmark_spec,load_plan_config,episode_tasks,EndpointSpec
    from exp.r17_integration.common.recipe import bundle_identity
    from openpi.conductor.task import ServerEndpoint
    seal,rows=selected_groups(campaign)
    benchmark=load_benchmark_spec(Path(campaign['local_assets'])/'benchmark.json')
    model=read(Path(campaign['local_assets'])/'model_spec.json')
    if seal['model']!=model or seal['benchmark_protocol']!=benchmark.identity:
        pass
    endpoint=ServerEndpoint(campaign['endpoint_host'],campaign.get('endpoint_port',campaign['public_port']))
    pool=seal.get('eval_pool','A')
    planning_spec=benchmark
    if pool=='B':
        from types import SimpleNamespace
        # B is a separate materialised pool. Its DEV indices do not change the
        # official A serving identity or the benchmark returned to the driver.
        planning_spec=SimpleNamespace(**vars(benchmark),identity=benchmark.identity,
            task_table_identity=benchmark.task_table_identity,max_policy_controls=benchmark.max_policy_controls)
        planning_spec.dev_indices=tuple(range(50))
    tasks=[];steps=set()
    for row in rows:
        path=Path(campaign['local_assets'])/row['file']
        if file_sha(path)!=row['sha256']:pass
        config=load_plan_config(path)
        k=config.adapter_binding['num_steps'];steps.add(k)
        expected_nfe=k
        if campaign['model']=='cosmos':
            from exp.r17_integration.models.cosmos.manifest import make_sampler
            expected_nfe=make_sampler(k).config.get('expected_nfe',k)
        if bundle_identity(config)!=row['bundle_identity'] or config.recipe.miss_mode not in ('never','stall_only'):
            pass
        suite_names=[r[0] for r in benchmark.tasks if r[2]==row['suite']]
        names=row.get('tasks',suite_names)
        if not names or not set(names)<=set(suite_names):raise ValueError('row tasks must belong to its suite')
        if campaign.get('smoke'):names=names[:2]
        episodes=episode_tasks(campaign=campaign.get('cache_plan_name',campaign['run_name']),arm=size_label(row)+row['group'],bundle_id=row['bundle_id'],
            endpoint=endpoint,model_identity=model['checkpoint_identity'],norm_identity=model['norm_identity'],
            layout_identity=digest(model['layout']),num_steps=k,
            spec=planning_spec,partition='smoke' if campaign.get('smoke') else 'dev' if pool=='B' else 'test',names=names,
            indices=[0] if campaign.get('smoke') else row.get('indices',list(range(50))),yaml_identity=row['sha256'],
            bundle_generation=row['bundle_id'],bundle_identity=row['bundle_identity'],
            recipe_identity=row['recipe_identity'],commit_horizon=config.adapter_binding['commit_horizon'])
        for task in episodes:
            if expected_nfe != k:task.extra['expected_nfe']=expected_nfe
            task.extra.update(cache_group=row['group'],init_pool=pool,cache_study=True,
                miss_mode=config.recipe.miss_mode,corrector=config.recipe.corrector,
                logical_alias=row['settings']['alias'])
            if 'library_size' in row:task.extra['library_size']=library_size(row)
            if '_' in row['group']:
                task.extra.update(periodic_gap=config.recipe.periodic_gap,serve_loeo=config.recipe.serve_loeo)
        tasks.extend(episodes)
    endpoint_spec=EndpointSpec(endpoint,model['checkpoint_identity'],model['norm_identity'],digest(model['layout']),
        benchmark.identity,allowed_k=tuple(sorted(steps)),capacity=campaign.get('execution_workers',campaign['workers']),
        bundle_identities=tuple(r['bundle_identity'] for r in rows))
    if campaign.get('cache_resume_selection'):
        from .cache_resume import check
        check(campaign)
        missing=set(campaign['cache_resume_selection']['missing_task_uids'])
        tasks=[t for t in tasks if t.task_uid in missing]
        if {t.task_uid for t in tasks}!=missing:pass
    return benchmark,model,tasks,[endpoint_spec]


def run_driver(campaign,directory):
    from exp.r17_integration.conductor.strategy import BenchmarkStrategy
    from exp.r17_integration.common.websocket_policy_server import WebsocketClientPolicy
    from .formal_driver import probe_validator
    from .formal_driver_fast import AsyncCampaignDriver
    import fcntl
    import signal
    import time
    directory=Path(directory);directory.mkdir(parents=True,exist_ok=True)
    seal,groups=selected_groups(campaign)
    acks=read(directory.parent/'STUDY_BUNDLE_ACKS.json')
    expected={(g['bundle_id'],g['bundle_identity']) for g in groups}
    if acks['seal_digest']!=seal['digest'] or {
            (r['bundle_id'],r['bundle_identity']) for r in acks['rows']}!=expected:
        raise ValueError('not every cache-study bundle was acknowledged before dispatch')
    benchmark,model,tasks,endpoints=plan(campaign)
    group_order=list(GROUPS)+[g['group'] for g in groups if g['group'] not in GROUPS]
    class Strategy(BenchmarkStrategy):
        def plan(self,yamls,assignment):
            graph=super().plan(yamls,assignment)
            # Qualification really switches groups; the full study preserves
            # the preregistered suite/group order on one resident fleet.
            ordered=sorted(graph.stages.values(),key=lambda s:(
                s.episodes[0].extra.get('library_size',500),SUITES.index(s.episodes[0].experiment),group_order.index(s.episodes[0].extra['cache_group'])))
            # Knob studies have no preregistered order: leave stages unchained so the next
            # YAML's episodes fill workers while the previous stage drains (no tail bubble).
            if all(g['group'] in GROUPS for g in groups):
                for before,after in zip(ordered,ordered[1:]):graph.add_dependency(before.stage_id,after.stage_id)
            return graph
    strategy=Strategy(tasks,endpoints=endpoints,bundles={g['bundle_id']:Path(campaign['local_assets'])/g['file'] for g in groups})
    lock=(directory/'.eval.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    driver_class=AsyncCampaignDriver
    if campaign.get('driver_mode') == 'legacy':
        from .cache_driver_legacy import LegacyCacheDriver
        driver_class=LegacyCacheDriver
    driver=driver_class(strategy,evidence_directory=directory,live_fleets=campaign.get('live_fleets',False),dispatch_path=directory/'dispatch.jsonl',
        rejection_path=directory/'rejected_worker_probes.json',expected_workers=campaign.get('execution_workers',campaign['workers']),shutdown_seconds=60,
        yaml_weights={t.yaml_id:1 for t in tasks},servers=[endpoints[0].endpoint],journal_path=str(directory/'journal.jsonl'),
        ctl_factory=lambda ep:WebsocketClientPolicy(ep.host,ep.port),bind_host='0.0.0.0',bind_port=campaign['driver_port'],
        worker_probe_validator=probe_validator(benchmark,campaign),episode_timeout_s=campaign['episode_seconds'],
        scheduler_kwargs=dict(eval_concurrency=campaign.get('execution_workers',campaign['workers']),max_episode_retries=campaign['retries']))
    if campaign.get('live_fleets'):
        from .live_fleet_driver import configure
        configure(driver,campaign,directory)
    stopped=False
    def stop(*unused):
        nonlocal stopped
        stopped=True
    for sig in (signal.SIGINT,signal.SIGTERM):signal.signal(sig,stop)
    atomic_json(directory/'manifest.json',dict(campaign,benchmark=benchmark.manifest(),model=model,
        tasks=[asdict(t) for t in tasks],expected_episodes=len(tasks)))
    start=time.monotonic();error=None;commit=None
    try:
        driver.run(stop=lambda:stopped or bool(driver.probe_rejections) or driver.evidence.failure is not None or
                   time.monotonic()-start>campaign['seconds'])
    except Exception as exc:error=repr(exc)
    finally:
        try:commit=driver.close()
        except Exception as exc:error=repr(exc)
        atomic_json(directory/'census.json',driver.worker_census)
        report=dict(status='PASS' if not error and driver.scheduler.all_done() and not driver.probe_rejections and
            driver.worker_shutdown['status']=='PASS' else 'PARTIAL',all_done=driver.scheduler.all_done(),
            elapsed_seconds=time.monotonic()-start,worker_shutdown=driver.worker_shutdown,
            rejected_worker_probes=len(driver.probe_rejections),error=error,group_commit=commit,
            orchestration=campaign.get('orchestration'))
        atomic_json(directory/'DRIVER_RECEIPT.json',report);lock.close()
    raise SystemExit(0 if report['status']=='PASS' else 2)


def aggregate(directory, *, driver_returncode=None):
    from .formal_eval import driver_completion
    directory=Path(directory);campaign=read(directory/'campaign.json')
    completion=driver_completion(directory,driver_returncode) if driver_returncode is not None else None
    planned=read(directory/('FULL_EPISODE_PLAN.json' if campaign.get('cache_resume_selection') else 'EPISODE_PLAN.json'))['tasks']
    composite=None
    if campaign.get('cache_resume_selection'):
        from .cache_resume import evidence
        composite=evidence(campaign,directory)
    def lines(name):
        if composite is not None:return composite[name]
        p=directory/'driver'/name
        return [json.loads(line) for line in p.read_text().splitlines() if line] if p.exists() else []
    journal={r['task_uid']:r for r in lines('journal.jsonl') if r.get('accepted') is True and r.get('status') in ('done','failed')}
    step_rows=lines('per_step.jsonl')
    decisions=[r for r in step_rows if not r.get('_kind') and r.get('accepted') is True and
               r.get('attempt')==journal.get(r.get('task_uid'),{}).get('attempt')]
    receipts=[r for r in step_rows if r.get('_kind')=='execution_ack' and r.get('accepted') is True]
    decision_index,ack_index=defaultdict(list),defaultdict(list)
    for row in decisions:decision_index[row['task_uid']].append(row)
    for row in receipts:ack_index[(row['task_uid'],row['attempt'])].append(row)
    groups={};violations=[]
    multiple_sizes=len({t['extra'].get('library_size',500) for t in planned})>1
    for task in planned:
        ex=task['extra'];key=(('n'+str(ex.get('library_size',500))+'/') if multiple_sizes else '')+ex['cache_group']+'/'+task['experiment']
        group=groups.setdefault(key,dict(planned=0,completed=0,successes=0,errors=0,
            source_counts={},nfe=0,corrector_active=0,corrector_nonzero=0,stall_calls=0,stall_looks=0,
            stall_observations=0,stall_states={},decisions=0))
        group['planned']+=1
        result=journal.get(task['task_uid'])
        if result:
            group['completed']+=1;group['successes']+=int(bool(result.get('success')) and not result.get('error'))
            group['errors']+=int(result.get('error') is not None)
        rows=decision_index[task['task_uid']]
        for row in rows:
            source=row.get('source');calls=row.get('stage_calls');seq=row.get('decision_seq',row.get('step'))
            if (row.get('bundle_identity')!=ex['bundle_identity'] or row.get('recipe_identity')!=ex['recipe_identity'] or
                    row.get('commit_horizon')!=ex['commit_horizon'] or row.get('capture_enabled') is not False):
                violations.append(dict(uid=task['task_uid'],step=seq,reason='binding/commit/capture mismatch'))
            if source not in ('library','library_tail','policy','policy_tail'):
                violations.append(dict(uid=task['task_uid'],step=seq,reason='non-study source'))
            if source=='policy' and (ex['miss_mode']!='stall_only' or not (
                    row.get('stall_call') is True or ex.get('periodic_gap') is not None and row.get('knob_call') is True)):
                violations.append(dict(uid=task['task_uid'],step=seq,reason='ordinary MISS forbidden'))
            if source!='policy' and (row.get('actual_nfe')!=0 or calls not in ([1,0,0],[0,0,0])):
                violations.append(dict(uid=task['task_uid'],step=seq,reason='cache executed action head'))
            expected_nfe=ex.get('expected_nfe',ex['num_steps'])
            if source=='policy' and (calls != [1,1,expected_nfe] or row.get('actual_nfe')!=expected_nfe):
                violations.append(dict(uid=task['task_uid'],step=seq,reason='MISS differs from bound sampler NFE'))
            if source=='library' and (row.get('corrector_enabled') is not ex['corrector'] or
                    row.get('miss_mode')!=ex['miss_mode']):
                violations.append(dict(uid=task['task_uid'],step=seq,reason='cache arm settings missing/mismatched'))
            if not ex['corrector'] and row.get('corrector_applied'):
                violations.append(dict(uid=task['task_uid'],step=seq,reason='corrector active in off arm'))
            group['source_counts'][str(source)]=group['source_counts'].get(str(source),0)+1
            group['nfe']+=row.get('actual_nfe',0);group['decisions']+=1
            group['corrector_active']+=int(bool(row.get('corrector_applied')))
            group['corrector_nonzero']+=int(bool(row.get('corrector_applied')) and (row.get('corrector_delta_norm') or 0)>0)
            group['stall_calls']+=int(bool(row.get('stall_call')))
            group['stall_looks']+=int(bool(row.get('stall_extra_look')))
            if row.get('stall_fresh') and ex['miss_mode']=='stall_only':
                group['stall_observations']+=1
                state=(row.get('stall_status') or {}).get('state','missing')
                group['stall_states'][state]=group['stall_states'].get(state,0)+1
        if result and not result.get('error'):
            acks=ack_index[(task['task_uid'],result['attempt'])]
            seqs=[r['decision_seq'] for r in rows]
            ack_seqs=[r['decision_seq'] for r in acks]
            terminal=[r for r in acks if r.get('terminal') is True]
            if (not rows or sorted(seqs)!=list(range(len(rows))) or sorted(ack_seqs)!=sorted(seqs) or
                    len(terminal)!=1 or terminal[0]['decision_seq']!=max(seqs) or
                    not terminal[0].get('server_terminal_ack') or
                    any(a['executed_count']<0 or a['cursor_end']!=next((d['cursor'] for d in rows
                        if d['decision_seq']==a['decision_seq']),-1)+a['executed_count'] for a in acks)):
                violations.append(dict(uid=task['task_uid'],reason='decision/terminal ACK incomplete'))
    generations=defaultdict(set)
    for row in decisions:
        if row.get('resident_pid') is None or not row.get('resident_generation'):
            violations.append(dict(uid=row['task_uid'],reason='resident lifetime evidence missing'))
        else:
            pid=(row.get('evidence_segment'),row['resident_pid']) if composite is not None else row['resident_pid']
            generations[pid].add(row['resident_generation'])
    segments=defaultdict(set)
    for key in generations:segments[key[0] if composite is not None else 'current'].add(key)
    from .cache_resume import segment_lifetimes
    lifetime_ok=(bool(generations) and all(len(g)==1 for g in generations.values()) and
        (segment_lifetimes(campaign,directory,segments) if composite is not None
         else len(generations)==campaign['replicas']))
    if not lifetime_ok:
        violations.append(dict(reason='resident count/restart evidence mismatch'))
    driver=read(directory/'driver/DRIVER_RECEIPT.json')
    done=sum(g['completed'] for g in groups.values())
    activity=dict(corrector_nonzero_observed=any(g['corrector_nonzero'] for g in groups.values()),
        stall_call_observed=any(g['stall_calls'] for g in groups.values()),
        extra_look_observed=any(g['stall_looks'] for g in groups.values()),
        scope='observed only; absent activity is unexercised, never fabricated')
    summary=dict(schema='r17.cache_summary.v1',status='PASS' if done==len(planned) and not violations and
        not sum(g['errors'] for g in groups.values()) and (driver.get('status')=='PASS' or completion and
        completion.get('verified_completion')) else 'PARTIAL',expected_episodes=len(planned),completed_episodes=done,
        cache_enabled=True,debug_capture=False,groups=groups,violations=violations,activity=activity,
        terminal_errors=sum(g['errors'] for g in groups.values()),full_arm_execution_verified=not violations,
        accepted_worker_ids=sorted({r['worker_id'] for r in lines('attempts.jsonl') if r.get('accepted') and r.get('worker_id')}),
        driver_seconds=driver.get('elapsed_seconds'),resident_generations={str(k):sorted(v) for k,v in generations.items()},
        no_replica_restart_verified=lifetime_ok,
        resume_segments=len(segments),resume=campaign.get('cache_resume_selection'),
        library_sizes=sorted({t['extra'].get('library_size',500) for t in planned}),
        alias_results=ALIASES if campaign['model']=='cosmos' and not campaign.get('smoke') and
            all(t['extra']['cache_group'] in GROUPS for t in planned) else {})
    from .cache_resume import segment_resources
    summary['segment_resources']=segment_resources(campaign,directory)
    atomic_json(directory/'SUMMARY.json',summary)
    return summary


def audit(path, output):
    """Real selected-B fit/metric/gate CPU queries; current model hooks stay strict."""
    import numpy as np
    import yaml
    from exp.r17_integration.common.provider import AdapterProvider,register_provider,get_provider
    from exp.r17_integration.common.config import load_cache_config
    from exp.r17_integration.common.recipe import make_recipe_factory,retrieval_model_identity
    from exp.r17_integration.common.contracts import RequestContext
    from exp.r17_integration.common.query_view import QueryView
    from exp.r17_integration.search.cache_recipe import sealed_store_descriptor
    from collections import Counter
    manifest=validate_manifest(path);report=dict(schema='r17.cache_cpu_audit.v1',cpu_only=True,
        gpu_executed=False,study_digest=manifest['digest'],models={},eval_pool='A',library_pool='B',
        library_sizes=manifest.get('library_sizes',[500]))
    for name,rows in manifest['models'].items():
        entries=[];control=Path('/data/mw_finetune/eval')/CONTROLS[name]/'assets/model_spec.json'
        current=read(control);differences=[]
        nested={}
        for n in sorted(manifest.get("library_sizes",[500])):
            for suite in SUITES:
                subset=[r for r in rows if r['suite']==suite and library_size(r)==n]
                parent=read(Path(subset[0]['library'])/'READY.json')
                selected=read(view_path(subset[0])/'VIEW.json')
                descriptor=sealed_store_descriptor(view_path(subset[0]),selected['content_digest'])
                episodes={r['library_episode_id']:r['task_name'] for r in descriptor.rows}
                counts=Counter(episodes.values())
                full=read(Path(subset[0]['library'])/'subsets/size_500/VIEW.json')
                if (len(counts)!=10 or set(counts.values())!={n//10} or
                        not set(episodes).issubset(set(full['episode_ids'])) or
                        not nested.get(suite,set()).issubset(set(episodes))):
                    raise ValueError('selected B cohorts must be balanced complete nested episode subsets')
                nested[suite]=set(episodes)
                source,serving=profiles(name,parent)
                teacher=parent['binding']['model']
                differences.append(dict(library_size=n,suite=suite,teacher=retrieval_model_identity(teacher),
                    control=retrieval_model_identity(current),fields=[k for k in teacher if teacher[k]!=current.get(k)]))
                spec=replace(model_spec(teacher),allowed_benchmark_protocols=(serving.identity,))
                provider=AdapterProvider('teacher_template_'+name,spec,lambda:None,lambda k,s=spec:s.default_sampler)
                try:get_provider(provider.provider_id)
                except ValueError:register_provider(provider)
                for row in subset:
                    config=load_cache_config(Path(path).parent/row['file'])
                    recipe=make_recipe_factory(provider)(config,None)
                    original=recipe.base
                    i=next(i for i,t in enumerate(original.task) if int(t) in (0,30))
                    index=np.load(view_path(row)/'row_indices.npy',allow_pickle=False)[i]
                    keys={field:np.load(Path(row['library'])/parent['arrays']['key_'+field]['file'],
                        mmap_mode='r',allow_pickle=False)[index].copy() for field in spec.feature_spec.fields if field!='robot_state'}
                    keys['robot_state']=original.state[i].copy()
                    tid=int(original.task[i]);states=[]
                    for seq in range(4):
                        if seq:recipe.on_execution(spec.layout.replan_steps)
                        query=QueryView(RequestContext('cpu_B_readonly_audit',source.identity,source.ordered_tasks[tid],
                            0,seq,task_id=tid,namespace='cpu_B_readonly_audit'),keys,original.state[i].copy(),
                            [True]*seq,[1]*seq,previous_action=original.action[i] if seq else None)
                        result=recipe.query(query)
                        if (result.action.shape!=original.action[i].shape or not np.isfinite(result.action).all() or
                                result.force_miss and not result.extras['stall_call']):
                            raise ValueError('real fitted selected B cache query invalid')
                        states.append(dict(seq=seq,donors=len(result.rows),corrector=result.extras['corrector_applied'],
                            delta_norm=result.extras['corrector_delta_norm'],stall=result.extras['stall_status']['state'],
                            stall_call=result.extras['stall_call']))
                    calibration=read(Path(row['stall'])/'READY.json')
                    entries.append(dict(library_size=n,suite=suite,group=row['group'],status='PASS',C=row['settings']['commit'],
                        fitted_digest=row['fit_digest'],queries=states,
                        stall_unavailable_tasks=calibration.get('unavailable_tasks',[]),
                        stall_availability_scope='unchanged Q3: fewer than four successful selected-B references => inactive'))
        report['models'][name]=dict(real_library_queries=entries,control_comparison=differences,
            deployment_status='needs '+{'dp_libero':'P4','fastwam_libero':'P3','cosmos_libero':'D'}[name],
            qualification_scope='teacher-bound CPU math; current resident admission remains model-owned')
    report['status']='PASS';report['digest']=digest(report);atomic_json(output,report)
    return report


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mode",choices=("generate","fit","audit","launch","smoke"))
    p.add_argument("--output",type=Path,default=HERE/"yaml")
    p.add_argument("--study",type=Path)
    p.add_argument("--model",choices=MODELS)
    p.add_argument("--sizes",type=int,nargs="+",help="library sizes; default historical B500")
    p.add_argument("--long-dp",type=int,default=10);p.add_argument("--long-fast",type=int,default=10)
    args, rest = p.parse_known_args(argv)
    if args.mode=="generate":
        if rest:p.error("unknown generation arguments")
        result=generate(args.output,long_dp=args.long_dp,long_fast=args.long_fast,sizes=args.sizes)
        print(json.dumps(dict(yamls=sum(map(len,result["models"].values())),manifest=str(args.output/"STUDY.json"),pending=len(result["artifacts_pending"]))));return 0
    if not args.study:p.error("--study required")
    if args.mode=="fit":
        if rest:p.error("unknown fit arguments")
        fit_inputs(args.study,bundle=args.model);return 0
    if args.mode=="audit":
        if rest:p.error('unknown audit arguments')
        result=audit(args.study,args.output)
        print(json.dumps(dict(status=result['status'],groups=sum(len(m['real_library_queries']) for m in result['models'].values()),output=str(args.output))));return 0
    if not args.model:p.error("--model required")
    validate_manifest(args.study)
    from .formal_eval import main as formal
    options=["--model",args.model,"--official-50","--cache-study",str(args.study)]
    if args.mode=="smoke":options += ["--smoke","--smoke-episodes","2"]
    return formal(options+rest)


if __name__=="__main__":
    raise SystemExit(main())
