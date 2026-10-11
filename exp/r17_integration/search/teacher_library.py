"""Streaming teacher parent, R10 row-index subsets and selected-library fits."""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

from exp.r17_integration.debug.reader_profiles import CaptureReader
from exp.r17_integration.debug.validate import decision_identity, digest, episode_identity, file_sha
from exp.r17_integration.benchmarks.spec import digest as task_data_digest
from .store import SCHEMA, Store, atomic_json, fsync_directory


def build_teacher_parent(sources, destination, binding, episodes, alignment, *, expected_episodes=500):
    """Two bounded passes; RGB never accumulates in a Python whole-run list."""
    sources = tuple(map(Path, sources))
    if alignment['init_pool'] != 'B' or alignment['status'] != 'PASS' or len(episodes) != expected_episodes:
        pass
    by_coordinate = {(r['task_id'],r['orig_init_state_idx']):r for r in alignment['rows']}
    from exp.r17_integration.debug.attempts import capture_counts, validate_prior_attempts
    attempt_counts, _ = capture_counts(sources, binding=binding)
    selected = {}
    for ep in episodes:
        coordinate = ep['suite_task_id'], ep['orig_init_idx']
        witness = by_coordinate[coordinate]
        canonical_name = alignment['suite']+'/'+witness['task_file']
        if (ep.get('split') != 'library' or ep.get('init_pool') != 'B' or not ep.get('closed_ok') or
            type(ep.get('success')) is not bool or ep.get('error') is not None or
            ep['library_episode_id'] != witness['episode_id'] or
            ep['init_state_sha256'] != witness['init_array']['state_sha256'] or ep['task_name'] != canonical_name or
            ep['task_data_hash'] != task_data_digest(dict(witness['init_array'],task_name=canonical_name))):
            raise ValueError('teacher cohort/init/episode alignment mismatch')
        validate_prior_attempts(ep, attempt_counts)
        key = episode_identity(ep)
        if key in selected:
            raise ValueError('duplicate selected teacher episode')
        selected[key] = ep
    ids = [ep['library_episode_id'] for ep in selected.values()]
    if len(set(ids)) != len(ids) or expected_episodes==500 and set(ids) != set(range(500)):
        raise ValueError('teacher must select each B coordinate/episode id exactly once')
    def iterate():
        for source in sources:
            reader = CaptureReader(source, expected_binding=binding, max_cached_blocks=2)
            for row, values, keys in reader.decisions():
                if episode_identity(row) in selected:
                    yield row, values, keys
    records, source_hashes = {}, {str(p/'r17_capture.json'):file_sha(p/'r17_capture.json') for p in sources}
    for row,values,keys in iterate():
        identity = decision_identity(row)
        ep = selected[identity[:-1]]
        k = binding['sampler']['num_steps']
        if (row.get('source') != 'policy' or row.get('actual_nfe') != k or row.get('stage_calls') != [1,1,k] or
            row.get('sampler_identity') != digest(binding['sampler']) or row.get('init_pool') != 'B' or
            row.get('library_episode_id') != ep['library_episode_id'] or
            row.get('split') != 'library' or row.get('suite_task_id') != ep['suite_task_id'] or
            row.get('init_state_sha256') != ep['init_state_sha256'] or
            row.get('task_data_hash') != ep['task_data_hash'] or row.get('executed_count') is None):
            raise ValueError('capture is not an acknowledged full-native B teacher decision')
        if identity in records:
            raise ValueError('duplicate teacher decision across replicas')
        records[identity] = dict(row, success=ep['success'], closed_ok=True, split='library',
            episode_id=ep['library_episode_id'], library_episode_id=ep['library_episode_id'])
    for key,ep in selected.items():
        seqs = sorted(identity[-1] for identity in records if identity[:-1] == key)
        if not seqs or seqs != list(range(ep['n_decisions'])):
            raise ValueError('teacher episode is incomplete/noncontiguous')
        if not records[key+(seqs[-1],)].get('terminal'):
            raise ValueError('teacher final decision acknowledgment is missing')
    ordered = sorted(records, key=lambda identity:(records[identity]['library_episode_id'],identity[-1]))
    rows = [records[identity] for identity in ordered]
    index = {identity:i for i,identity in enumerate(ordered)}
    for row in rows:
        seq, ep = row['decision_seq'], selected[episode_identity(row)]
        row.update(row_id=digest(decision_identity(row)), progress=seq/max(1,ep['n_decisions']-1),
            prev_id=digest(episode_identity(row)+(seq-1,)) if seq else None,
            next_id=digest(episode_identity(row)+(seq+1,)) if seq+1<ep['n_decisions'] else None)
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    atomic_json(destination/'PARTIAL.json', dict(schema=SCHEMA,status='partial',binding=binding))
    buffers = {}
    layout = binding['model']['layout']
    fields = binding['model']['feature_spec']['fields']
    for row,values,keys in iterate():
        payload = {name:values[name] for name in ('state','normalized_actions','image_0','image_1')}
        payload.update({'key_'+name:keys[name] for name in fields})
        width = layout.get('model_action_width') or layout['valid_action_dim']
        if payload['normalized_actions'].shape != (layout['horizon'],width):
            raise ValueError('teacher normalized action geometry mismatch')
        if payload['state'].ndim != 1 or max(layout['valid_state_indices']) >= payload['state'].size:
            raise ValueError('teacher state geometry mismatch')
        for name,size in fields.items():
            if keys[name].shape != (size,) or keys[name].dtype != np.dtype(binding['model']['feature_spec']['key_dtype']):
                raise ValueError('teacher retrieval field geometry mismatch')
        for name,value in payload.items():
            if name.startswith('image_'):
                if value.dtype != np.uint8 or value.ndim != 3 or value.shape[-1] != 3:
                    raise ValueError('teacher requires both received uint8 RGB images')
            elif not np.isfinite(value).all():
                raise ValueError('teacher has nonfinite fit arrays')
            if name not in buffers:
                buffers[name] = np.lib.format.open_memmap(destination/(name+'.npy'), mode='w+',
                    dtype=value.dtype, shape=(len(rows),*value.shape))
            if buffers[name].dtype != value.dtype or buffers[name].shape[1:] != value.shape:
                raise ValueError('teacher row dtype/shape drift')
            buffers[name][index[decision_identity(row)]] = value
    entries = {}
    for name,data in buffers.items():
        data.flush()
        path = destination/(name+'.npy')
        with path.open('rb') as stream:
            os.fsync(stream.fileno())
        entries[name] = dict(file=path.name, sha256=file_sha(path), dtype=data.dtype.str,
                            shape=list(data.shape), mode='dense')
    metadata = dict(recipe='teacher_library', episodes=episodes, key_fields=list(fields),
        init_pool='B', alignment_digest=alignment['digest'], images='received_wire_uint8; both real cameras',
        failed_episodes_kept=True, expected_episodes=expected_episodes, full_native_sampler=True,
        fit_reads='selected B-only arrays; no A/query/outcome features')
    if any(ep.get('prior_attempts') for ep in episodes):
        prior = [row for ep in episodes for row in ep.get('prior_attempts', [])]
        metadata['pre_decision_infrastructure_attempts'] = [row for row in prior if row.get('disposition') not in ('discarded_node_failure','discarded_missing_worker_result')]
        discarded = [row for row in prior if row.get('disposition') == 'discarded_node_failure']
        if discarded:
            metadata['quarantined_node_failure_attempts'] = discarded
        missing = [row for row in prior if row.get('disposition')=='discarded_missing_worker_result']
        if missing:metadata['quarantined_missing_worker_result_attempts']=missing
    manifest = dict(schema=SCHEMA,status='READY',binding=binding,rows=rows,sources=source_hashes,
                    metadata=metadata,arrays=entries)
    manifest['content_digest'] = digest(manifest)
    atomic_json(destination/'READY.json', manifest)
    fsync_directory(destination)
    return Store(destination)


class IndexedArray:
    """Lazy parent selection; no persistent subset array or eager RGB copy."""
    def __init__(self, parent, indices):
        self.parent, self.indices = parent, indices
        self.shape, self.dtype = (len(indices),*parent.shape[1:]), parent.dtype
    def __len__(self):
        return len(self.indices)
    def __getitem__(self, item):
        return self.parent[self.indices[item]]
    def __array__(self, dtype=None, copy=None):
        value = np.asarray(self.parent[self.indices], dtype=dtype)
        value.flags.writeable = False
        return value


class RowIndexStore:
    def __init__(self, directory, *, parent=None):
        self.directory = Path(directory)
        self.manifest = json.loads((self.directory/'VIEW.json').read_text())
        m = self.manifest
        if m['schema'] != 'r17.row_index_view.v1' or digest({k:v for k,v in m.items() if k!='content_digest'}) != m['content_digest']:
            pass
        self.parent = parent or Store(self.directory / m['parent_path'])
        if self.parent.manifest['content_digest'] != m['parent_digest'] or self.parent.directory.resolve() != (self.directory/m['parent_path']).resolve():
            pass
        path = self.directory/'row_indices.npy'
        if file_sha(path) != m['row_indices_sha256']:
            pass
        self.indices = np.load(path, allow_pickle=False, mmap_mode='r')
        if self.indices.dtype != np.int64 or self.indices.ndim != 1 or not len(self.indices) or np.any(np.diff(self.indices)<=0):
            raise ValueError('subset indices must be sorted unique int64 rows')
        selected = set(m['episode_ids'])
        parent_rows = self.parent.rows
        expected = np.array([i for i,r in enumerate(parent_rows) if r['library_episode_id'] in selected],np.int64)
        if not np.array_equal(self.indices,expected):
            raise ValueError('subset must contain complete selected episodes')
        self._rows = tuple(parent_rows[i] for i in self.indices)
    @property
    def rows(self):
        return self._rows
    @property
    def fields(self):
        return self.parent.fields
    def array(self,name):
        return IndexedArray(self.parent.array(name),self.indices)


def nested_views(parent, alignment, *, sizes=(50,100,200,300,400,500)):
    result, previous = {}, set()
    for size in sizes:
        selected = set(sum(alignment['selections'][str(size)].values(),[]))
        if len(selected) != size or not previous <= selected:
            raise ValueError('R10 subsets must be nested complete episode quotas')
        available = {r['library_episode_id'] for r in parent.rows}
        if not selected <= available:
            raise ValueError('subset episodes absent from500 parent')
        indices = np.array([i for i,r in enumerate(parent.rows) if r['library_episode_id'] in selected], np.int64)
        directory = parent.directory/'subsets'/('size_'+str(size))
        if directory.exists():
            view = RowIndexStore(directory,parent=parent)
            if set(view.manifest['episode_ids']) != selected or view.manifest['metadata']['alignment_digest'] != alignment['digest']:
                raise ValueError('existing subset differs from audited R10 selection')
            result[size] = view
            previous = selected
            continue
        directory.mkdir(parents=True,exist_ok=False)
        path = directory/'row_indices.npy'
        with path.open('xb') as stream:
            np.save(stream,indices,allow_pickle=False); stream.flush(); os.fsync(stream.fileno())
        manifest = dict(schema='r17.row_index_view.v1',parent_path='../..',parent_digest=parent.manifest['content_digest'],
            binding=parent.manifest['binding'],rows=len(indices),episode_ids=sorted(selected),size=size,
            row_indices_sha256=file_sha(path),metadata=dict(init_pool='B',split='library',
                subset_seed=alignment['subset_seed'],alignment_digest=alignment['digest'],selected_fit_only=True))
        manifest['content_digest'] = digest(manifest)
        atomic_json(directory/'VIEW.json',manifest)
        result[size] = RowIndexStore(directory,parent=parent)
        previous = selected
    return result


def fit_nested_libraries(parent, alignment, *, task_names, sizes=(50,100,200,300,400,500), **settings):
    from .fit import fit_recipe, save_fitted_recipe
    views = nested_views(parent,alignment,sizes=sizes)
    products = {}
    layout = parent.manifest['binding']['model']['layout']
    commit = max(layout['replan_steps'], min(10,layout['horizon']))
    settings = {**dict(commit_horizon=commit,corrector=True,guard=True,early=True,k=16,kref=8),**settings}
    for size,view in views.items():
        fit_path = parent.directory/'fits'/('size_'+str(size))
        if (fit_path/'READY.json').exists():
            from .fit import load_fitted_recipe
            load_fitted_recipe(fit_path,expected_binding=parent.manifest['binding'])
            sealed = json.loads((fit_path/'READY.json').read_text())
            prior = sealed['receipt']
            if (prior['store_digest'] != view.manifest['content_digest'] or
                prior['config'] != dict(task_names=list(task_names),**{k:v for k,v in settings.items() if k!='corrector'}) or
                prior['corrector'] != settings['corrector']):
                raise ValueError('existing fit differs from selected library/settings')
            products[str(size)] = dict(fit_digest=sealed['digest'],receipt_digest=prior['digest'],
                directory=str(fit_path),episodes=size,rows=len(view.indices))
            continue
        if fit_path.exists():
            raise ValueError('partial fit retained; move it aside before CPU --finish retry')
        recipe,receipt = fit_recipe(view,task_names=task_names,compact_loeo=True,**settings)
        receipt.update(library_size=size,selected_parent_rows=len(view.indices),
            selection_digest=view.manifest['content_digest'], parent_digest=parent.manifest['content_digest'],
            serving_loeo_required_on_B=size==500,
            serving_loeo='B query task/local-init must exclude library_episode_id; A queries have no B own episode',
            shared_products=['PCA','Mahalanobis','early_metric','LOEO_corrector','distance_calibration',
                             'continuity_scale','corrector_caps','no_progress_guard','periodic_knob'],
            guard_defaults=dict(n=recipe.guard.n,eps=recipe.guard.eps))
        from exp.r17_integration.common.layers import recipe as recipe_module
        receipt['caps_knobs'] = dict(shared_recipe_sha256=file_sha(recipe_module.__file__),
            corrector='unchanged P1 distance ramp, motion-only0.5 cap, fitted sigma/distance_scale',
            guard_threshold_source='unchanged P1 NoProgressGuard defaults; not outcome-trained',
            periodic_gap=recipe.periodic_gap,
            owner_IR_status='requires selected-library schedule calibration and measured native cost profile')
        receipt['digest'] = digest({k:v for k,v in receipt.items() if k!='digest'})
        manifest = save_fitted_recipe(recipe,receipt,fit_path)
        products[str(size)] = dict(fit_digest=manifest['digest'],receipt_digest=receipt['digest'],
            directory=str(parent.directory/'fits'/('size_'+str(size))),episodes=size,rows=len(view.indices))
    atomic_json(parent.directory/'FIT_PRODUCTS.json',products)
    return products


def calibrate_teacher_knobs(parent, products, *, task_names, cost_profile=None, miss_sampler=None):
    """Use P1's selected-library LOEO schedule and exact R11 cap solver.

    Unmeasured new-model costs produce explicit pending receipts. They never
    inherit π/GR coefficients, approximate wall latency, or pretend MISS-rate
    labels are owner IR. Once supplied, one profile is bound to one native MISS
    grid; teacher sampler/fit products remain unchanged.
    """
    from .fit import load_fitted_recipe
    from .analyze import _validate_ir_costs
    from exp.r17_integration.common.recipe import retrieval_model_identity
    from exp.r17_integration.common.ir import calibrate_periodic,library_schedule
    from exp.r17_integration.common.contracts import SamplerSpec
    from types import SimpleNamespace
    binding = parent.manifest['binding']
    if cost_profile is not None:
        _validate_ir_costs(cost_profile)
        sampler = SamplerSpec(**(miss_sampler or binding['sampler']))
        if (set(cost_profile) != {'schema','vision','miss','retrieval_model_identity','sampler_identity','benchmark_protocol','source_identity'} or
            cost_profile['retrieval_model_identity'] != retrieval_model_identity(binding['model']) or
            cost_profile['benchmark_protocol'] != binding['benchmark_protocol'] or
            cost_profile['sampler_identity'] != sampler.identity or cost_profile['miss']<=0 or
            cost_profile['vision']+cost_profile['miss']>1+1e-9):
            raise ValueError('knob costs differ from teacher model/protocol or explicit MISS grid')
    result = {}
    for size,product in products.items():
        view = RowIndexStore(parent.directory/'subsets'/('size_'+size),parent=parent)
        document = dict(schema='r17.teacher_knobs.v1',size=int(size),store_digest=view.manifest['content_digest'],
            fit_digest=product['fit_digest'],status='pending_cost_qualification',
            reason='new-model GPU capture-off cost profile required for owner IR25/32/40/45',calibrations={})
        if cost_profile is not None:
            recipe = load_fitted_recipe(product['directory'],expected_binding=binding)
            config = SimpleNamespace(adapter_binding={'benchmark_protocol':binding['benchmark_protocol']},
                recipe=SimpleNamespace(ordered_tasks=tuple(task_names),guard=recipe.use_guard))
            replay = library_schedule(recipe,view,config)
            for level in (.25,.32,.40,.45):
                try:
                    calibration = calibrate_periodic(level,replay,cost_profile)
                    calibration.update(store_digest=view.manifest['content_digest'],fit_digest=product['fit_digest'])
                    calibration['digest'] = digest(calibration)
                    document['calibrations'][str(level)] = calibration
                except ValueError as exc:
                    if 'feasible interval' not in str(exc): raise
                    document['calibrations'][str(level)] = dict(status='infeasible',reason=str(exc))
            document.update(status='calibrated',reason='P1 library-only LOEO schedules; R11 cap solver',
                miss_sampler=miss_sampler or binding['sampler'],cost_profile=cost_profile)
        document['digest'] = digest(document)
        path = parent.directory/'fits'/('size_'+size)/('KNOBS_'+document['digest'][:16]+'.json')
        if path.exists():
            if json.loads(path.read_text()) != document: raise ValueError('knob receipt conflict')
        else:
            atomic_json(path,document)
        result[size] = dict(status=document['status'],receipt=str(path),digest=document['digest'])
    atomic_json(parent.directory/'KNOB_PRODUCTS.json',result)
    return result
