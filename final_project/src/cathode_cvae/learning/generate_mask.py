"""Generate fixed-count vacancy proposals and retain every requested attempt.

Energy bins are requested conditions, not measured output energies. An optional
physical temperature annotates an acquisition allocation; it never changes the
latent prior or decoder sampling temperature.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from .models import make_model, validate_context_spec
from .runtime import device_name, digest, trusted_checkpoint
from ..runtime import require_compute_allocation
from ..conditions import (BIN_NAMES, encode_requests, load_schema, schema_identity,
                          validate_schema, condition_dimension)


def canonical_masks(occupancy, permutations):
    """Minimum vacancy strings and inverse-site-permutation witnesses.

    This transforms only discrete occupancies. It does not rotate force vectors.
    """
    x = np.asarray(occupancy)
    perms = np.asarray(permutations)
    n = x.shape[1]
    if perms.ndim != 2 or perms.shape[1] != n or not np.array_equal(
            np.sort(perms, axis=1), np.tile(np.arange(n), (len(perms), 1))):
        raise ValueError('Parent permutations must be valid site permutations')
    inv = np.argsort(perms, axis=1)
    keys, witnesses = [], []
    for row in x:
        candidates = (1 - row)[inv]
        strings = [''.join(map(str, candidate.astype(int))) for candidate in candidates]
        witness = min(range(len(strings)), key=strings.__getitem__)
        keys.append(strings[witness]); witnesses.append(witness)
    return np.asarray(keys, dtype=f'<U{n}'), np.asarray(witnesses, dtype=np.int32)


def load_model(checkpoint, device):
    ck = trusted_checkpoint(checkpoint, device)
    cfg = ck.get('configuration')
    if cfg is None:
        cfg = json.loads(Path(checkpoint).with_name('configuration.json').read_text())
    format_version = cfg.get('model_format_version')
    if format_version is not None:
        if format_version not in {2, 3, 4, 5}:
            raise ValueError('Unsupported model_format_version; use its matching model loader')
        required = {'kind', 'n_sites', 'condition_dim', 'latent_dim', 'hidden_dim',
                    'encoder_kind', 'graph_context', 'graph'}
        if not required.issubset(cfg):
            raise ValueError('Versioned checkpoint is missing explicit architecture dimensions or graph metadata')
        if format_version in {3,4,5} and not {'decoder_context_spec', 'energy_target', 'label_energy_protocol', 'proposal_geometry_protocol', 'beta_schedule'} <= set(cfg):
            raise ValueError('Version3 checkpoint lacks context, schedule, or label/proposal metadata')
        if format_version in {4,5} and not isinstance(cfg.get('remaining_count_features'),bool):
            raise ValueError('Version4 checkpoint lacks explicit remaining-count architecture metadata')
        if format_version<4 and (cfg.get('remaining_count_features') or (cfg.get('decoder_context_spec') or {}).get('kernel_mode','legacy')!='legacy'):
            raise ValueError('New count/shell architecture requires version4 checkpoint metadata')
        context_version = (cfg.get('decoder_context_spec') or {}).get('version', 'causal-context-v1')
        if format_version < 5 and context_version not in {'causal-context-v1','causal-context-v2'}:
            raise ValueError('Angular context requires version5 checkpoint metadata')
        if format_version == 5 and context_version not in {'causal-angular-hist-v1',
                'causal-angular-smooth-v1', 'causal-directional-v1', 'causal-redundant-radial-v1'}:
            raise ValueError('Version5 checkpoint lacks a recognized angular context specification')
        if format_version in {3,4,5} and cfg.get('graph_context'):
            spec = cfg.get('decoder_context_spec')
            if not isinstance(spec, dict) or not {'version', 'association', 'feature_dim'} <= set(spec):
                raise ValueError('Versioned graph context lacks immutable identity metadata')
            if format_version in {3,4} and 'kernels_sha256' not in spec:
                raise ValueError('Versioned graph context lacks its kernel hash')
            if format_version == 5:
                radial = spec.get('radial_spec')
                if not isinstance(radial, dict) or not {'feature_dim','kernels_sha256'} <= set(radial):
                    raise ValueError('Angular context lacks its radial geometry hash')
                if context_version in {'causal-angular-hist-v1','causal-angular-smooth-v1'} and 'angular_buffers_sha256' not in spec:
                    raise ValueError('Angular context lacks its geometry hash')
                if context_version == 'causal-directional-v1' and 'directional_buffers_sha256' not in spec:
                    raise ValueError('Directional context lacks its geometry hash')
                if context_version == 'causal-redundant-radial-v1' and 'extra_features' not in spec:
                    raise ValueError('Redundant control lacks its declared width')
    else:
        # Recognized historical writers all stored these dimensions. We retain
        # only their documented default hidden width/architecture, never infer a
        # new latent width by reshaping tensors or silently padding a model.
        required = {'kind', 'n_sites', 'condition_dim', 'latent_dim'}
        if not required.issubset(cfg):
            raise ValueError('Unrecognized legacy checkpoint lacks explicit kind/site/condition/latent dimensions')
    kind = cfg['kind']
    if kind in {'cvae', 'composition', 'composition_only'}:
        kind = 'dense_bce'
    if kind not in {'dense', 'dense_bce', 'dense_count', 'ar', 'ar_cvae', 'ar_baseline'}:
        raise ValueError('Unrecognized checkpoint model kind')
    if (not isinstance(cfg['latent_dim'], int) or cfg['latent_dim'] < 0 or
            (kind != 'ar_baseline' and cfg['latent_dim'] < 1) or
            (kind == 'ar_baseline' and cfg['latent_dim'] != 0)):
        raise ValueError('Checkpoint latent dimension is inconsistent with its CVAE/AR model kind')
    if 'condition_schema' in cfg:
        validate_schema(cfg['condition_schema'])
        descriptor_dim = cfg.get('descriptor_condition_dim', 0)
        if descriptor_dim not in {0, 2} or bool(descriptor_dim) != bool(cfg.get('descriptor_schema')):
            raise ValueError('Checkpoint descriptor dimension and schema disagree')
        if descriptor_dim:
            from ..descriptor import validate_schema as validate_descriptor_schema
            validate_descriptor_schema(cfg['descriptor_schema'])
        expected = condition_dimension(cfg['condition_mode']) + descriptor_dim
        if cfg.get('condition_dim') != expected:
            raise ValueError('Checkpoint condition dimension disagrees with its versioned encoding')
    model = make_model(kind, n_sites=cfg['n_sites'],
                       condition_dim=cfg['condition_dim'], latent_dim=cfg['latent_dim'],
                       hidden_dim=cfg.get('hidden_dim', 64),
                       encoder_kind=cfg.get('encoder_kind', 'dense'),
                       graph_data=cfg.get('graph'), graph_context=cfg.get('graph_context', False),
                       context_spec=cfg.get('decoder_context_spec'),remaining_count_features=cfg.get('remaining_count_features',False)).to(device)
    model.load_state_dict(ck['state_dict'], strict=True)
    if format_version in {3,4,5} and cfg.get('graph_context'):
        try:
            validate_context_spec(cfg['decoder_context_spec'], model.graph_context.to_spec())
        except ValueError as exc:
            raise ValueError('Saved decoder context buffers disagree with their declared association') from exc
    model.eval()
    return model, cfg


def resolve_request(cfg, n_target, requested_bin=None, binary_condition=None, request_schema=None,
                    descriptor_range=None):
    """Resolve one request to its actual encoder/decoder vector and provenance.

    Legacy binary models can be asked to target middle/high under an external
    three-bin evaluation schema, but both map to their single 'rest' class.
    This limitation is explicit in effective_model_class rather than hidden.
    """
    n_sites = int(cfg.get('n_sites', 48))
    if not 0 <= n_target <= n_sites:
        raise ValueError('Target count is outside checkpoint parent size')
    mode = cfg.get('condition_mode') or ('composition' if cfg.get('condition_dim') == 1 else 'energy')
    schema = cfg.get('condition_schema')
    if request_schema is not None:
        validate_schema(request_schema)
        if schema is not None and schema_identity(schema) != schema_identity(request_schema):
            raise ValueError('Requested partition differs from the checkpoint condition schema')
        schema = request_schema
    requested = requested_bin
    if requested is None:
        requested = ('unconditioned' if mode == 'composition' else
                     ('low' if binary_condition is None or binary_condition == 1 else 'rest'))
    if mode == 'multibin' and requested not in BIN_NAMES:
        raise ValueError('Multibin generation requires low, middle, or high')
    if mode == 'energy' and binary_condition is not None and requested_bin is not None:
        if binary_condition != int(requested == 'low'):
            raise ValueError('--condition conflicts with --requested-bin')
    if schema is not None:
        if schema['n_sites'] != n_sites:
            raise ValueError('Request schema and checkpoint parent dimensions differ')
        if 'condition_schema' not in cfg and mode == 'energy':
            entry = cfg.get('condition_stats', {}).get(str(int(n_target)), {})
            old_q = entry.get('q20_eV_atom', entry.get('low_quantile_eV_atom'))
            if old_q is not None and float(old_q) != float(schema['compositions'][str(int(n_target))]['edges_eV_atom'][0]):
                raise ValueError('Legacy binary checkpoint and request schema use different low thresholds')
        c = encode_requests(schema, np.asarray(n_target), np.asarray(requested), mode=mode)
        identity = schema_identity(schema)
    else:
        if mode == 'multibin' or requested in {'middle', 'high'}:
            raise ValueError('A three-bin request needs its fitted --condition-schema-in for this legacy checkpoint')
        if requested not in {'low', 'rest', 'unconditioned'}:
            raise ValueError('Unknown requested condition')
        c = np.asarray([n_target / n_sites] + ([float(requested == 'low')] if mode == 'energy' else []), dtype=np.float32)
        body = dict(version='legacy-condition-v0', mode=mode, n_sites=n_sites,
                    condition_stats=cfg.get('condition_stats', {}), parent_sha256=cfg.get('parent_sha256'))
        identity = 'legacy-' + hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    effective = requested if mode == 'multibin' else ('unconditioned' if mode == 'composition' else ('low' if requested == 'low' else 'rest'))
    if cfg.get('descriptor_schema') is not None:
        from ..descriptor import encode_requests as encode_descriptor_requests
        if descriptor_range is None:
            raise ValueError('This checkpoint requires an explicit descriptor low/high request')
        extra = encode_descriptor_requests(cfg['descriptor_schema'], np.asarray(n_target),
                                           np.asarray(requested), np.asarray(descriptor_range))
        c = np.concatenate((c, extra), axis=-1)
    elif descriptor_range is not None:
        raise ValueError('A descriptor request needs a descriptor-conditioned checkpoint')
    return np.asarray(c, dtype=np.float32), requested, effective, identity, schema


def attempt_records(count, ids, keys, *, method, fit_seed, sampling_seed, n_target,
                    requested_bin, effective_model_class, schema_id, checkpoint_sha256,
                    parent_sha256, geometry_protocol, temperature_K=None,
                    allocation_policy="fixed-request", allocation_weight=None, error=None,
                    descriptor_range=None, descriptor_schema_id=None):
    """One provenance row per planned attempt, including unsupported requests."""
    if count < 1 or (error is None and (len(ids) != count or len(keys) != count)):
        raise ValueError("Attempt count and generated identities disagree")
    rows = []
    for i in range(count):
        rows.append(dict(method=method, seed=sampling_seed, fit_seed=fit_seed,
            sampling_seed=sampling_seed, n_Li=n_target, attempt=i,
            status="generated" if error is None else "failed_request",
            scoring_structure_id=str(ids[i]) if error is None else "",
            canonical_mask=str(keys[i]) if error is None else "",
            requested_bin=requested_bin, effective_model_class=effective_model_class,
            condition_schema_id=schema_id, temperature_K="" if temperature_K is None else temperature_K,
            allocation_policy=allocation_policy, allocation_weight="" if allocation_weight is None else allocation_weight,
            descriptor_range=descriptor_range or "", descriptor_schema_id=descriptor_schema_id or "",
            checkpoint_sha256=checkpoint_sha256, parent_sha256=parent_sha256,
            geometry_protocol=geometry_protocol, error=error or ""))
    return rows


def arguments(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--n-li', '--n-target', dest='n_li', type=int, required=True)
    p.add_argument('--count', type=int, default=512)
    p.add_argument('--condition', type=int, choices=[0, 1], help='Legacy binary condition; defaults to low (1)')
    p.add_argument('--requested-bin', choices=list(BIN_NAMES), help='Requested measured-energy range')
    p.add_argument('--condition-schema-in', type=Path, help='Frozen request partition for legacy controls; new model partition must match exactly')
    p.add_argument('--descriptor-range', choices=['low', 'high'], help='Required descriptor class for an augmented checkpoint')
    p.add_argument('--temperature-K', '--temperature-k', dest='temperature_K', type=float, help='Physical acquisition temperature in kelvin, not a decoder parameter')
    p.add_argument('--allocation-policy', default='fixed-request')
    p.add_argument('--allocation-weight', type=float, help='Declared fraction in the acquisition schedule')
    p.add_argument('--method-name', help='Explicit experiment/method label in every attempt')
    p.add_argument('--sampling', choices=['auto', 'exact', 'topk'], default='auto')
    p.add_argument('--seed', type=int, default=20261106, help='Sampling seed; fit seed comes from checkpoint')
    p.add_argument('--device', choices=['cpu', 'cuda', 'auto'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    p.add_argument('--output', type=Path, required=True)
    return p.parse_args(argv)


def main(argv=None):
    a = arguments(argv)
    require_compute_allocation()
    a.device = device_name(a.device)
    if a.output.suffix != '.npz' or a.count < 1:
        raise ValueError('A positive count and .npz output path are required')
    unique_output = a.output.with_name(a.output.stem + '.unique.npz')
    for output in [a.output, unique_output, a.output.with_suffix('.attempts.csv'), a.output.with_suffix('.json')]:
        if output.exists():
            raise FileExistsError(output)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    model, cfg = load_model(a.checkpoint, a.device)
    parent_sha = digest(a.parent)
    if cfg.get('parent_sha256') and cfg['parent_sha256'] != parent_sha:
        raise ValueError('Parent hash differs from checkpoint parent; train a new model for another material/cell')
    with np.load(a.parent, allow_pickle=False) as z:
        perms = z['permutations']
    if perms.shape[1] != model.n_sites:
        raise ValueError('Checkpoint and parent site dimensions differ')
    requested = a.requested_bin or ('unconditioned' if model.condition_dim == 1 else ('rest' if a.condition == 0 else 'low'))
    schema_id = ''
    effective = ''
    error = None
    request_schema = None
    try:
        request_schema = load_schema(a.condition_schema_in) if a.condition_schema_in else None
        if request_schema is not None:
            expected_parent = request_schema.get('provenance', {}).get('parent_sha256')
            if expected_parent is not None and expected_parent != parent_sha:
                raise ValueError('Request schema belongs to a different parent structure')
        vector, requested, effective, schema_id, schema = resolve_request(
            cfg, a.n_li, a.requested_bin, a.condition, request_schema, a.descriptor_range)
        if a.temperature_K is not None and (not np.isfinite(a.temperature_K) or a.temperature_K <= 0):
            raise ValueError('Physical acquisition temperature must be positive kelvin')
        if a.allocation_weight is not None and (not np.isfinite(a.allocation_weight) or not 0 <= a.allocation_weight <= 1):
            raise ValueError('Allocation weight must be a finite fraction in [0,1]')
    except (ValueError, KeyError) as failure:
        error = str(failure)
        schema = request_schema or cfg.get('condition_schema')
        schema_id = schema_identity(schema) if schema else 'unsupported-legacy-request'
    sampling = a.sampling
    if sampling == 'auto':
        sampling = 'topk' if str(cfg.get('projection', '')).startswith('top-') else 'exact'
    if error is None:
        try:
            c = torch.tensor(vector, dtype=torch.float32, device=a.device).repeat(a.count, 1)
            rng = torch.Generator(device=a.device).manual_seed(a.seed)
            with torch.no_grad():
                if sampling == 'topk':
                    if not cfg.get('kind', '').startswith(('dense', 'cvae', 'composition')):
                        raise ValueError('topk is only defined for the independent dense decoder')
                    z = torch.randn((a.count, model.latent_dim), generator=rng, device=a.device)
                    logits_np = model.decode_logits(z, c).cpu().numpy()
                    np_rng = np.random.default_rng(a.seed)
                    order = np.lexsort((np_rng.random(logits_np.shape), -logits_np), axis=1)
                    occupancy = np.zeros(logits_np.shape, dtype=np.uint8)
                    np.put_along_axis(occupancy, order[:, :a.n_li], 1, axis=1)
                else:
                    occupancy = model.sample(c, a.n_li, generator=rng).cpu().numpy().astype(np.uint8)
            if not np.all(occupancy.sum(1) == a.n_li):
                raise RuntimeError('Constrained sampler produced an invalid count')
        except (ValueError, RuntimeError, FloatingPointError) as failure:
            error = f'{type(failure).__name__}: {failure}'
    if error is not None:
        occupancy = np.empty((0, model.n_sites), dtype=np.uint8)
    keys, witness = canonical_masks(occupancy, perms)
    canonical_occupancy = (1 - np.asarray([[int(bit) for bit in key] for key in keys], np.uint8)
                           if len(keys) else np.empty_like(occupancy))
    width = (model.n_sites + 3) // 4
    ids = np.asarray([f'mask{model.n_sites}_{int(k, 2):0{width}x}' for k in keys], dtype=f'<U{10+width}')
    checkpoint_sha = digest(a.checkpoint)
    fit_seed = cfg.get('seed', cfg.get('fit_seed', 'unknown'))
    common = dict(requested_bin=np.asarray(requested), effective_model_class=np.asarray(effective),
        descriptor_range=np.asarray(a.descriptor_range or ''),
        descriptor_schema_id=np.asarray(cfg.get('descriptor_schema', {}).get('identity', '') if cfg.get('descriptor_schema') else ''),
        condition_schema_id=np.asarray(schema_id), temperature_K=np.asarray(np.nan if a.temperature_K is None else a.temperature_K),
        allocation_policy=np.asarray(a.allocation_policy), allocation_weight=np.asarray(np.nan if a.allocation_weight is None else a.allocation_weight),
        checkpoint_sha256=np.asarray(checkpoint_sha), fit_seed=np.asarray(str(fit_seed)), sampling_seed=np.asarray(a.seed),
        parent_sha256=np.asarray(parent_sha), geometry_protocol=np.asarray(cfg.get('proposal_geometry_protocol', cfg.get('geometry_protocol', 'pristine-derived deletion; unrelaxed'))),
        label_energy_protocol=np.asarray(cfg.get('label_energy_protocol', cfg.get('geometry_protocol', 'pristine-derived deletion; unrelaxed'))),
        energy_target=np.asarray(cfg.get('energy_target', 'deletion')))
    np.savez_compressed(a.output, occupancy=occupancy, canonical_occupancy=canonical_occupancy,
        canonical_vacancy_mask=keys, masks=1-occupancy, canonicalizing_inverse_permutation_index=witness,
        structure_ids=ids, n_Li=np.full(len(occupancy), a.n_li), n_target=np.full(len(occupancy), a.n_li), seed=a.seed, **common)
    unique_keys, first = np.unique(keys, return_index=True)
    unique_occupancy = canonical_occupancy[first]
    _, unique_witness = canonical_masks(unique_occupancy, perms)
    np.savez_compressed(unique_output, occupancy=unique_occupancy, canonical_occupancy=unique_occupancy,
        canonical_vacancy_mask=unique_keys, masks=1-unique_occupancy,
        canonicalizing_inverse_permutation_index=unique_witness, structure_ids=ids[first],
        n_Li=np.full(len(first), a.n_li), n_target=np.full(len(first), a.n_li),
        orientation=np.asarray('canonical'), seed=a.seed, **common)
    method = a.method_name or cfg.get('kind', 'legacy_cvae')
    rows = attempt_records(a.count, ids, keys, method=method, fit_seed=fit_seed,
        sampling_seed=a.seed, n_target=a.n_li, requested_bin=requested,
        effective_model_class=effective, schema_id=schema_id,
        checkpoint_sha256=checkpoint_sha, parent_sha256=parent_sha,
        geometry_protocol=common['geometry_protocol'].item(), temperature_K=a.temperature_K,
        allocation_policy=a.allocation_policy, allocation_weight=a.allocation_weight, error=error,
        descriptor_range=a.descriptor_range, descriptor_schema_id=common['descriptor_schema_id'].item())
    with a.output.with_suffix('.attempts.csv').open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    receipt = dict(status='GENERATED' if error is None else 'FAILED_REQUEST', attempts=a.count,
        generated_attempts=len(occupancy), failed_requests=0 if error is None else a.count,
        unique_orbits=len(set(keys)), n_Li=a.n_li, requested_bin=requested,
        effective_model_class=effective, condition_schema_id=schema_id,
        descriptor_range=a.descriptor_range, descriptor_schema_id=common['descriptor_schema_id'].item(),
        temperature_K=a.temperature_K if a.temperature_K is None or np.isfinite(a.temperature_K) else None,
        requested_temperature_input=None if a.temperature_K is None else str(a.temperature_K),
        allocation_policy=a.allocation_policy,
        allocation_weight=a.allocation_weight if a.allocation_weight is None or np.isfinite(a.allocation_weight) else None,
        sampling=sampling, fit_seed=fit_seed, sampling_seed=a.seed, seed=a.seed, device=a.device,
        checkpoint_sha256=checkpoint_sha, parent_sha256=parent_sha,
        parent_identity_verified=bool(cfg.get('parent_sha256')),
        parent_identity_note='Hash-verified' if cfg.get('parent_sha256') else 'Historical checkpoint lacks parent hash; dimensions verified, chemical identity not independently certified',
        request_schema_file_sha256=digest(a.condition_schema_in) if a.condition_schema_in else None,
        output_sha256=digest(a.output), unique_output=str(unique_output), unique_output_sha256=digest(unique_output),
        model_configuration=cfg, error=error,
        scope='Prior occupancy proposals only; no geometry minimization or measured energy. All planned attempts retained, including unsupported requests.')
    a.output.with_suffix('.json').write_text(json.dumps(receipt, indent=2, allow_nan=False) + '\n')
    print(json.dumps({k: receipt[k] for k in ['status','attempts','unique_orbits','n_Li','requested_bin','sampling']}, indent=2))
    return 0 if error is None else 1


if __name__ == '__main__':
    raise SystemExit(main())
