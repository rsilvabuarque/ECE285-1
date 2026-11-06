"""Decode conditional atomic displacements without performing minimization.

Input is one occupancy row per attempt from generate-mask (or an NPZ containing
occupancy/masks). The supplied parent cell stays fixed. Conditions are attained
training-endpoint tuples stored in the checkpoint, never held-out properties.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
from ase import Atoms
from ase.data import atomic_masses

from .geometry_model import DisplacementModel
from .generate_mask import canonical_masks
from .runtime import device_name, digest, trusted_checkpoint
from ..runtime import require_compute_allocation


def remove_translation(displacement, presence, numbers):
    """Fix the mass-weighted translation gauge over present atoms, in Angstrom."""
    mass = atomic_masses[np.asarray(numbers)][None, :] * presence
    translation = (displacement*mass[:, :, None]).sum(1) / mass.sum(1)[:, None]
    result = displacement-translation[:, None, :]
    result[~presence] = 0
    return result



def load_occupancy(path,parent_sha256=None):
    """Load binary inputs without truncation and preserve their atom-site orientation."""
    with np.load(path,allow_pickle=False) as z:
        if 'parent_sha256' in z and parent_sha256 is not None and str(z['parent_sha256'].item()) != parent_sha256:
            raise ValueError('Input occupancy parent hash differs from supplied parent')
        if 'occupancy' in z:
            values=z['occupancy']; vacancy_input=False
        elif 'canonical_occupancy' in z:
            values=z['canonical_occupancy']; vacancy_input=False
        elif 'masks' in z:
            values=z['masks']; vacancy_input=True
        else:
            raise ValueError('Input needs occupancy, canonical_occupancy, or vacancy masks')
        if values.ndim != 2 or not len(values) or not np.isin(values,[0,1]).all():
            raise ValueError('Input occupancies/masks must be binary before integer conversion')
        occupancy=(1-values if vacancy_input else values).astype(np.uint8)
        source_ids=z['structure_ids'].astype(str) if 'structure_ids' in z else np.asarray([f'input_{i}' for i in range(len(occupancy))])
    if source_ids.shape != (len(occupancy),):
        raise ValueError('structure_ids length differs from occupancy rows')
    return occupancy,source_ids


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--input', type=Path, required=True, help='Occupancy NPZ, ideally canonical orientation')
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--draws', type=int, default=3)
    p.add_argument('--seed', type=int, default=20261402)
    p.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    p.add_argument('--threads', type=int, default=4)
    a=p.parse_args(argv)
    require_compute_allocation()
    a.device=device_name(a.device)
    if a.draws < 1:
        raise ValueError('--draws must be positive')
    if a.output.suffix != '.npz':
        raise ValueError('--output must end in .npz')
    for output in [a.output,a.output.with_suffix('.json')]:
        if output.exists():
            raise FileExistsError(output)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    with np.load(a.parent,allow_pickle=False) as z:
        parent={k:z[k] for k in ['numbers','positions_A','cell_A']}
        indices=z['target_indices'] if 'target_indices' in z else z['li_indices']
        permutations=z['permutations']
    occupancy,source_ids=load_occupancy(a.input,digest(a.parent))
    if occupancy.ndim != 2 or occupancy.shape[1] != len(indices) or len(occupancy)==0 or not np.isin(occupancy,[0,1]).all():
        raise ValueError('Occupancy must be binary with one column per parent target site')
    keys,_=canonical_masks(occupancy,permutations)
    # The training trajectories retain each observed mask's parent orientation.
    # Keys identify symmetry orbits; do not use them to reorient coordinates.
    ck=trusted_checkpoint(a.checkpoint,a.device)
    cfg=ck.get('configuration') or json.loads(a.checkpoint.with_name('configuration.json').read_text())
    kind=cfg['kind']
    latent_dim=cfg.get('latent_dim',16)
    coordinates=cfg.get('coordinates',int(ck['state_dict']['decoder.4.weight'].shape[0]))
    context_dim=cfg.get('context_dim',int(ck['state_dict']['decoder.0.weight'].shape[1])-(latent_dim if kind=='cvae' else 0))
    B,N=len(occupancy),len(parent['numbers'])
    if coordinates != 3*N or context_dim != occupancy.shape[1]+17:
        raise ValueError('This displacement checkpoint does not match the supplied parent dimensions')
    # The historical prototype was fitted against this exact bundled parent.
    # New fits record their supplied parent hash rather than inheriting it.
    expected_parent=cfg.get('parent_sha256','ae8f8bc8d7e767165b0172b396e3c723b15e09f4b79429ac01b7c78fd4741e95')
    if expected_parent != digest(a.parent):
        raise ValueError('Parent hash differs from displacement checkpoint parent; a new material/cell requires a new fitted model')
    n_target=occupancy.sum(1)
    targets={str(k):v for k,v in cfg['target_conditions'].items()}
    missing=set(map(str,n_target.tolist()))-set(targets)
    if missing:
        raise ValueError(f'No attained training-endpoint condition for counts {sorted(missing)}')
    props=np.asarray([targets[str(int(n))]['standardized'] for n in n_target])
    context=np.concatenate([occupancy,(n_target/occupancy.shape[1])[:,None],
                              np.tile([0,0,1],(B,1)),np.tile(np.eye(3).reshape(1,9),(B,1)),props],1).astype(np.float32)
    presence=np.ones((B,N),bool)
    presence[:,indices]=occupancy.astype(bool)
    model=DisplacementModel(context_dim=context_dim,kind=kind,latent_dim=latent_dim,
                            coordinates=coordinates).to(a.device)
    model.load_state_dict(ck['state_dict'],strict=True);model.eval()
    context_t=torch.tensor(context,device=a.device)
    rng=torch.Generator(device=a.device).manual_seed(a.seed)
    records=[]
    for draw in range(a.draws if kind=='cvae' else 1):
        z=torch.randn((B,latent_dim),generator=rng,device=a.device) if kind=='cvae' else None
        with torch.no_grad():
            u=model.decode(z,context_t).cpu().numpy().reshape(B,N,3)*cfg['displacement_scale_A']
        u=remove_translation(u,presence,parent['numbers'])
        positions=parent['positions_A'][None,:,:]+u
        for i in range(B):
            atoms=Atoms(numbers=parent['numbers'][presence[i]],positions=positions[i,presence[i]],
                        cell=parent['cell_A'],pbc=True)
            distance=atoms.get_all_distances(mic=True);np.fill_diagonal(distance,np.inf)
            minimum=float(distance.min()) if len(atoms)>1 else float('inf')
            records.append((i,draw,positions[i],u[i],minimum))
    ii=np.asarray([r[0] for r in records]);draws=np.asarray([r[1] for r in records])
    positions=np.asarray([r[2] for r in records]);displacement=np.asarray([r[3] for r in records])
    minimum=np.asarray([r[4] for r in records])
    ids=np.asarray([f'decoded_{source_ids[i]}__attempt{i}__{kind}{draw}' for i,draw in zip(ii,draws)])
    np.savez_compressed(a.output,positions_parent_A=positions,displacement_A=displacement,
        atom_presence=presence[ii],cell_A=np.tile(parent['cell_A'],(len(ii),1,1)),
        parent_sha256=np.asarray(digest(a.parent)),
        numbers_parent=parent['numbers'],structure_ids=ids,source_mask_ids=source_ids[ii],
        occupancy=occupancy[ii],masks=1-occupancy[ii],n_Li=n_target[ii],n_target=n_target[ii],
        canonical_vacancy_mask=keys[ii],parent_key=keys[ii],draw=draws,
        methods=np.full(len(ii),kind),valid=np.isfinite(positions).all((1,2)),
        minimum_separation_A=minimum)
    receipt=dict(kind=kind,input_masks=B,geometries=len(ii),draws_per_mask=a.draws if kind=='cvae' else 1,
        checkpoint_sha256=digest(a.checkpoint),parent_sha256=digest(a.parent),input_sha256=digest(a.input),
        output_sha256=digest(a.output),seed=a.seed,device=a.device,
        geometry_frame='Fixed supplied parent atom-index frame, preserving input occupancy orientation; mass-weighted translation removed; missing target atoms masked',
        conditions='Stored attained training endpoint property tuple per composition; no held-out target properties',
        minimum_separation_note='Measured only; no chemistry threshold/filter introduced',
        scope='Unminimized coordinate predictions; no energy gradient or zero-force/stress guarantee; draws are not independent training seeds')
    a.output.with_suffix('.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({k:receipt[k] for k in ['kind','input_masks','geometries','scope']},indent=2))


if __name__ == '__main__':
    main()
