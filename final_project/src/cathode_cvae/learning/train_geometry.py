"""Train a fixed-cell displacement CVAE or a matching deterministic regressor.

Uses already aligned relaxation frames and frozen parent-configuration splits.
Properties are context only: this objective contains no MLIP energy gradient or
force minimizer. Good posterior coordinate reconstruction is not low-force prior
sample validation. The cell is supplied, not generated.
"""
import argparse
import csv
import json
import os
from pathlib import Path
import random
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from .geometry_model import DisplacementModel
from .runtime import device_name, digest, trusted_checkpoint
from ..runtime import require_compute_allocation

COHORTS = ['deletion', 'fixed_intermediate', 'fixed_endpoint']
# This is a provenance binding, not a geometric similarity tolerance.
LEGACY_GEOMETRY_SHA256 = '1c71d8c340378c93d79f7db67c7ac503c2a1229f07c242897c1754d8f7e5d6fb'
LEGACY_PARENT_SHA256 = 'ae8f8bc8d7e767165b0172b396e3c723b15e09f4b79429ac01b7c78fd4741e95'


def validate_parent_binding(source, parent_path, declared_sha256=None):
    """Bind an aligned archive to its exact atom-index parent, not just its size.

    New archives must carry a scalar parent_sha256 or an explicit provenance
    assertion from their producer. Supplying --parent alone cannot establish
    which atom frame generated the stored displacements and forces.
    """
    source_hash=digest(source)
    if source_hash == LEGACY_GEOMETRY_SHA256:
        expected=LEGACY_PARENT_SHA256
        basis='Published project receipt for the immutable bundled geometry archive'
    else:
        with np.load(source,allow_pickle=False) as archive:
            expected=str(archive['parent_sha256'].item()) if 'parent_sha256' in archive else declared_sha256
        basis='Archive parent_sha256 field' if expected != declared_sha256 else 'Explicit producer provenance assertion --parent-sha256'
    if not expected:
        raise ValueError('New aligned geometry archives need a parent_sha256 field or --parent-sha256 producer assertion')
    if expected != digest(parent_path):
        raise ValueError('Aligned geometry archive parent hash does not match --parent; equal dimensions do not establish atom-frame identity')
    if declared_sha256 is not None and declared_sha256 != expected:
        raise ValueError('--parent-sha256 contradicts archive provenance')
    return basis


def prepare_data(path):
    fields = ['localized', 'split', 'cohort', 'structure_ids', 'presence', 'displacement_A',
              'frame_weight', 'energy_eV', 'force_component_rms_eV_A',
              'configurational_pressure_GPa', 'deviatoric_stress_GPa',
              'occupancy', 'deformation', 'cell_A']
    with np.load(path, allow_pickle=False) as z:
        d = {k: z[k] for k in fields}
        d['n_Li']=z['n_target'] if 'n_target' in z else z['n_Li']
    if not np.isin(d['localized'],[0,1]).all() or not np.isin(d['presence'],[0,1]).all() or not np.isin(d['occupancy'],[0,1]).all():
        raise ValueError('Presence, localization, and occupancy fields must be binary')
    keep = d['localized'].astype(bool)
    d = {k: v[keep] for k, v in d.items()}
    split = d['split'].astype(str)
    ownership = {}
    for sid, group in zip(d['structure_ids'].astype(str), split):
        if sid in ownership and ownership[sid] != group:
            raise ValueError('Frames from one vacancy configuration cross protected splits')
        ownership[sid] = group
    train = np.flatnonzero(split == 'train')
    val = np.flatnonzero(split == 'val')
    if not len(train) or not len(val):
        raise ValueError('Both train and validation configuration groups are required')
    cohort = np.asarray([COHORTS.index(str(s)) for s in d['cohort']])
    n_sites = d['occupancy'].shape[1]
    n_atoms = d['presence'].shape[1]
    if not np.array_equal(d['occupancy'].sum(1), d['n_Li']):
        raise ValueError('Occupancy and target count differ')
    presence = np.repeat(d['presence'], 3, axis=1).astype(np.float32)
    u = d['displacement_A'].reshape(-1, n_atoms * 3).astype(np.float32)
    weight = d['frame_weight'].astype(np.float64)
    if not np.isfinite(u).all() or not np.isfinite(weight).all() or np.any(weight <= 0):
        raise ValueError('Nonfinite geometry or invalid frame weights')
    if np.any(u[presence == 0] != 0):
        raise ValueError('Absent-atom displacement slots must be zero')
    scale = float(np.sqrt((weight[train, None] * u[train]**2 * presence[train]).sum()
                          / (weight[train, None] * presence[train]).sum()))
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError('Training displacement scale must be finite and positive')
    raw = np.stack([d['energy_eV'] / d['presence'].sum(1),
                    np.log1p(d['force_component_rms_eV_A']),
                    d['configurational_pressure_GPa'],
                    np.log1p(d['deviatoric_stress_GPa'])], axis=1)
    if not np.isfinite(raw).all():
        raise ValueError('Condition properties must be finite')
    stats, targets = {}, {}
    props = np.zeros_like(raw)
    for n in sorted(set(d['n_Li'].tolist())):
        for cidx, cname in enumerate(COHORTS):
            same = (d['n_Li'] == n) & (cohort == cidx)
            ids = np.flatnonzero(same & (split == 'train'))
            if not len(ids):
                raise ValueError(f'No training reference for target count {n}, cohort {cname}')
            mean = raw[ids].mean(0)
            std = np.maximum(raw[ids].std(0), 1e-8)
            stats[f'{int(n)}|{cname}'] = dict(mean=mean.tolist(), std=std.tolist(),
                                             train_frames=len(ids),
                                             train_parents=len(set(d['structure_ids'][ids])))
            props[same] = (raw[same] - mean) / std
            if cname == 'fixed_endpoint':
                candidates = ids[d['force_component_rms_eV_A'][ids]
                                 <= np.median(d['force_component_rms_eV_A'][ids])]
                chosen = candidates[np.argmin(d['energy_eV'][candidates])]
                targets[str(int(n))] = dict(raw=raw[chosen].tolist(),
                    standardized=props[chosen].tolist(), source_training_parent=str(d['structure_ids'][chosen]),
                    rule='Attained training endpoint with lowest energy among force-RMS-below-median training endpoints')
    context = np.concatenate([d['occupancy'], (d['n_Li']/n_sites)[:, None],
                               np.eye(3)[cohort], d['deformation'].reshape(-1, 9), props], axis=1).astype(np.float32)
    return d, u, presence, context, weight, train, val, scale, stats, targets


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--parent', type=Path, required=True, help='Exact canonical parent used by the aligned frame archive')
    p.add_argument('--parent-sha256', help='Producer assertion for a new aligned archive lacking an embedded parent_sha256')
    p.add_argument('--teacher', default='uma', help='Provenance label for the archive energy/force/stress teacher')
    p.add_argument('--kind', choices=['cvae', 'regressor'], default='cvae')
    p.add_argument('--seed', type=int, default=20261400)
    p.add_argument('--device', choices=['cpu','cuda','auto'], default='auto')
    p.add_argument('--epochs', type=int, default=150)
    p.add_argument('--batch-size', type=int, default=128)
    p.add_argument('--learning-rate', type=float, default=.001)
    p.add_argument('--beta-final', type=float, default=1.)
    p.add_argument('--beta-warmup', type=int, default=20)
    p.add_argument('--validation-beta', type=float, default=1.)
    p.add_argument('--minimum-epochs', type=int, default=50)
    p.add_argument('--checkpoint-after', type=int, default=20,
                   help='Historical geometry fits first considered checkpoints at epoch 20')
    p.add_argument('--patience', type=int, default=25)
    p.add_argument('--max-seconds', type=float, default=900)
    p.add_argument('--max-updates', type=int)
    p.add_argument('--warm-start', type=Path)
    p.add_argument('--threads', type=int, default=4)
    a = p.parse_args(argv)
    require_compute_allocation()
    a.device = device_name(a.device)
    if min(a.epochs, a.batch_size, a.beta_warmup, a.minimum_epochs, a.patience, a.checkpoint_after) < 1:
        raise ValueError('Epoch, batch, patience and warmup settings must be positive')
    if a.beta_final < 0 or a.validation_beta < 0 or a.max_seconds <= 0:
        raise ValueError('Invalid regularization or time budget')
    if a.max_updates is not None and a.max_updates < 1:
        raise ValueError('--max-updates must be positive')
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    parent_binding=validate_parent_binding(a.data,a.parent,a.parent_sha256)
    d, u, presence, context, weights, train, val, scale, stats, targets = prepare_data(a.data)
    n_atoms, n_sites = d['presence'].shape[1], d['occupancy'].shape[1]
    with np.load(a.parent,allow_pickle=False) as parent:
        indices=parent['target_indices'] if 'target_indices' in parent else parent['li_indices']
        if len(parent['numbers']) != n_atoms or len(indices) != n_sites:
            raise ValueError('Parent dimensions differ from aligned geometry archive')
        expected_presence=np.ones((len(d['presence']),n_atoms),bool)
        expected_presence[:,indices]=d['occupancy'].astype(bool)
        if not np.array_equal(d['presence'],expected_presence):
            raise ValueError('Presence slots do not match occupancy at parent target indices, or host atoms are absent')
        if not np.array_equal(d['cell_A'],np.broadcast_to(parent['cell_A'],d['cell_A'].shape)):
            raise ValueError('Fixed-cell archive must preserve the exact parent cell; variable-cell training is a different task')
        if not np.isfinite(d['deformation']).all() or np.max(np.abs(d['deformation']-np.eye(3))) > 64*np.finfo(d['deformation'].dtype).eps:
            raise ValueError('Fixed-cell deformation must be identity up to floating-point roundoff')
    U = torch.tensor(u/scale, device=a.device)
    C = torch.tensor(context, device=a.device)
    P = torch.tensor(presence, device=a.device)
    W = torch.tensor(weights/weights[train].mean(), dtype=torch.float32, device=a.device)
    model = DisplacementModel(context_dim=context.shape[1], kind=a.kind,
                              coordinates=n_atoms*3).to(a.device)
    if a.warm_start:
        model.load_state_dict(trusted_checkpoint(a.warm_start, a.device)['state_dict'], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    ids_train = torch.tensor(train, device=a.device)
    ids_val = torch.tensor(val, device=a.device)
    metadata = dict(kind=a.kind, seed=a.seed, teacher=a.teacher, source=str(a.data.resolve()),
        source_sha256=digest(a.data), parent_sha256=digest(a.parent), parent=str(a.parent.resolve()), model_source_sha256=digest(Path(__file__).with_name('geometry_model.py')),
        trainer_source_sha256=digest(__file__), displacement_scale_A=scale,
        condition_stats=stats, target_conditions=targets, parent_binding=parent_binding,
        cell_validation='Exact archived cell equality to parent; deformation identity within 64 machine epsilons (dimensionless roundoff guard)',
        context_dim=context.shape[1],
        n_sites=n_sites, n_atoms_parent=n_atoms, coordinates=n_atoms*3, latent_dim=16,
        context_order=['occupancy','n_target/n_sites','cohort_onehot3','deformation_rowmajor9',
                       'energy_peratom_z','log1p(forceRMS/1eVA)_z','pressure_GPa_z','log1p(devstress/1GPa)_z'],
        cohorts=COHORTS, train_parents=len(set(d['structure_ids'][train])),
        val_parents=len(set(d['structure_ids'][val])),
        geometry_frame='Fixed supplied parent atom-index frame; affine/global translation removed; original observed mask orientations retained; absent slots masked; no Li-only symmetry augmentation',
        likelihood='Independent Gaussian in masked normalized displacement, fixed physical scale; MSE/2 summed over present coordinates; common constant omitted',
        scope='Fixed-cell prototype; supplied cell, no lattice prediction; target properties do not guarantee energy or force minimization',
        checkpoint_selection='Validation reconstruction plus validation_beta*KL; posterior conditioned coordinate fit',
        validation_beta=a.validation_beta, beta_final=a.beta_final, beta_warmup=a.beta_warmup,
        batch_size=a.batch_size, learning_rate=a.learning_rate, maximum_epochs=a.epochs,
        maximum_updates=a.max_updates, checkpoint_after=a.checkpoint_after,
        warm_start=None if not a.warm_start else dict(path=str(a.warm_start), sha256=digest(a.warm_start)),
        torch=torch.__version__, numpy=np.__version__, device=a.device, dtype='float32', tf32=False)
    (a.output_dir/'configuration.json').write_text(json.dumps(metadata, indent=2)+'\n')
    # Fixed validation noise is reused across epochs; never evaluate test frames here.
    eps = torch.randn((len(val),16), device=a.device)
    history, best, best_epoch, updates = [], float('inf'), 0, 0
    stop = 'maximum_epochs'
    for epoch in range(1, a.epochs+1):
        model.train()
        order = ids_train[torch.randperm(len(train), device=a.device)]
        total, kl_total, row_count = 0., 0., 0
        beta = a.beta_final*min(1., epoch/a.beta_warmup)
        for ids in order.split(a.batch_size):
            pred, mu, lv = model(U[ids], C[ids])
            rec = .5*((pred-U[ids]).square()*P[ids]).sum(-1)
            kl = .5*(mu.square()+lv.exp()-1-lv).sum(-1) if mu is not None else torch.zeros_like(rec)
            loss = ((rec+beta*kl)*W[ids]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite training objective')
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.)
            optimizer.step(); updates += 1
            total += float(loss)*len(ids); kl_total += float(kl.mean())*len(ids); row_count += len(ids)
            if a.max_updates is not None and updates >= a.max_updates:
                break
        model.eval()
        with torch.no_grad():
            if a.kind == 'cvae':
                mu, lv = model.encode(U[ids_val],C[ids_val])
                pred = model.decode(mu+torch.exp(.5*lv)*eps,C[ids_val])
                kl = .5*(mu.square()+lv.exp()-1-lv).sum(-1)
            else:
                pred = model.decode(None,C[ids_val]); kl = torch.zeros(len(val),device=a.device)
            rec = .5*((pred-U[ids_val]).square()*P[ids_val]).sum(-1)
            objective = float(((rec+a.validation_beta*kl)*W[ids_val]).sum()/W[ids_val].sum())
            rmse = float(torch.sqrt(((pred-U[ids_val]).square()*P[ids_val]*W[ids_val,None]).sum()
                                   /(P[ids_val]*W[ids_val,None]).sum())*scale)
        if not np.isfinite(objective):
            raise FloatingPointError('Nonfinite validation objective')
        row = dict(epoch=epoch, optimizer_updates=updates, beta=beta, train_objective=total/row_count,
                   train_KL=kl_total/row_count, val_objective=objective, val_coordinate_RMSE_A=rmse)
        history.append(row)
        # Short smoke fits still emit a checkpoint; full historical fits use the
        # recorded epoch-20 eligibility rule. This deviation is recorded below.
        eligible = epoch >= min(a.checkpoint_after,a.epochs) or (a.max_updates is not None and updates >= a.max_updates)
        if eligible and objective < best-1e-4:
            best, best_epoch = objective, epoch
            torch.save(dict(state_dict=model.state_dict(),configuration=metadata,epoch=epoch), a.output_dir/'best.pt')
        print(json.dumps(row), flush=True)
        if a.max_updates is not None and updates >= a.max_updates:
            stop='optimizer_update_cap'; break
        if time.perf_counter()-started >= a.max_seconds and best_epoch:
            stop='time_cap_after_complete_epoch'; break
        if epoch >= a.minimum_epochs and epoch-best_epoch >= a.patience:
            stop='validation_early_stop'; break
    with (a.output_dir/'history.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(history[0]));writer.writeheader();writer.writerows(history)
    summary=dict(status='FIT_COMPLETE',best_epoch=best_epoch,best_val_objective=best,
                 epochs=len(history),optimizer_updates=updates,stop_reason=stop,
                 training_seconds=time.perf_counter()-started,checkpoint_sha256=digest(a.output_dir/'best.pt'),
                 heldout_test_evaluated=False,short_fit_checkpoint_rule=a.epochs<a.checkpoint_after,
                 validation_note='Posterior-conditioned coordinate fit; independently check prior-decoded E/F/stress and matched relaxation savings')
    (a.output_dir/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print('GEOMETRY_TRAINED',a.output_dir,flush=True)


if __name__ == '__main__':
    main()
