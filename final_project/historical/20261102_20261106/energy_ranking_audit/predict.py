"""Evaluate frozen UMA/omat on exact archived geometries; never relax or relabel them.

E: eV/cell; F: eV/Angstrom; tensile-positive stress: eV/Angstrom^3,
Voigt xx, yy, zz, yz, xz, xy. Hu virials use the conventional -V*stress.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = Path('/global/cfs/cdirs/m5056/rsb/2026/cathode_relithiation/cathode_relithiation_project')
TINY = ROOT / 'results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny'
SLIDE = TINY / 'vasp_workflow/analysis_completed_only/uma_vs_vasp_completed_cases.csv'
LABELS = TINY / 'uma_finetune_workflow/launch_full_completed_20260320_032818/dataset_completed_only_strict_v3_efs_same_cases/extxyz/raw'
MODEL = Path('/global/homes/r/rsb/.cache/fairchem/models--facebook--UMA/snapshots/f611b917d9c68566bbbeccbb0aa0f7cad1696cb2/checkpoints/uma-s-1p2.pt')
MODEL_SHA = 'ba5c0d912efa22dc238e5fb1b5b7f66ee2e68c48c1b95b7cfd7fe1da5938398b'


def dump(path, data):
    Path(path).write_text(json.dumps(data, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def key_for(numbers, cell, positions):
    h = hashlib.sha256()
    for a, dtype in ((numbers, '<i4'), (cell, '<f8'), (positions, '<f8')):
        h.update(np.asarray(a, dtype=dtype).tobytes())
    return h.hexdigest()


def records():
    """Yield actual labels with deterministic IDs; a hash permits exact duplicate reuse."""
    from ase import Atoms
    from ase.io import read
    from ase.data import atomic_numbers
    with SLIDE.open() as f:
        old = list(csv.DictReader(f))
    for i, row in enumerate(sorted(old, key=lambda r: r['case_id'])):
        path = LABELS / (row['case_id'] + '.extxyz')
        a = read(path, format='extxyz')
        e, forces, stress = a.get_potential_energy(), a.get_forces(), a.get_stress()
        if abs(e - float(row['vasp_energy_eV'])) > 1e-7:
            raise ValueError(f'Historical label/slide energy mismatch: {path}')
        a.calc = None
        yield dict(dataset='historical', phase='O3', cohort='rattled', structure_id=f'structure_{i+1}',
                   source=str(path), frame=0, case_id=row['case_id'], temperature_K=float(row['temperature_k']),
                   pressure_MPa=float(row['pressure_mpa']), cached_uma_eV=float(row['uma_energy_eV']),
                   atoms=a, dft_energy=e, dft_forces=forces, dft_stress=stress)
    serial = 0
    for name, root in [('bulk', ROOT/'for_chat_gpt/LiCoO2_All_data/Data_Set'),
                       ('migration', ROOT/'for_chat_gpt/Final_Dataset/dpgen_work/Total_Data_Set')]:
        for ef in sorted(root.rglob('set.*/energy.npy')):
            sd, system = ef.parent, ef.parent.parent
            types = np.loadtxt(system/'type.raw', dtype=int, ndmin=1)
            symbols = (system/'type_map.raw').read_text().split()
            numbers = np.array([atomic_numbers[symbols[t]] for t in types])
            arrays = {n: np.load(sd/(n+'.npy'), mmap_mode='r') for n in ('energy','coord','box','force','virial')}
            phase = 'migration' if name == 'migration' else ('O3' if '1_O3' in ef.parts else 'spinel' if '2_Spinel' in ef.parts else 'O1')
            cohort = 'migration' if name == 'migration' else 'optimized' if '1_Opt_Structures' in ef.parts else 'DPGEN'
            for j in range(len(arrays['energy'])):
                serial += 1
                cell = np.asarray(arrays['box'][j], dtype=float).reshape(3,3)
                a = Atoms(numbers=numbers, positions=np.asarray(arrays['coord'][j], dtype=float).reshape(-1,3), cell=cell, pbc=True)
                volume = a.get_volume()
                w = np.asarray(arrays['virial'][j], dtype=float).reshape(3,3)
                if volume <= 0 or not np.isfinite(w).all():
                    raise ValueError(f'Invalid cell/virial: {sd}:{j}')
                yield dict(dataset='hu', phase=phase, cohort=cohort, structure_id=f'structure_{serial}',
                           source=str(sd), frame=j, case_id='', temperature_K=None, pressure_MPa=None,
                           cached_uma_eV=None, atoms=a, dft_energy=float(arrays['energy'][j]),
                           dft_forces=np.asarray(arrays['force'][j],dtype=float).reshape(-1,3),
                           dft_stress=-(w+w.T)[[0,1,2,1,0,0],[0,1,2,2,2,1]]/(2*volume))


def load_cache(out):
    cache = {}
    for f in sorted((out/'predictions').glob('chunk_*.npz')):
        with np.load(f) as z:
            for i, key in enumerate(z['keys']):
                cache[str(key)] = (float(z['energy'][i]), z['force'][z['offset'][i]:z['offset'][i+1]].copy(), z['stress'][i].copy())
    return cache


def save_chunk(out, pending):
    if not pending: return
    dest = out/'predictions'
    dest.mkdir(exist_ok=True)
    n = max((int(f.stem.split('_')[1]) for f in dest.glob('chunk_*.npz')),default=-1)+1
    keys, vals = zip(*pending)
    offsets = np.cumsum([0]+[len(v[1]) for v in vals])
    np.savez_compressed(dest/f'chunk_{n:04d}.npz', keys=np.array(keys), energy=[v[0] for v in vals],
                        force=np.concatenate([v[1] for v in vals]), stress=np.stack([v[2] for v in vals]), offset=offsets)
    pending.clear()


def run(out, max_seconds):
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME','').startswith('nid'):
        raise RuntimeError('Inference/analysis requires the requested compute allocation, not a login node')
    print('STAGE: importing installed UMA environment',flush=True)
    import torch
    from fairchem.core import FAIRChemCalculator
    from fairchem.core.units.mlip_unit import load_predict_unit
    from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
    if not torch.cuda.is_available(): raise RuntimeError('CUDA is unavailable')
    out.mkdir(parents=True, exist_ok=True)
    print('STAGE: checking checkpoint identity',flush=True)
    model_sha = sha(MODEL)
    if model_sha != MODEL_SHA: raise RuntimeError('Checkpoint hash differs from the inspected lock')
    torch.set_num_threads(4)
    settings = InferenceSettings(tf32=False, merge_mole=False, compile=False, activation_checkpointing=True,
                                 base_precision_dtype='float32', execution_mode='general', external_graph_gen=False,
                                 internal_graph_gen_version=1)
    sig = dict(checkpoint=str(MODEL), sha256=model_sha, task='omat',
               inference_settings=settings.to_omegaconf(), torch=torch.__version__,
               fairchem=importlib.metadata.version('fairchem-core'), script_sha256=sha(__file__),
               job_id=os.environ['SLURM_JOB_ID'], account=os.environ.get('SLURM_JOB_ACCOUNT'),
               host=os.environ.get('SLURMD_NODENAME'), gpu=torch.cuda.get_device_name(),
               stress_convention='ASE tensile positive; xx yy zz yz xz xy; eV/A^3',
               hu_stress_provenance='Conventional DeepMD virial conversion -W/V; original exporter unavailable',
               source_labels='Archived VASP/DFT labels; original Hu frame-level SCF/spin records unavailable')
    from omegaconf import OmegaConf
    if OmegaConf.is_config(sig['inference_settings']):
        sig['inference_settings'] = OmegaConf.to_container(sig['inference_settings'], resolve=True)
    # The installed export contains Python sets for optional task names.
    sig['inference_settings'] = json.loads(json.dumps(sig['inference_settings'],
        default=lambda obj: sorted(obj) if isinstance(obj,set) else str(obj)))
    if (out/'model.json').exists():
        old=json.loads((out/'model.json').read_text())
        if any(old[k]!=sig[k] for k in ('sha256','task','inference_settings','fairchem','torch')):
            raise RuntimeError('Refusing cache reuse across different scorer signatures')
    dump(out/'model.json',sig)
    print('STAGE: loading pinned UMA/omat checkpoint',flush=True)
    torch.serialization.add_safe_globals([slice])
    calc = FAIRChemCalculator(load_predict_unit(MODEL, inference_settings=settings, device='cuda'),task_name='omat')
    print('STAGE: evaluating exact archived geometries',flush=True)
    cache=load_cache(out); pending=[]; rows=[]; seen_labels={}; times=[]; start=time.monotonic()
    errors=[]; counters=dict(raw=0, computed=0, reused=0); last_reported=0
    for rec in records():
        a=rec.pop('atoms'); de=rec.pop('dft_energy'); df=rec.pop('dft_forces'); ds=rec.pop('dft_stress')
        key=key_for(a.numbers,a.cell.array,a.positions)
        counters['raw']+=1
        if key in seen_labels:
            prev=seen_labels[key]
            if not (de==prev[0] and np.array_equal(df,prev[1]) and np.array_equal(ds,prev[2])):
                raise ValueError('Identical geometry has conflicting DFT labels')
        else: seen_labels[key]=(de,df.copy(),ds.copy())
        if key not in cache:
            t=time.monotonic()
            try:
                image=a.copy(); image.calc=calc; calc.reset()
                ue=float(image.get_potential_energy()); uf=image.get_forces(); us=image.get_stress()
                if not all(np.isfinite(x).all() for x in (ue,uf,us)): raise ValueError('Nonfinite UMA result')
                cache[key]=(ue,uf,us); pending.append((key,cache[key])); counters['computed']+=1
                times.append(time.monotonic()-t)
            except Exception as e:
                save_chunk(out,pending)
                errors.append(dict(structure_id=rec['structure_id'],dataset=rec['dataset'],source=rec['source'],frame=rec['frame'],error=repr(e)))
                dump(out/'errors.json',errors)
                raise
        else: counters['reused']+=1
        ue,uf,us=cache[key]
        symbols=np.array(a.get_chemical_symbols()); nc=int(np.sum(symbols=='Co')); nl=int(np.sum(symbols=='Li'))
        ferr=uf-df; serr=us-ds
        row={**rec,'geometry_key':key,'n_atoms':len(a),'n_Li':nl,'n_Co':nc,'x':nl/nc,
             'dft_energy_eV':de,'uma_energy_eV':ue,'volume_A3':a.get_volume(),
             'dft_force_rms':float(np.sqrt(np.mean(df*df))),'uma_force_rms':float(np.sqrt(np.mean(uf*uf))),
             'force_rmse':float(np.sqrt(np.mean(ferr*ferr))),
             'force_mean_vector_error':float(np.linalg.norm(ferr,axis=1).mean()),
             'stress_rmse':float(np.sqrt(np.mean(serr*serr))),
             'dft_pressure_eV_A3':float(-ds[:3].mean()),'uma_pressure_eV_A3':float(-us[:3].mean())}
        for species in ('Li','Co','O'):
            ix=symbols==species
            row[f'force_rmse_{species}']=float(np.sqrt(np.mean(ferr[ix]**2))) if ix.any() else None
        for k in range(6): row[f'dft_stress_{k}']=float(ds[k]); row[f'uma_stress_{k}']=float(us[k])
        rows.append(row)
        if len(pending)>=128: save_chunk(out,pending)
        if counters['computed']-last_reported>=64:
            last_reported=counters['computed']
            elapsed=time.monotonic()-start
            print(json.dumps(dict(counters,elapsed_s=round(elapsed,1),median_s=round(float(np.median(times[-64:])),4))),flush=True)
        if len(times)==32:
            estimate=float(np.median(times[-24:]))*(10932+524)
            dump(out/'pilot.json',dict(n=32,median_s=float(np.median(times[-24:])),estimated_total_s=estimate,hard_compute_limit_s=max_seconds))
            if estimate>max_seconds-(time.monotonic()-start)-300:
                save_chunk(out,pending)
                raise RuntimeError('Pilot forecast exceeds bounded allocation; no automatic extension')
        if time.monotonic()-start>max_seconds-180:
            save_chunk(out,pending)
            raise RuntimeError('Bounded wall time reached; cache preserved')
    save_chunk(out,pending)
    first={}
    for r in rows:
        key=r['geometry_key']; r['duplicate_of']=first.get(key,''); first.setdefault(key,r['dataset']+':'+r['structure_id'])
    with (out/'frame_metrics.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    dump(out/'completion.json',dict(counters,unique_geometries=len(cache),elapsed_s=time.monotonic()-start,
                                   errors=errors,status='INFERENCE_COMPLETE'))
    print('INFERENCE_COMPLETE',json.dumps(counters),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',type=Path,default=HERE/'results');p.add_argument('--max-seconds',type=int,default=6600)
    args=p.parse_args(); run(args.output,args.max_seconds)
