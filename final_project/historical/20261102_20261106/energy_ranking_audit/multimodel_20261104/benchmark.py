"""Exact-geometry E/F/stress audit; one calculator process per allocated GPU.

No structure relaxation or DFT is performed. Reference and prediction energies
are total eV/cell, forces eV/Angstrom, and ASE tensile-positive stresses eV/A^3
in xx yy zz yz xz xy order. Full ordered-geometry duplicates share predictions.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import importlib.metadata as metadata
import importlib.util
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

HERE = Path(__file__).resolve().parent


def dump(path, data):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(data, indent=2, allow_nan=False, default=str) + '\n')
    tmp.replace(path)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024**2), b''):
            h.update(block)
    return h.hexdigest()


def compute_guard():
    if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME', '').startswith('nid'):
        raise RuntimeError('Use an allocated compute node for export/inference/analysis')


def export():
    compute_guard()
    spec = importlib.util.spec_from_file_location('archived_inputs', HERE.parent/'predict.py')
    old = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(old)
    unique, lookup, raw = [], {}, []
    numbers, positions, forces, cells, stresses, energies, offsets = [], [], [], [], [], [], [0]
    start = time.monotonic()
    for rec in old.records():
        a = rec.pop('atoms')
        e, f, s = rec.pop('dft_energy'), rec.pop('dft_forces'), rec.pop('dft_stress')
        key = old.key_for(a.numbers, a.cell.array, a.positions)
        if key not in lookup:
            i = len(unique)
            lookup[key] = i
            symbols = np.asarray(a.get_chemical_symbols())
            nc, nl = int(np.sum(symbols=='Co')), int(np.sum(symbols=='Li'))
            unique.append(dict(index=i,geometry_key=key,**rec,n_atoms=len(a),n_Co=nc,n_Li=nl,
                               x=nl/nc,volume_A3=a.get_volume()))
            numbers.append(a.numbers); positions.append(a.positions)
            forces.append(f); cells.append(a.cell.array); stresses.append(s); energies.append(e)
            offsets.append(offsets[-1]+len(a))
        else:
            i = lookup[key]
            if not (e==energies[i] and np.array_equal(f,forces[i]) and np.array_equal(s,stresses[i])):
                raise ValueError(f'Conflicting DFT labels for duplicate {key}')
        raw.append(dict(unique_index=i,geometry_key=key,**rec))
    for name, rows in [('frames.csv',unique),('raw_frame_map.csv',raw)]:
        with (HERE/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    np.savez_compressed(HERE/'reference_frames.npz',numbers=np.concatenate(numbers),
                        positions=np.concatenate(positions),forces=np.concatenate(forces),
                        cell=np.asarray(cells),stress=np.asarray(stresses),energy=np.asarray(energies),
                        offsets=np.asarray(offsets,dtype=np.int64))
    assert len(raw)==15468 and len(unique)==11456
    dump(HERE/'reference_receipt.json',dict(raw_frames=len(raw),unique_geometries=len(unique),
         duplicates=len(raw)-len(unique),historical=sum(r['dataset']=='historical' for r in unique),
         hu=sum(r['dataset']=='hu' for r in unique),elapsed_s=time.monotonic()-start,
         importer_sha256=sha(HERE.parent/'predict.py'),reference_sha256=sha(HERE/'reference_frames.npz'),
         geometry_manifest_sha256=sha(HERE/'frames.csv'),job_id=os.environ['SLURM_JOB_ID'],
         conventions=__doc__,Hu_label_convention='standard DeepMD virial converted as -W/V; original exporter unavailable'))
    print('REFERENCE_EXPORTED',len(unique),flush=True)


def load_calculator(name, dtype):
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA unavailable')
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32=False
    torch.backends.cudnn.allow_tf32=False
    paths=json.loads((HERE/'checkpoints/paths.json').read_text())
    receipt=dict(model=name,dtype=dtype,torch=torch.__version__,ase=metadata.version('ase'),
                 gpu=torch.cuda.get_device_name(),job_id=os.environ['SLURM_JOB_ID'],
                 step_id=os.environ.get('SLURM_STEP_ID'),host=os.environ['SLURMD_NODENAME'],
                 account=os.environ.get('SLURM_JOB_ACCOUNT'),CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'),
                 script_sha256=sha(__file__),stress_convention='ASE tensile-positive; xx yy zz yz xz xy; eV/A^3')
    if name=='uma':
        from fairchem.core import FAIRChemCalculator
        from fairchem.core.units.mlip_unit import load_predict_unit
        from fairchem.core.units.mlip_unit.api.inference import InferenceSettings
        settings=InferenceSettings(tf32=False,nitro=False,merge_mole=False,compile=False,
              execution_mode='general',base_precision_dtype=dtype,internal_graph_gen_version=2,
              activation_checkpointing=True)
        checksum=sha(paths[name])
        assert checksum=='b2673b85037b075674c25f55c34ffe1ff1e15db924be977b10a184765df0d5ce'
        calc=FAIRChemCalculator(load_predict_unit(paths[name],inference_settings=settings,device='cuda'),task_name='omat')
        receipt.update(checkpoint=paths[name],sha256=checksum,checkpoint_name='uma-s-1p2p1',task='omat',
                       fairchem=metadata.version('fairchem-core'),settings=str(settings),dispersion=False)
    elif name=='mace':
        from mace.calculators import MACECalculator
        checksum=sha(paths[name])
        assert checksum=='a522eb7f59c7879963d41586528f4980baf33e086c94aa92e3eafdeccad3be47'
        calc=MACECalculator(model_paths=paths[name],device='cuda',default_dtype=dtype,
                            head='omat_pbe',enable_cueq=False,compile_mode=None)
        assert calc.head=='omat_pbe' and 'omat_pbe' in calc.available_heads
        assert next(calc.models[0].parameters()).device.type=='cuda'
        receipt.update(checkpoint=paths[name],sha256=checksum,checkpoint_name='MACE-MH-1',head='omat_pbe',
                       mace=metadata.version('mace-torch'),dispersion=False)
    else:
        import matgl
        matgl.set_default_dtype('float',64 if dtype=='float64' else 32)
        from matgl.ext.ase import PESCalculator
        potential=matgl.load_model(paths[name]).to(device='cuda',dtype=getattr(torch,dtype)).eval()
        assert potential.data_mean.device.type=='cuda'
        assert potential.calc_stresses and potential.calc_forces
        calc=PESCalculator(potential=potential,stress_unit='eV/A3',stress_weight=1.0,use_voigt=True)
        receipt.update(checkpoint=paths[name],checkpoint_name='M3GNet-PES-MatPES-PBE-2025.2',
                       checkpoint_hashes={p.name:sha(p) for p in Path(paths[name]).iterdir() if p.is_file()},
                       matgl=metadata.version('matgl'),stress_unit='eV/A3',stress_weight=1.0)
    return calc,receipt


def atoms_at(data, i):
    from ase import Atoms
    lo,hi=data['offsets'][i:i+2]
    return Atoms(numbers=data['numbers'][lo:hi],positions=data['positions'][lo:hi],cell=data['cell'][i],pbc=True)


def evaluate(calc,a):
    a=a.copy();a.calc=calc;calc.reset()
    energy=float(a.get_potential_energy())
    force=np.asarray(a.get_forces(),dtype=np.float64)
    stress=np.asarray(a.get_stress(),dtype=np.float64)
    if force.shape!=(len(a),3) or stress.shape!=(6,) or not all(np.isfinite(v).all() for v in [energy,force,stress]):
        raise ValueError('Invalid E/F/stress prediction')
    return energy,force,stress


def derivative_check(calc,a):
    """Two-step central differences check output sign/units, not DFT accuracy.

    Steps are numerical probes in Angstrom and dimensionless isotropic strain.
    Floors below only bound float64 arithmetic/finite-difference agreement;
    they are not force-field accuracy tolerances or physical acceptance gates.
    """
    e,f,s=evaluate(calc,a)
    atom,axis=np.unravel_index(np.abs(f).argmax(),f.shape)
    fd=[]
    for h in [1e-3,5e-4]:
        ap,am=a.copy(),a.copy();ap.positions[atom,axis]+=h;am.positions[atom,axis]-=h
        ep=evaluate(calc,ap)[0];em=evaluate(calc,am)[0];fd.append(-(ep-em)/(2*h))
    sd=[]
    for h in [1e-4,5e-5]:
        ap,am=a.copy(),a.copy();ap.set_cell(a.cell.array*(1+h),scale_atoms=True);am.set_cell(a.cell.array*(1-h),scale_atoms=True)
        sd.append((evaluate(calc,ap)[0]-evaluate(calc,am)[0])/(2*h*3*a.get_volume()))
    fbound=max(1e-5,10*abs(fd[1]-fd[0]));sbound=max(1e-7,10*abs(sd[1]-sd[0]))
    result=dict(atom=int(atom),axis=int(axis),force_eV_A=float(f[atom,axis]),force_FD_eV_A=fd,
                hydrostatic_stress_eV_A3=float(s[:3].mean()),stress_FD_eV_A3=sd,
                force_numeric_bound=fbound,stress_numeric_bound=sbound,
                force_difference=abs(float(f[atom,axis])-fd[1]),stress_difference=abs(float(s[:3].mean())-sd[1]))
    result['passed']=bool(result['force_difference']<=fbound and result['stress_difference']<=sbound)
    return result


def predict(args):
    compute_guard()
    with np.load(HERE/'reference_frames.npz') as archive:
        data={key:archive[key] for key in archive.files}
    rows=list(csv.DictReader((HERE/'frames.csv').open()))
    out=HERE/'results'/args.model/(f'pilot_{args.dtype}' if args.pilot else f'shard_{args.shard:02d}_of_{args.nshards:02d}')
    out.mkdir(parents=True,exist_ok=True)
    print('LOADING',args.model,args.dtype,flush=True)
    calc,receipt=load_calculator(args.model,args.dtype)
    reference_hash=sha(HERE/'reference_frames.npz')
    if (out/'model.json').exists() and any(out.glob('chunk_*.npz')):
        prior=json.loads((out/'model.json').read_text())
        prior_ref=prior.get('reference_sha256')
        if prior_ref is None and (out/'completion.json').exists():
            prior_ref=json.loads((out/'completion.json').read_text()).get('reference_sha256')
        fields=['model','dtype','torch','ase','sha256','checkpoint_hashes','task','head','fairchem','mace','matgl','settings']
        if prior_ref!=reference_hash or any(prior.get(k)!=receipt.get(k) for k in fields):
            raise RuntimeError('Refusing cached chunks with a different or unrecorded model/reference signature')
    receipt['reference_sha256']=reference_hash
    dump(out/'model.json',receipt)
    if args.pilot:
        # Cover source, family, cohort, Li-free and optimized/thermal cases.
        chosen=[]
        for condition in [lambda r:r['dataset']=='historical' and r['n_Li']=='18',
                          lambda r:r['dataset']=='hu' and r['phase']=='O3' and r['cohort']=='optimized' and r['n_Li']=='10',
                          lambda r:r['dataset']=='hu' and r['phase']=='O3' and r['cohort']=='optimized' and r['n_Li']=='12',
                          lambda r:r['phase']=='spinel',lambda r:r['phase']=='O1',lambda r:r['phase']=='migration']:
            chosen.extend([int(r['index']) for r in rows if condition(r)][:2])
        chosen=list(dict.fromkeys(chosen))
    else:
        chosen=list(range(args.shard,len(rows),args.nshards))
    cached={}
    for path in sorted(out.glob('chunk_*.npz')):
        with np.load(path) as z:
            for idx in z['indices']:cached[int(idx)]=True
    pending=[];timings=[];start=time.monotonic();errors=[]
    def save():
        if not pending:return
        offset=np.cumsum([0]+[len(p[2]) for p in pending])
        dest=out/f'chunk_{len(list(out.glob("chunk_*.npz"))):05d}.npz'
        np.savez_compressed(dest,indices=[p[0] for p in pending],energy=[p[1] for p in pending],
                            force=np.concatenate([p[2] for p in pending]),stress=np.stack([p[3] for p in pending]),offsets=offset)
        pending.clear()
    for i in chosen:
        if i in cached:continue
        try:
            t=time.monotonic();e,f,s=evaluate(calc,atoms_at(data,i));timings.append(time.monotonic()-t)
            pending.append((i,e,f,s));cached[i]=True
        except Exception as exc:
            errors.append(dict(index=i,geometry_key=rows[i]['geometry_key'],error=repr(exc)))
            dump(out/'errors.json',errors);save();raise
        if len(pending)>=64:save()
        if len(timings)%64==0:
            progress=dict(computed=len(timings),required=len(chosen),elapsed_s=time.monotonic()-start,
                          median_s=float(np.median(timings[-64:])))
            dump(out/'progress.json',progress);print(json.dumps(progress),flush=True)
    save()
    result=dict(status='COMPLETE',model=args.model,shard=args.shard,nshards=args.nshards,
                predictions=len(chosen),elapsed_s=time.monotonic()-start,errors=errors,
                median_inference_s=float(np.median(timings[1:])) if len(timings)>1 else None,
                reference_sha256=reference_hash)
    if args.pilot:
        opt=next(i for i in chosen if rows[i]['dataset']=='hu' and rows[i]['cohort']=='optimized')
        checks=[dict(index=i,**derivative_check(calc,atoms_at(data,i))) for i in [chosen[0],opt]]
        result['derivative_checks']=checks
        result['derivative_checks_passed']=all(x['passed'] for x in checks)
        dump(out/'completion.json',result)
        if not result['derivative_checks_passed']:raise RuntimeError('Derivative sign/units check failed; inspect pilot receipt')
    else:dump(out/'completion.json',result)
    print('COMPLETE',json.dumps(result,default=str),flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['export','predict'])
    parser.add_argument('--model',choices=['uma','mace','m3gnet'])
    parser.add_argument('--dtype',choices=['float32','float64'],default='float64')
    parser.add_argument('--shard',type=int,default=0)
    parser.add_argument('--nshards',type=int,default=1)
    parser.add_argument('--pilot',action='store_true')
    args=parser.parse_args()
    if args.action=='export':export()
    else:predict(args)
