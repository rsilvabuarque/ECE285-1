"""Fit only the secondary teacher's evaluation bins on protected H0 training keys."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
from cathode_cvae.conditions import fit_schema,save_schema
from cathode_cvae.data import sha256
from cathode_cvae.mlip.registry import MODELS
p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args();r=a.root;historical=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/final_project/preliminary_dataset_20261106')
with np.load(r/'artifacts/current/improvement_v1/training_round0.npz') as z:d={k:z[k] for k in z.files}
lookup={};sources=[]
for p in [r/'artifacts/current/core/data/learning_O3_N48_pilot_mace.npz',historical/'data/generated_scalar_mace.npz']:
 with np.load(p) as z:
  for sid,e,natoms in zip(z['structure_ids'].astype(str),z['energy_eV'],z['n_atoms']):lookup.setdefault(format(int(sid.split('_')[-1],16),'048b'),float(e/natoms))
 sources.append(dict(path=str(p),sha256=sha256(p)))
missing=set(d['canonical_masks'])-set(lookup)
if missing:raise ValueError(f'MACE labels missing for{len(missing)} training-key geometries')
e=np.array([lookup[k] for k in d['canonical_masks']]);np.savez_compressed(r/'runs/overnight_20261107/data/training_mace_evaluation_labels.npz',**{**d,'energy_eV_atom':e})
provenance=dict(teacher='mace',teacher_head='omat_pbe',teacher_checkpoint_name=MODELS['mace']['name'],teacher_checkpoint_hashes=MODELS['mace']['files'],parent_sha256=sha256(r/'artifacts/current/core/data/parents/O3_N48_pilot.npz'),geometry_protocol='pristine-derived deletion; unrelaxed; symmetry-orbit scalar labels from original H0 sources',source_labels=sources,protected_split_sha256=hashlib.sha256('\n'.join(f'{sid}\t{sp}' for sid,sp in zip(d['structure_ids'],d['split'])).encode()).hexdigest(),use='MACE evaluation namespace only; generative model remains UMA-conditioned',orientation_note='H0 inherited symmetry-orbit scalar labels; no force vectors or relaxed geometries reused')
s=fit_schema(d['n_Li'],e,d['split'],n_sites=48,provenance=provenance);save_schema(s,r/'runs/overnight_20261107/data/condition_schema_mace.json');print('MACE_SCHEMA_COMPLETE',len(e),s['identity'])
