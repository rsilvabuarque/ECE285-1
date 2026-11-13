"""Freeze matched decoded starts and actual archived phase-context structures."""
import csv,hashlib,json,os
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
HERE=Path(__file__).resolve().parent
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    with np.load(HERE/'data/decoded_geometry_candidates.npz') as z:d={k:z[k] for k in z.files}
    with np.load(HERE/'data/parents/O3_N48_pilot.npz') as z:p={k:z[k] for k in z.files}
    nli=d['atom_presence'][:,p['li_indices']].sum(1);chosen=[]
    for n,k in [(12,11),(24,10),(36,11)]:
        ii=np.flatnonzero((nli==n)&(d['methods']=='deletion'))[:k];assert len(ii)==k
        chosen.extend(d['source_mask_ids'][ii].tolist())
    tasks=[];folder=HERE/'data/decoded_relax_inputs';folder.mkdir(exist_ok=True)
    for sid in chosen:
        for method in ['deletion','regressor','cvae_draw0']:
            ii=np.flatnonzero((d['source_mask_ids']==sid)&(d['methods']==method));assert len(ii)==1;i=int(ii[0]);assert d['valid'][i]
            keep=d['atom_presence'][i];slots=np.flatnonzero(keep);a=Atoms(numbers=p['numbers'][keep],positions=d['positions_parent_A'][i,keep],cell=d['cell_A'][i],pbc=True);a.set_array('parent_slot',slots)
            path=folder/f'{sid}__{method}.extxyz';write(path,a,format='extxyz')
            tasks.append(dict(task_id=f'decoded_relax__uma__{sid}__{method}',stage='decoded_relax',model='uma',parent='O3_N48_pilot',structure_id=str(d['structure_ids'][i]),source_mask_id=sid,method=method,n_Li=int(nli[i]),split='fresh_mask_assessment',canonical_mask=''.join(str(int(x)) for x in ~keep[p['li_indices']]),input_file=str(path),input_sha256=sha(path),cell=False,optimizer='FIRE',factor=.25,max_steps=500,stride=5))
    (HERE/'data/decoded_relax_tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
    source=HERE.parent/'energy_ranking_audit/multimodel_20261104/structure_mapping';meta={int(r['index']):r for r in csv.DictReader((source/'joint_template_dataset.csv').open())}
    archive=source/'joint_training_candidates.npz'
    with np.load(archive) as z:raw={k:z[k] for k in z.files}
    rng=np.random.default_rng(20261403);selected=[]
    for phase in ['O3','O1','spinel']:
        groups={}
        for i,ref in enumerate(raw['reference_index']):
            r=meta[int(ref)]
            if r['phase']==phase and r['dataset']!='historical':groups.setdefault((r['cohort'],int(r['n_Li']),int(r['n_Co'])),[]).append(i)
        keys=sorted(groups);pool=[]
        for key in keys:rng.shuffle(groups[key])
        while len(pool)<16:
            for key in keys:
                if groups[key] and len(pool)<16:pool.append(groups[key].pop())
        selected.extend(pool)
    maxN=max(int(raw['offsets'][i+1]-raw['offsets'][i]) for i in selected);positions=[];numbers=[];presence=[];forces=[];rows=[]
    for i in selected:
        lo,hi=raw['offsets'][i:i+2];N=int(hi-lo);ref=int(raw['reference_index'][i]);r=meta[ref]
        xyz=np.zeros((maxN,3));xyz[:N]=raw['positions_A'][lo:hi];zz=np.zeros(maxN,int);zz[:N]=raw['numbers'][lo:hi];ff=np.zeros((maxN,3));ff[:N]=raw['dft_forces_eV_A'][lo:hi]
        positions.append(xyz);numbers.append(zz);presence.append(np.arange(maxN)<N);forces.append(ff)
        rows.append(dict(structure_id=f'phase_archive_{ref}',reference_index=ref,phase=r['phase'],cohort=r['cohort'],dataset=r['dataset'],n_Li=r['n_Li'],n_Co=r['n_Co'],x=r['x'],source=r['source'],frame=r['frame']))
    path=HERE/'data/phase_context_candidates.npz';np.savez_compressed(path,structure_ids=[r['structure_id'] for r in rows],numbers_parent=numbers,positions_parent_A=positions,atom_presence=presence,cell_A=raw['cell_A'][selected],dft_energy_eV=raw['dft_energy_eV'][selected],dft_forces_parent_eV_A=forces,dft_stress_eV_A3=raw['dft_stress_eV_A3'][selected],reference_index=raw['reference_index'][selected],phase=[r['phase'] for r in rows])
    with (HERE/'data/phase_context_manifest.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    digest=sha(path);tasks_phase=[]
    for m in ['uma','mace','sevennet','chgnet','m3gnet']:
        for start in range(0,len(rows),16):tasks_phase.append(dict(task_id=f'phase_context__{m}__{start:04d}',stage='phase_context',model=m,input_npz=str(path),input_sha256=digest,start=start,stop=min(start+16,len(rows))))
    (HERE/'data/phase_context_tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks_phase))
    receipt=dict(decoded_relaxations=len(tasks),decoded_independent_masks=len(chosen),decoded_methods=['deletion','regressor','cvae_draw0'],decoded_selection='Firstfixed32freshmasksstratified11/10/11atLi12/24/36; firstCVAEdraw, nooraclebest-ofselection; sameUMA/FIREfixedcellcriteria.',phase_frames=len(rows),phase_counts={x:sum(r['phase']==x for r in rows) for x in ['O3','O1','spinel']},phase_input_sha256=digest,archive_sha256=sha(archive),phase_scope='Actualarchivedgeometry/DFTlabels, notcompletedparenttemplates. ExistingSCF/spinprovenanceinherited, notrecertified. Differentphases/compositionsnotorderedbyrawtotalenergy; x=nLi/nCo.')
    (HERE/'data/diagnostics_design.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
