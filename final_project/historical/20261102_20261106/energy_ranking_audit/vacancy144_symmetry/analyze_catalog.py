"""Exact parent-symmetry audit of the existing 144-site fixed-host catalog.

Only existing structures/scores are read. No scoring, relaxation or raw-data
editing occurs. Mask equivalence is established by species-preserving periodic
parent permutations, independently of cached energies.
"""
from __future__ import annotations
import os
for name in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):
    os.environ[name]='1'
import csv
import hashlib
import itertools
import json
import math
import time
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
import scipy
from scipy.spatial import cKDTree
import spglib
from ase.io import read
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE=Path(__file__).resolve().parent
PROJECT=Path('/global/cfs/cdirs/m5056/rsb/2026/cathode_relithiation/cathode_relithiation_project')
SOURCE=PROJECT/'results/publication/default_systems/deepmd_collaborator_electrode/full_study_4x6x2/structure_selection'
MANIFEST=SOURCE/'candidate_manifest.csv'
SCORES=SOURCE/'uma_omat_top500_candidates/uma_omat_all_candidate_energies.csv'
PARENT=SOURCE/'pristine_4x6x2_POSCAR'
OLD=HERE.parents[1]/'collaborator_brief/assets/symmetry_counts.json'
WORKER=None

def dump(path,data):Path(path).write_text(json.dumps(data,indent=2,allow_nan=False)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write_csv(path,rows):
    rows=list(rows)
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

def make_symmetry(a):
    h,f,n=a.cell.array,a.get_scaled_positions(),a.numbers
    sensitivity=[]
    for tol in (1e-6,1e-5,1e-4,.001,.01):
        d=spglib.get_symmetry_dataset((h,f,n),symprec=tol)
        sensitivity.append(dict(symprec_A=tol,spacegroup=d.international,n_operations=len(d.rotations)))
    d=spglib.get_symmetry_dataset((h,f,n),symprec=1e-5)
    trees={z:cKDTree(f[n==z]%1,boxsize=1.) for z in set(n)}
    perms=[];residual=0.
    for R,t in zip(d.rotations,d.translations):
        for z,tree in trees.items():
            ff=f[n==z];trans=(ff@R.T+t)%1
            _,p=tree.query(trans)
            assert len(set(p))==len(p)
            df=trans-ff[p];df-=np.rint(df)
            residual=max(residual,float(np.linalg.norm(df@h,axis=1).max()))
            if z==3:perms.append(tuple(map(int,p)))
    perms=np.array(sorted(set(perms)),dtype=np.int16)
    group={p.tobytes() for p in perms}
    assert np.arange(144,dtype=np.int16).tobytes() in group
    assert all(q.tobytes() in group for p in perms for q in p[perms])
    # Burnside coefficient calculation, retaining exact Python integers.
    total=[0]*145;cycles=Counter()
    for p in perms:
        seen=set();lengths=[]
        for i in range(144):
            if i in seen:continue
            length=0;j=i
            while j not in seen:seen.add(j);length+=1;j=int(p[j])
            lengths.append(length)
        cycles[','.join(map(str,sorted(lengths)))]+=1
        coefficients=[1]+[0]*144
        for length in lengths:
            for k in range(144,length-1,-1):coefficients[k]+=coefficients[k-length]
        total=[x+y for x,y in zip(total,coefficients)]
    assert all(v%len(perms)==0 for v in total)
    counts=[v//len(perms) for v in total]
    old=json.loads(OLD.read_text())
    previous=next(c for c in old['cases'] if c['n_sites']==144)
    assert counts==previous['orbit_counts_by_k']
    pair_orbits={min(tuple(sorted((int(p[i]),int(p[j])))) for p in perms) for i,j in itertools.combinations(range(144),2)}
    assert len(pair_orbits)==counts[2]
    assert len({min(int(p[i]) for p in perms) for i in range(144)})==counts[1]==1
    assert counts[0]==counts[144]==1
    receipt=dict(spacegroup=d.international,n_sites=144,n_full_operations=len(d.rotations),
                 n_site_permutations=len(perms),symprec_A=1e-5,sensitivity=sensitivity,
                 max_species_mapping_residual_A=residual,rotations=d.rotations.tolist(),translations=d.translations.tolist(),
                 Li_site_permutations_source_to_destination=perms.tolist(),cycle_histogram=dict(cycles),
                 possible_orbit_counts_by_vacancies=counts,
                 checks=dict(identity=True,closure=True,independent_k0_k1_k2=True,exact_match_previous_Burnside=True),
                 scope='Species-preserving symmetries compatible with this fixed periodic supercell. Does not remove cross-cell tilings.')
    dump(HERE/'parent_symmetry.json',receipt)
    return perms,counts

def load_catalog():
    with MANIFEST.open() as f:rows=list(csv.DictReader(f))
    with SCORES.open() as f:score_rows=list(csv.DictReader(f))
    scores={r['candidate_relpath']:r for r in score_rows}
    assert len(scores)==len(score_rows)==len(rows)==24145
    assert set(scores)=={r['candidate_relpath'] for r in rows}
    masks=np.zeros((len(rows),144),dtype=np.uint8)
    for i,r in enumerate(rows):
        removed=json.loads(r['remove_local_li_indices'])
        assert len(removed)==len(set(removed))==int(r['n_li_remove'])
        assert all(isinstance(j,int) and 0<=j<144 for j in removed)
        assert 144-len(removed)==int(r['n_li_keep'])
        masks[i,removed]=1
        s=scores[r['candidate_relpath']]
        assert int(s['n_li_keep'])==int(r['n_li_keep'])
        assert np.isfinite(float(s['energy_ev']))
        r['cached_UMA_energy_eV']=s['energy_ev'];r['cached_UMA_status']=s['uma_status']
    assert len({np.packbits(m).tobytes() for m in masks})==len(rows)
    return rows,masks

def words(mask):
    packed=np.packbits(mask,axis=1,bitorder='big')
    return np.pad(packed,((0,0),(0,6))).copy().view('>u8').astype(np.uint64)

def canonicalize(masks,perms):
    raw=words(masks);canonical=raw.copy()
    identity=next(j for j,p in enumerate(perms) if np.array_equal(p,np.arange(144)))
    witness=np.full(len(masks),identity,dtype=int);stabilizer=np.zeros(len(masks),dtype=int)
    for j,p in enumerate(perms):
        tr=words(masks[:,np.argsort(p)])
        stabilizer+=np.all(tr==raw,axis=1)
        lower=(tr[:,0]<canonical[:,0])|((tr[:,0]==canonical[:,0])&((tr[:,1]<canonical[:,1])|((tr[:,1]==canonical[:,1])&(tr[:,2]<canonical[:,2]))))
        canonical[lower]=tr[lower];witness[lower]=j
    assert np.all(stabilizer>0)
    assert np.all(len(perms)%stabilizer==0)
    keys=[b.tobytes()[:18].hex() for b in canonical.astype('>u8').view(np.uint8).reshape(-1,24)]
    # Independent Python set-of-vacancy-indices representation.
    rng=np.random.default_rng(20261105)
    chosen=set(rng.choice(len(masks),256,replace=False).tolist())
    chosen.update(np.flatnonzero(masks.sum(axis=1)<=1).tolist())
    independent=0
    for i in sorted(chosen):
        empty=np.flatnonzero(masks[i])
        images={sum(1<<(143-int(p[a])) for a in empty) for p in perms}
        assert format(min(images),'036x')==keys[i]
        assert len(images)*stabilizer[i]==len(perms)
        # The canonical mask itself stays canonical under a fresh group action.
        bits=np.array([(int(keys[i],16)>>(143-a))&1 for a in range(144)],dtype=np.uint8)
        values=[]
        for p in perms:
            v=np.empty(144,dtype=np.uint8);v[p]=bits
            values.append(int.from_bytes(np.packbits(v).tobytes(),'big'))
        assert min(values)==int(keys[i],16)
        independent+=1
    return keys,stabilizer,witness,independent

def worker_init():
    global WORKER
    a=read(PARENT,format='vasp');f=a.get_scaled_positions()%1
    WORKER=(a.cell.array,{z:(f[a.numbers==z],cKDTree(f[a.numbers==z],boxsize=1.)) for z in set(a.numbers)})

def verify_saved_batch(batch):
    cell,trees=WORKER;out=[]
    # Bound follows float64 roundoff at the cell scale. It certifies numerical
    # reproduction only, and is never used to merge distinct structures.
    bound=256*np.finfo(float).eps*np.linalg.norm(cell,axis=1).sum()
    for r in batch:
        path=SOURCE/'uma_omat_top500_candidates'/r['candidate_relpath']/'POSCAR'
        lines=path.read_text().splitlines();scale=float(lines[1])
        h=np.array([[float(v) for v in line.split()[:3]] for line in lines[2:5]])*scale
        symbols=lines[5].split();counts=list(map(int,lines[6].split()))
        index=7
        if lines[index].lower().startswith('s'):index+=1
        direct=lines[index].lower().startswith('d');index+=1
        coords=np.array([[float(v) for v in line.split()[:3]] for line in lines[index:index+sum(counts)]])
        if not direct:coords=coords*scale@np.linalg.inv(h)
        coords%=1.
        maximum=0.;start=0;occupied_li=None
        for symbol,count in zip(symbols,counts):
            z={'Li':3,'Co':27,'O':8}[symbol];parent,tree=trees[z]
            ff=coords[start:start+count];start+=count
            _,assigned=tree.query(ff)
            assert len(set(assigned))==count,(path,'nonbijective species match')
            delta=ff-parent[assigned];delta-=np.rint(delta)
            maximum=max(maximum,float(np.linalg.norm(delta@cell,axis=1).max()) if count else 0.)
            if z==3:occupied_li=set(map(int,assigned))
        expected=set(range(144))-set(json.loads(r['remove_local_li_indices']))
        assert occupied_li==expected,(path,'mask mismatch')
        cell_error=float(np.abs(h-cell).max())
        assert maximum<=bound and cell_error<=bound,(path,maximum,cell_error,bound)
        out.append(dict(candidate_relpath=r['candidate_relpath'],max_coordinate_difference_A=maximum,
                        max_cell_difference_A=cell_error,numerical_bound_A=bound,mask_matches=True))
    return out

def make_figures(summary):
    x=np.array([r['n_Li'] for r in summary]);possible=np.array([float(r['possible_symmetry_orbits']) for r in summary])
    raw=np.array([r['available_raw_entries'] for r in summary]);unique=np.array([r['available_unique_orbits'] for r in summary]);available=raw>0
    plt.rcParams.update({'font.size':10,'savefig.dpi':180})
    fig,axs=plt.subplots(3,1,figsize=(11,11),constrained_layout=True,gridspec_kw={'height_ratios':[1.35,1,1]})
    ax=axs[0]
    ax.plot(x,possible,color='black',lw=1.7,label='All possible symmetry-distinct vacancy masks')
    ax.scatter(x[available],unique[available],s=22,color='#0072b2',label='Available symmetry-distinct configurations',zorder=3)
    ax.axvspan(0,71.5,color='.93');ax.text(7,1e30,'No catalog entries\nbelow 50% lithiation',color='.35')
    ax.set(yscale='log',xlim=(0,144),ylim=(.7,2e40),xlabel='Retained Li atoms',ylabel='Number of symmetry orbits',title='144-site LCO parent: theoretical space and observed coverage')
    ax.secondary_xaxis('top',functions=(lambda n:n/144*100,lambda p:p/100*144)).set_xlabel('Lithiation (%)')
    ax.legend(loc='upper right',fontsize=9);ax.grid(alpha=.2)
    ax=axs[1];sel=x>=72
    ax.bar(x[sel],raw[sel],width=.8,color='.78',label='Available raw entries')
    ax.scatter(x[available],unique[available],s=21,color='#0072b2',label='Available unique orbits',zorder=3)
    missing=sel&~available;ax.scatter(x[missing],np.zeros(missing.sum()),marker='x',s=20,color='#d55e00',label='Unsampled integer composition',zorder=4)
    ax.set(xlim=(71,145),ylim=(-20,550),xlabel='Retained Li atoms',ylabel='Available configurations',title='Catalog detail: raw entries versus symmetry-unique configurations')
    ax.legend(loc='lower left',fontsize=8);ax.grid(alpha=.2)
    ax=axs[2];coverage=unique[available]/possible[available]
    ax.scatter(x[available],coverage,s=22,color='#009e73')
    ax.axhline(1,color='black',ls=':',lw=1)
    ax.set(yscale='log',xlim=(71,145),ylim=(1e-39,3),xlabel='Retained Li atoms',ylabel='Fraction of possible orbits represented',title='Configurational coverage; no claim of energetic or thermodynamic coverage')
    ax.grid(alpha=.2)
    fig.suptitle(f'{int(raw.sum()):,} raw candidates; {int(unique.sum()):,} symmetry-unique configurations',fontsize=15)
    for suffix in ('pdf','png'):fig.savefig(HERE/f'observed_orbit_coverage.{suffix}')
    plt.close(fig)

def main():
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use a compute allocation')
    start=time.monotonic();HERE.mkdir(exist_ok=True)
    inputs=[MANIFEST,SCORES,PARENT,SOURCE/'source_manifest.json',OLD]
    before={str(p):sha(p) for p in inputs}
    a=read(PARENT,format='vasp');assert Counter(a.get_chemical_symbols())=={'Li':144,'Co':144,'O':288}
    perms,possible=make_symmetry(a)
    rows,masks=load_catalog()
    keys,stabilizer,witness,n_independent=canonicalize(masks,perms)
    print('CANONICALIZED',len(rows),len(set(keys)),round(time.monotonic()-start,1),flush=True)
    verification=[];batches=[rows[i:i+256] for i in range(0,len(rows),256)]
    with ProcessPoolExecutor(max_workers=8,initializer=worker_init) as pool:
        for result in pool.map(verify_saved_batch,batches):
            verification.extend(result)
            if len(verification)%5000<len(result):print('VERIFIED_SAVED',len(verification),round(time.monotonic()-start,1),flush=True)
    assert len(verification)==len(rows)
    write_csv(HERE/'saved_geometry_verification.csv',verification)
    parent_id=sha(PARENT)[:16]
    for i,r in enumerate(rows):
        r.update(canonical_vacancy_mask_hex=keys[i],mask_orbit_id=parent_id+':'+keys[i],
                 orbit_size=len(perms)//int(stabilizer[i]),stabilizer_size=int(stabilizer[i]),
                 canonicalizing_operation_index=int(witness[i]))
    groups=defaultdict(list)
    for i,key in enumerate(keys):groups[key].append(i)
    orbit_rows=[];representatives=[]
    for key,ix in sorted(groups.items()):
        ix=sorted(ix,key=lambda i:rows[i]['candidate_relpath']);rep=rows[ix[0]]
        ee=np.array([float(rows[i]['cached_UMA_energy_eV']) for i in ix])
        for i in ix:
            rows[i]['representative_candidate_relpath']=rep['candidate_relpath'];rows[i]['available_orbit_members']=len(ix)
            rows[i]['is_representative']=i==ix[0]
        representatives.append(dict(rows[ix[0]]))
        orbit_rows.append(dict(mask_orbit_id=rep['mask_orbit_id'],n_Li=int(rep['n_li_keep']),n_vacancies=int(rep['n_li_remove']),
                              available_members=len(ix),full_orbit_size=rep['orbit_size'],stabilizer_size=rep['stabilizer_size'],
                              representative_candidate_relpath=rep['candidate_relpath'],cached_energy_min_eV=float(ee.min()),
                              cached_energy_max_eV=float(ee.max()),cached_energy_mean_eV=float(ee.mean()),cached_energy_span_eV=float(np.ptp(ee))))
    summary=[]
    for nli in range(145):
        rr=[r for r in rows if int(r['n_li_keep'])==nli];gg=[r for r in orbit_rows if r['n_Li']==nli]
        summary.append(dict(n_Li=nli,n_vacancies=144-nli,lithiation_fraction=nli/144,lithiation_percent=100*nli/144,
                            possible_raw_masks=math.comb(144,nli),possible_symmetry_orbits=possible[144-nli],
                            available_raw_entries=len(rr),available_unique_raw_masks=len(rr),available_unique_orbits=len(gg),
                            redundant_symmetry_entries=len(rr)-len(gg),
                            orbit_coverage_fraction=len(gg)/possible[144-nli],
                            maximum_within_orbit_cached_energy_span_eV=max((r['cached_energy_span_eV'] for r in gg),default=0.)))
    write_csv(HERE/'candidate_orbit_mapping.csv',rows)
    write_csv(HERE/'unique_representatives.csv',sorted(representatives,key=lambda r:(int(r['n_li_keep']),r['candidate_relpath'])))
    write_csv(HERE/'orbit_groups.csv',orbit_rows)
    write_csv(HERE/'coverage_by_lithiation.csv',summary)
    make_figures(summary)
    assert next(r for r in summary if r['n_Li']==143)['available_unique_orbits']==1
    assert next(r for r in summary if r['n_Li']==144)['available_unique_orbits']==1
    assert all(sha(p)==before[str(p)] for p in inputs),'Input changed during analysis'
    completion=dict(status='COMPLETE',raw_entries=len(rows),unique_raw_masks=len(rows),unique_symmetry_orbits=len(groups),
                    redundant_symmetry_entries=len(rows)-len(groups),n_sampled_compositions=sum(r['available_raw_entries']>0 for r in summary),
                    n_saved_POSCARs_verified=len(verification),maximum_geometry_difference_A=max(r['max_coordinate_difference_A'] for r in verification),
                    maximum_cell_difference_A=max(r['max_cell_difference_A'] for r in verification),
                    n_independent_canonicalization_checks=n_independent,n_group_operations=len(perms),
                    cached_UMA_status_counts=dict(Counter(r['cached_UMA_status'] for r in rows)),
                    maximum_within_orbit_cached_energy_span_eV=max(r['cached_energy_span_eV'] for r in orbit_rows),
                    input_sha256=before,script_sha256=sha(__file__),job_id=os.environ['SLURM_JOB_ID'],
                    software=dict(numpy=np.__version__,scipy=scipy.__version__,spglib=spglib.__version__),
                    elapsed_seconds=time.monotonic()-start,
                    representative_policy='Lexicographically first actual candidate_relpath per orbit; not lowest cached energy. Original labels and files preserved.')
    dump(HERE/'completion.json',completion)
    print('COMPLETE',len(groups),'unique;',len(rows)-len(groups),'redundant;',round(time.monotonic()-start,1),'seconds',flush=True)

if __name__=='__main__':main()
