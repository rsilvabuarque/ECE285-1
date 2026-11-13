"""Map exact archived geometries to phase-specific vacancy/displacement templates.

No relaxation, relabeling, or mask-only deduplication. Coordinates remain in the
original Cartesian frame. The half-site-spacing check is a sufficient geometric
localization certificate, not a physical validity or DFT convergence criterion.
"""
from __future__ import annotations
import os
for _name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
    os.environ[_name] = '1'
import argparse
import csv
import hashlib
import itertools
import json
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import time
import numpy as np
import scipy
from scipy.optimize import linear_sum_assignment
import spglib
from ase import Atoms
from ase.io import read, write

HERE = Path(__file__).resolve().parent
AUDIT = HERE.parent
CLASS = AUDIT.parents[2]
SOURCE = Path('/global/cfs/cdirs/m5056/rsb/2026/cathode_relithiation/cathode_relithiation_project')
HIST = SOURCE / 'results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/md_runs/engine_uma/T_393K/lith_100.00pct/base_00000000.extxyz'
SHIFTS = np.array(list(itertools.product((-1, 0, 1), repeat=3)))
DATA = PARENTS = None

def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')

def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()

def csv_write(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)

def frame_atoms(data, i):
    s = slice(data['offsets'][i], data['offsets'][i+1])
    return Atoms(numbers=data['numbers'][s], positions=data['positions'][s], cell=data['cell'][i], pbc=True)

def periodic_vectors(delta, cell):
    """Closest image among 27 neighbors of component-wrapped fractional deltas.

    Independently checked against ASE's general MIC for the selected templates
    and representative stored frames; no assumption of orthogonal coordinates.
    """
    delta = delta - np.rint(delta)
    cart = delta @ cell
    candidates = cart[..., None, :] + SHIFTS @ cell
    norms = np.einsum('...ij,...ij->...i', candidates, candidates)
    idx = np.argmin(norms, axis=-1)
    return np.take_along_axis(candidates, idx[..., None, None], axis=-2)[..., 0, :]

def distance_matrix(a, b, cell):
    v = periodic_vectors(a[:, None, :] - b[None, :, :], cell)
    return np.linalg.norm(v, axis=-1)

def symmetry(parent, tolerance):
    f, h, n = parent.get_scaled_positions(), parent.cell.array, parent.numbers
    d = spglib.get_symmetry_dataset((h, f, n), symprec=tolerance)
    li = np.flatnonzero(n == 3)
    permutations = []
    residual = 0.
    for r, t in zip(d.rotations, d.translations):
        cost = distance_matrix((f[li] @ r.T + t) % 1, f[li], h)
        p = cost.argmin(axis=1)
        if len(set(p)) != len(li): raise ValueError('Nonbijective Li symmetry')
        permutations.append(tuple(map(int, p)))
        for z in set(n):
            ff = f[n == z]
            cc = distance_matrix((ff @ r.T + t) % 1, ff, h)
            pp = cc.argmin(axis=1)
            if len(set(pp)) != len(ff): raise ValueError('Nonbijective host symmetry')
            residual = max(residual, float(cc[np.arange(len(pp)), pp].max()))
    permutations = sorted(set(permutations))
    # Closure and identity of induced removable-site permutations.
    pa = np.array(permutations)
    group = {tuple(p) for p in permutations}
    assert tuple(range(len(li))) in group
    assert all(tuple(q) in group for p in pa for q in p[pa])
    return dict(spacegroup=d.international, symprec_A=tolerance,
                rotations=d.rotations.tolist(), translations=d.translations.tolist(),
                li_permutations=[list(p) for p in permutations],
                n_full_operations=len(d.rotations), n_li_permutations=len(permutations),
                max_species_symmetry_residual_A=residual)

def make_parents(data):
    parents = {}
    a0 = read(HIST, index=0, format='extxyz'); a0.calc = None
    raw = [('Historical_O3_24', a0, 1e-5, 'Exact historical pristine frame 0', None),
           ('Hu_O3_12', frame_atoms(data, 4687), .001, 'Exact optimized Hu Li12 frame; matches earlier symmetry_counts.json', 4687)]
    # Complete the O1 alkali lattice; its archived Li3 seed is not a full parent.
    seed = frame_atoms(data, 6633)
    ff, hh, nn = seed.get_scaled_positions(), seed.cell.array, seed.numbers
    a = (np.linalg.norm(hh[0]) + np.linalg.norm(hh[1])) / 2
    c = np.linalg.det(hh) / (a*a*np.sqrt(3)/2)
    h = np.array([[a,0,0],[-a/2,a*np.sqrt(3)/2,0],[0,0,c]])
    co = np.round(ff[nn == 27] * [2,2,3]) / [2,2,3]
    ox = ff[nn == 8].copy()
    ox[:, :2] = np.round(ox[:, :2]*6)/6
    dz = ff[nn == 8, 2] - np.round(ff[nn == 8, 2]*3)/3
    oxygen_height = float(np.mean(np.abs(dz)))
    ox[:, 2] = np.round(ff[nn == 8, 2]*3)/3 + np.sign(dz)*oxygen_height
    li = np.array([(x,y,(m+.5)/3) for m in range(3) for x in (0,.5) for y in (0,.5)])
    o1 = Atoms(numbers=[3]*12+[27]*12+[8]*24, scaled_positions=np.vstack([li,co,ox])%1, cell=h, pbc=True)
    raw.append(('Hu_O1_completed_12', o1, 1e-5,
                'Derived ideal O1 template: completed 12 alkali sites, rational Co/O in-plane registry, fitted paired oxygen height; NOT a DFT-relaxed Li12 O1 structure', 6633))
    # Ideal Fd-3m spinel retaining a fitted oxygen internal parameter.
    seed = frame_atoms(data, 6039)
    ff, hh, nn = seed.get_scaled_positions(), seed.cell.array, seed.numbers
    fs = ff.copy()
    fs[nn == 3] = np.round(ff[nn == 3]*4)/4
    fs[nn == 27] = np.round(ff[nn == 27]*8)/8
    anchor = np.round(ff[nn == 8]*8)/8
    delta = ff[nn == 8] - anchor
    delta -= np.rint(delta)
    oxygen_offset = float(np.mean(np.abs(delta)))
    fs[nn == 8] = anchor + np.sign(delta)*oxygen_offset
    spinel = Atoms(numbers=nn, scaled_positions=fs%1, cell=np.eye(3)*np.cbrt(np.linalg.det(hh)), pbc=True)
    raw.append(('Hu_spinel_8', spinel, 1e-5,
                'Derived ideal Fd-3m Li8Co16O32 template: Li quarter-grid, Co eighth-grid, fitted oxygen internal offset; NOT a newly calculated geometry', 6039))
    for name, atoms, tol, provenance, seed_index in raw:
        sym = symmetry(atoms, tol)
        frac = atoms.get_scaled_positions()
        # Cell holohedry supplies orientation candidates, not extra mask symmetries.
        lattice = atoms.cell.array.copy()
        if name in ('Hu_O3_12', 'Hu_O1_completed_12'):
            aa = (np.linalg.norm(lattice[0])+np.linalg.norm(lattice[1]))/2
            cc = np.linalg.det(lattice)/(aa*aa*np.sqrt(3)/2)
            lattice = np.array([[aa,0,0],[-aa/2,aa*np.sqrt(3)/2,0],[0,0,cc]])
        hol = spglib.get_symmetry((lattice, [[0,0,0]], [1]), symprec=.001)['rotations']
        # Identity first, deterministic lexicographic order thereafter.
        ident = np.eye(3, dtype=int)
        qs = [ident] + [q for q in hol if not np.array_equal(q, ident)]
        p = dict(id=name, numbers=atoms.numbers.tolist(), fractional_sites=frac.tolist(),
                 cell_A=atoms.cell.array.tolist(), provenance=provenance, seed_unique_index=seed_index,
                 **sym, orientation_candidates=[q.tolist() for q in qs])
        if name == 'Hu_O1_completed_12': p['fitted_oxygen_height_fractional_c'] = oxygen_height
        if name == 'Hu_spinel_8': p['fitted_oxygen_offset_fractional'] = oxygen_offset
        p['n_removable_sites'] = int(sum(atoms.numbers == 3))
        parents[name] = p
        write(HERE/(name+'.extxyz'), atoms, format='extxyz')
    return parents

def fit_host(f, n, cell, p, q, max_iter=12, extra_starts=False):
    ref = np.asarray(p['fractional_sites']) @ q.T
    pn = np.asarray(p['numbers'])
    # The archived atom lists usually preserve host order; Hungarian assignment
    # is nevertheless applied at every iteration and does not rely on that order.
    aa, bb = f[n == 27], ref[pn == 27]
    delta = aa-bb
    t0 = np.angle(np.exp(2j*np.pi*delta).mean(axis=0))/(2*np.pi)
    best = None
    starts=[t0, np.zeros(3)]
    if extra_starts: starts.extend(aa[0]-bb)
    for t in starts:
        t = t.copy()
        for _ in range(max_iter):
            vectors = []
            for z in (27,8):
                ai, bi = np.flatnonzero(n == z), np.flatnonzero(pn == z)
                cost = distance_matrix(f[ai], ref[bi]+t, cell)
                ii, jj = linear_sum_assignment(cost**2)
                vectors.append(periodic_vectors(f[ai[ii]]-ref[bi[jj]]-t, cell))
            vv = np.vstack(vectors)
            update = vv.mean(axis=0) @ np.linalg.inv(cell)
            t += update
            if np.linalg.norm(update@cell) < 1e-9: break
        # Re-evaluate at the final translation.
        ss = 0.
        for z in (27,8):
            cost = distance_matrix(f[n == z], ref[pn == z]+t, cell)
            ii,jj = linear_sum_assignment(cost**2)
            ss += float(np.sum(cost[ii,jj]**2))
        score = ss / np.sum(n != 3)
        if best is None or score < best[0]: best = (score,t.copy())
    return best

def assign_geometry(f,n,cell,p,q,t):
    pn=np.asarray(p['numbers']);ref=np.asarray(p['fractional_sites'])@q.T+t
    atom_slot=np.empty(len(n),dtype=np.int16)
    u=np.empty((len(n),3));ratios=np.empty(len(n));nearest_gaps=np.empty(len(n))
    for z in (3,27,8):
        ai,bi=np.flatnonzero(n==z),np.flatnonzero(pn==z)
        if not len(ai): continue
        cost=distance_matrix(f[ai],ref[bi],cell)
        ii,jj=linear_sum_assignment(cost**2)
        if len(ii)!=len(ai): raise ValueError('More atoms than parent slots')
        atom_slot[ai[ii]]=bi[jj]
        u[ai[ii]]=periodic_vectors(f[ai[ii]]-ref[bi[jj]],cell)
        refdist=distance_matrix(ref[bi],ref[bi],cell)
        np.fill_diagonal(refdist,np.inf)
        spacing=refdist.min(axis=1)
        ratios[ai[ii]]=2*cost[ii,jj]/spacing[jj]
        sorted_cost=np.sort(cost,axis=1)
        nearest_gaps[ai[ii]]=sorted_cost[:,1]-sorted_cost[:,0]
    return ref,atom_slot,u,ratios,nearest_gaps

def initialize():
    global DATA, PARENTS
    with np.load(AUDIT/'reference_frames.npz') as z: DATA = {k:z[k] for k in z.files}
    PARENTS = json.loads((HERE/'parents.json').read_text())

def map_source(task):
    parent_id, rows = task
    p = PARENTS[parent_id]
    pf, pn = np.asarray(p['fractional_sites']), np.asarray(p['numbers'])
    li_slots = np.flatnonzero(pn == 3)
    qcache = None
    canonical_cache = {}
    output = []
    for row in rows:
        i = int(row['index']); s = slice(DATA['offsets'][i],DATA['offsets'][i+1])
        n, positions, cell = DATA['numbers'][s],DATA['positions'][s],DATA['cell'][i]
        f = positions @ np.linalg.inv(cell)
        qs = [np.asarray(q) for q in p['orientation_candidates']] if qcache is None else [qcache]
        best = None
        for q in qs:
            score,t = fit_host(f,n,cell,p,q)
            if best is None or score < best[0]: best = (score,t,q)
        score,t,q = best
        assigned=assign_geometry(f,n,cell,p,q,t)
        retried=False
        # Revisit all orientations and independent Co-anchor origins whenever
        # the actual localization certificate fails; no arbitrary RMS gate.
        if assigned[3].max()>=1 and row['phase']!='migration':
            retried=True
            best_local=None
            for candidate in p['orientation_candidates']:
                qq = np.asarray(candidate)
                sc,tt = fit_host(f,n,cell,p,qq,extra_starts=True)
                aa=assign_geometry(f,n,cell,p,qq,tt)
                if sc < best[0]: best=(sc,tt,qq)
                if aa[3].max()<1 and (best_local is None or sc<best_local[0]):
                    best_local=(sc,tt,qq)
            if best_local is not None: best=best_local
            score,t,q=best
            assigned=assign_geometry(f,n,cell,p,q,t)
        qcache=q
        ref,atom_slot,u,ratios,nearest_gaps=assigned
        mask=np.ones(len(li_slots),dtype=np.uint8)
        occupied=set(atom_slot[n==3])
        for j,slot in enumerate(li_slots):
            if slot in occupied: mask[j]=0
        bit=''.join(map(str,mask))
        if bit not in canonical_cache:
            transformed=[]
            for perm in p['li_permutations']:
                m=np.empty_like(mask);m[np.asarray(perm)]=mask
                transformed.append(''.join(map(str,m)))
            canonical_cache[bit]=(min(transformed),len(set(transformed)))
        canonical,orbit_size=canonical_cache[bit]
        reconstructed=ref[atom_slot]@cell+u
        diff=periodic_vectors((positions-reconstructed)@np.linalg.inv(cell),cell)
        error=float(np.linalg.norm(diff,axis=1).max())
        if error>1e-10: raise ValueError('Coordinate roundtrip failed')
        host_ratio=float(ratios[n!=3].max())
        li_ratio=float(ratios[n==3].max()) if np.any(n==3) else 0.
        localized=bool(max(host_ratio,li_ratio)<1.)
        migration=row['phase']=='migration'
        status='migration_geometry_only' if migration else 'localized_joint' if localized else 'mapping_review'
        out={**row,'parent_id':parent_id,'n_parent_Li_sites':len(li_slots),
             'retained_site_fraction_y':int(row['n_Li'])/len(li_slots),
             'vacancy_mask':bit,'canonical_vacancy_mask':canonical,
             'mask_orbit_id':parent_id+':'+canonical,'mask_orbit_size':orbit_size,
             'mask_stabilizer_size':len(p['li_permutations'])//orbit_size,
             'host_displacement_RMS_A':float(np.sqrt(np.mean(np.sum(u[n!=3]**2,axis=1)))),
             'host_displacement_max_A':float(np.linalg.norm(u[n!=3],axis=1).max()),
             'Li_displacement_max_A':float(np.linalg.norm(u[n==3],axis=1).max()) if np.any(n==3) else 0.,
             'host_half_spacing_ratio_max':host_ratio,'Li_half_spacing_ratio_max':li_ratio,
             'assignment_margin_min_A':float(nearest_gaps.min()),'roundtrip_max_A':error,
             'mapping_status':status,'joint_template_eligible':localized and not migration,
             'orientation_origin_retry':retried,
             'orientation_matrix':json.dumps(q.tolist(),separators=(',',':')),
             'origin_shift_fractional':json.dumps(t.tolist(),separators=(',',':'))}
        output.append((i,out,atom_slot,u,q,t))
    return output

def duplicate_check():
    with (AUDIT/'raw_frame_map.csv').open() as f: raw=list(csv.DictReader(f))
    groups=defaultdict(list)
    for r in raw: groups[r['unique_index']].append(r)
    pairs=defaultdict(list)
    for entries in groups.values():
        if len(entries)>1:
            assert len(entries)==2
            a,b=entries
            pairs[(a['source'],b['source'])].append((int(a['frame']),int(b['frame'])))
    result=[]
    for paths,ix in pairs.items():
        ia,ib=np.asarray(ix).T
        for name in ['coord','box','energy','force','virial']:
            aa=np.load(Path(paths[0])/(name+'.npy'),mmap_mode='r')
            bb=np.load(Path(paths[1])/(name+'.npy'),mmap_mode='r')
            a,b=aa[ia],bb[ib]
            equal=np.array_equal(a,b)
            result.append(dict(source_a=paths[0],source_b=paths[1],n_pairs=len(ix),quantity=name,
                               dtype=str(aa.dtype),exactly_equal=equal,max_absolute_difference=float(np.max(np.abs(a.astype(float)-b.astype(float))))))
            if not equal: raise ValueError('Archived duplicate label conflict')
    dump(HERE/'duplicate_verification.json',dict(status='PASS',n_duplicate_pairs=sum(len(x) for x in pairs.values()),checks=result))

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--pilot',action='store_true');parser.add_argument('--workers',type=int,default=8)
    args=parser.parse_args()
    if not os.environ.get('SLURM_JOB_ID'): raise RuntimeError('Run mapping in a compute allocation')
    start=time.monotonic()
    with np.load(AUDIT/'reference_frames.npz') as z: data={k:z[k] for k in z.files}
    parents=make_parents(data);dump(HERE/'parents.json',parents)
    duplicate_check()
    with (AUDIT/'frames.csv').open() as f: rows=list(csv.DictReader(f))
    def parent_for(r):
        return 'Historical_O3_24' if r['dataset']=='historical' else 'Hu_O1_completed_12' if r['phase']=='O1' else 'Hu_spinel_8' if r['phase']=='spinel' else 'Hu_O3_12'
    tasks=defaultdict(list)
    for r in rows: tasks[(parent_for(r),r['source'])].append(r)
    tasklist=[(key[0],rr) for key,rr in tasks.items()]
    if args.pilot:
        chosen=[]
        for parent in parents:
            matching=[(p,rr) for p,rr in tasklist if p==parent]
            for p,rr in matching[:3]+matching[-3:]: chosen.append((p,rr[:2]))
        tasklist=chosen
    output=[]
    with ProcessPoolExecutor(max_workers=args.workers,initializer=initialize) as pool:
        for batch in pool.map(map_source,tasklist):
            output.extend(batch)
            if len(output)%1000<len(batch): print('MAPPED',len(output),'seconds',round(time.monotonic()-start,1),flush=True)
    output.sort(key=lambda x:x[0])
    if args.pilot:
        csv_write(HERE/'pilot_mapping.csv',[x[1] for x in output])
        print('PILOT',len(output),dict(Counter(x[1]['mapping_status'] for x in output)),round(time.monotonic()-start,1),flush=True)
        return
    assert [x[0] for x in output]==list(range(len(rows)))
    mapped=[x[1] for x in output]
    csv_write(HERE/'frame_mapping.csv',mapped)
    csv_write(HERE/'joint_template_dataset.csv',[r for r in mapped if r['joint_template_eligible']])
    csv_write(HERE/'continuous_or_review_dataset.csv',[r for r in mapped if not r['joint_template_eligible']])
    slots=np.concatenate([x[2] for x in output]);uu=np.concatenate([x[3] for x in output])
    np.savez_compressed(HERE/'site_displacements.npz',atom_parent_slot=slots,displacement_A=uu,
                        orientation=np.stack([x[4] for x in output]),origin_shift=np.stack([x[5] for x in output]),offsets=data['offsets'])
    orbitrows=[]
    groups=defaultdict(list)
    for r in mapped: groups[r['mask_orbit_id']].append(r)
    for key,rr in sorted(groups.items()):
        orbitrows.append(dict(mask_orbit_id=key,parent_id=rr[0]['parent_id'],n_Li=rr[0]['n_Li'],
                              vacancy_count=rr[0]['canonical_vacancy_mask'].count('1'),n_geometries=len(rr),
                              n_joint_eligible=sum(r['joint_template_eligible'] for r in rr),
                              n_sources=len(set(r['source'] for r in rr)),orbit_size=rr[0]['mask_orbit_size'],
                              min_energy_eV=float(min(data['energy'][int(r['index'])] for r in rr)),
                              max_energy_eV=float(max(data['energy'][int(r['index'])] for r in rr))))
    csv_write(HERE/'observed_mask_orbits.csv',orbitrows)
    summary=[]
    for key,rr in itertools.groupby(sorted(mapped,key=lambda r:(r['dataset'],r['phase'],int(r['n_Li']))),key=lambda r:(r['dataset'],r['phase'],int(r['n_Li']))):
        rr=list(rr);ok=[r for r in rr if r['joint_template_eligible']]
        summary.append(dict(dataset=key[0],phase=key[1],n_Li=key[2],n_geometries=len(rr),n_joint_eligible=len(ok),
                            n_assigned_mask_orbits=len(set(r['mask_orbit_id'] for r in rr)),
                            n_localized_mask_orbits=len(set(r['mask_orbit_id'] for r in ok)),
                            max_roundtrip_A=max(r['roundtrip_max_A'] for r in rr)))
    csv_write(HERE/'mapping_summary.csv',summary)
    dump(HERE/'completion.json',dict(status='MAPPING_COMPLETE',n_unique_geometries=len(mapped),
         statuses=dict(Counter(r['mapping_status'] for r in mapped)),
         max_coordinate_roundtrip_A=max(r['roundtrip_max_A'] for r in mapped),
         reference_sha256=file_sha(AUDIT/'reference_frames.npz'),script_sha256=file_sha(__file__),
         job_id=os.environ['SLURM_JOB_ID'],numpy=np.__version__,scipy=scipy.__version__,spglib=spglib.__version__,
         elapsed_seconds=time.monotonic()-start,
         criterion='All atom displacements strictly below half nearest same-species parent-site spacing in supplied cell. Sufficient localization certificate, not a physical validity threshold.',
         caveat='Every unique exact geometry and original label is retained. Equal masks are grouped, never deduplicated. Migration uses O3 only as a diagnostic reference and remains geometry-only. No training splits assigned.'))
    print('COMPLETE',dict(Counter(r['mapping_status'] for r in mapped)),round(time.monotonic()-start,1),flush=True)

if __name__=='__main__':main()
