"""Energy-free, symmetry-aware distance audit with explicit portable inputs.

All proposal attempts remain observations. Nearest distance to a deterministic
1024-orbit TRAIN subset is an UPPER bound on nearest distance to all TRAIN.
Mutation's raw exact-k edit size is distinct from its symmetry-minimum size.
"""
import argparse,csv,hashlib,json,os,time,traceback
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import numpy as np
from cathode_cvae.data import validate_permutation_group
from cathode_cvae.runtime import require_compute_allocation

REFERENCE_NAMESPACE='overnight-distance-reference-20261108-v1'
BYTE_COUNTS=np.asarray([i.bit_count() for i in range(256)],np.uint8)
def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b)
    return h.hexdigest()
def read(path):
    with Path(path).open() as f:return list(csv.DictReader(f))
def write(path,rows):
    with Path(path).open('x',newline='') as f:
        if rows:
            w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)));w.writeheader();w.writerows(rows)
def save(path,value):
    with Path(path).open('x') as f:json.dump(value,f,indent=2,allow_nan=False);f.write('\n')
def pack(x):
    x=np.asarray(x)
    if x.ndim!=2 or not 1<=x.shape[1]<=62 or not np.isin(x,[0,1]).all():raise ValueError('Packing requires 1..62 binary mobile sites')
    padded=np.zeros((len(x),8),np.uint8);bits=np.packbits(x.astype(np.uint8),axis=1,bitorder='big');padded[:,:bits.shape[1]]=bits
    return padded.view('>u8').ravel().astype(np.uint64)
def popcount(x):
    x=np.asarray(x,dtype=np.uint64)
    if hasattr(np,'bitwise_count'):return np.bitwise_count(x)
    return BYTE_COUNTS[np.ascontiguousarray(x).view(np.uint8).reshape(x.shape+(8,))].sum(-1,dtype=np.uint8)
def orbit_codes(x,inverse):return np.stack([pack(x[:,p]) for p in inverse],axis=1)
def canonical_codes(x,inverse):
    best=np.full(len(x),np.iinfo(np.uint64).max,np.uint64)
    for p in inverse:np.minimum(best,pack(1-x[:,p]),out=best)
    return best
def code_keys(codes,sites):return [format(int(c)>>(64-sites),f'0{sites}b') for c in codes]
def bounded_nearest(query,reference,inverse,*,deadline,batch=32):
    images=np.unique(orbit_codes(reference,inverse));codes=pack(query);out=np.empty(len(query),np.uint8)
    for first in range(0,len(query),batch):
        if time.monotonic()>deadline:raise TimeoutError('Declared CPU wall-time bound reached; partial evidence is preserved')
        out[first:first+batch]=popcount(codes[first:first+batch,None]^images[None]).min(1)
    return out,len(images)
def load_identities(path):
    """Do not open energy, force, stress, or other outcome members."""
    with np.load(path,allow_pickle=False) as z:
        x=z['occupancy'];roles=z['split'].astype(str);ids=z['structure_ids'].astype(str)
        counts=z['n_target'] if 'n_target' in z else z['n_Li']
    if x.ndim!=2 or len(roles)!=len(x) or len(ids)!=len(x) or len(counts)!=len(x):raise ValueError('Identity arrays are not aligned')
    if len(set(ids))!=len(ids) or not np.isin(x,[0,1]).all() or not np.array_equal(x.sum(1),counts):raise ValueError('Invalid training identity/count data')
    return dict(occupancy=x,split=roles,structure_ids=ids,counts=counts)
def target(row):return int(row.get('n_target') or row['n_Li'])
def summaries(rows):
    groups=defaultdict(list)
    for r in rows:groups[r['method'],r['replicate_id'],r['n_target'],r['requested_bin'],r['diagnostic']].append(r)
    cells=[]
    for key,rr in sorted(groups.items()):
        out=dict(zip(['method','replicate_id','n_target','requested_bin','diagnostic'],key));d=[r['nearest_TRAIN_reference_orbit_Hamming'] for r in rr if r['nearest_TRAIN_reference_orbit_Hamming']!='']
        out.update(attempts=len(rr),valid_attempts=sum(r['exact_valid'] for r in rr),distance_known=len(d),
            mean_nearest_TRAIN_reference_orbit_Hamming=float(np.mean(d)) if d else '',
            median_nearest_TRAIN_reference_orbit_Hamming=float(np.median(d)) if d else '',
            zero_reference_distance_attempts=sum(q==0 for q in d))
        for field in ['raw_seed_child_Hamming','symmetry_minimum_seed_child_Hamming']:
            values=[r[field] for r in rr if r[field]!=''];out['mean_'+field]=float(np.mean(values)) if values else ''
        cells.append(out)
    byfit=defaultdict(list)
    for r in cells:byfit[r['method'],r['replicate_id'],r['diagnostic']].append(r)
    fits=[]
    for (method,rep,diagnostic),rr in sorted(byfit.items()):
        r=dict(method=method,replicate_id=rep,diagnostic=diagnostic,condition_cells=len(rr),attempts=sum(q['attempts'] for q in rr))
        for field in ['mean_nearest_TRAIN_reference_orbit_Hamming','mean_raw_seed_child_Hamming','mean_symmetry_minimum_seed_child_Hamming']:
            values=[q[field] for q in rr if q[field]!=''];r[field]=float(np.mean(values)) if values else ''
        fits.append(r)
    return cells,fits

def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for name in ['generated','parent','training-data','output-dir']:p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--seed-source-data',type=Path,action='append',default=[],help='Additional explicitly bound TRAIN seed datasets, if mutation uses another stage')
    p.add_argument('--include-diagnostics',action='store_true');p.add_argument('--max-seconds',type=float,default=300)
    a=p.parse_args(argv);require_compute_allocation()
    if not 10<=a.max_seconds<=300:raise ValueError('Audit CPU wall bound must be 10..300 seconds')
    started=time.monotonic();deadline=started+a.max_seconds-5;a.output_dir.mkdir(parents=True,exist_ok=False);before={}
    try:
        gen=a.generated;receiptpath=gen/'receipt.json';protocolpath=gen/'protocol.json';receipt=json.loads(receiptpath.read_text());protocol=json.loads(protocolpath.read_text())
        if receipt.get('status')!='FROZEN_BEFORE_SCORING' or receipt.get('protocol_sha256')!=sha(protocolpath):raise ValueError('Unfrozen or modified proposal protocol')
        for name,digest in receipt['outputs'].items():
            if sha(gen/name)!=digest:raise ValueError('Generated input changed: '+name)
        for name,path in [('parent',a.parent),('data',a.training_data)]:
            if sha(path)!=protocol['inputs'][name]['sha256']:raise ValueError('Parent or TRAIN reference differs from generation binding')
        attempts=gen/('attempts_all.csv' if a.include_diagnostics else 'attempts.csv');rawpath=gen/'raw_attempts.npz'
        sources=[receiptpath,protocolpath,attempts,rawpath,a.parent,a.training_data,*a.seed_source_data];before={str(q.resolve()):sha(q) for q in sources}
        rows=read(attempts)
        if not rows:raise ValueError('No proposal attempts were supplied')
        with np.load(a.parent,allow_pickle=False) as z:perms=z['permutations']
        validate_permutation_group(perms);inverse=np.argsort(perms,axis=1);sites=perms.shape[1]
        with np.load(rawpath,allow_pickle=False) as z:raw=z['occupancy'];seeds=z['seed_occupancy']
        if raw.ndim!=2 or raw.shape[1]!=sites or seeds.shape!=raw.shape:raise ValueError('Raw parent dimensions differ')
        expected=receipt['total_attempts'] if a.include_diagnostics else receipt['primary_attempts']
        indices=np.asarray([int(r['raw_row']) for r in rows])
        if len(rows)!=expected or len(set(indices))!=len(indices) or not np.all((0<=indices)&(indices<len(raw))):raise ValueError('Attempt denominator or raw alignment changed')
        valid=np.asarray([bool(r['status']=='generated' and np.isin(raw[i],[0,1]).all() and raw[i].sum()==target(r)) for r,i in zip(rows,indices)])
        if any(r['status']=='generated' and not v for r,v in zip(rows,valid)):raise ValueError('A generated row has invalid occupancy/count')
        canonical=code_keys(canonical_codes(raw[indices[valid]],inverse),sites)
        if canonical!=[r['canonical_mask'] for r,v in zip(rows,valid) if v]:raise ValueError('Canonical keys differ from raw masks/full parent group')
        attempt_rows=[dict(attempt_id=r['attempt_id'],method=r['method'],replicate_id=r['replicate_id'],fit_seed=r.get('fit_seed',''),n_target=target(r),
            requested_bin=r['requested_bin'],diagnostic=r.get('diagnostic','primary'),raw_row=int(i),status=r['status'],canonical_mask=r['canonical_mask'],exact_valid=bool(v),
            nearest_TRAIN_reference_orbit_Hamming='',TRAIN_reference_orbits='',seed_structure_id=r.get('seed_structure_id',''),swap_count=r.get('swap_count',''),
            raw_seed_child_Hamming='',symmetry_minimum_seed_child_Hamming='') for r,i,v in zip(rows,indices,valid)]
        write(a.output_dir/'attempt_index.csv',attempt_rows)
        data=load_identities(a.training_data);train=np.flatnonzero(data['split']=='train')
        if data['occupancy'].shape[1]!=sites or not len(train):raise ValueError('No aligned TRAIN reference')
        trainkeys=code_keys(canonical_codes(data['occupancy'][train],inverse),sites);eligible=defaultdict(dict)
        for i,key in zip(train,trainkeys):
            n=int(data['counts'][i]);previous=eligible[n].get(key)
            if previous is None or (str(data['structure_ids'][i]),int(i))<(str(data['structure_ids'][previous]),int(previous)):eligible[n][key]=int(i)
        counts=sorted({target(r) for r in rows});references={};reference_rows=[]
        for n in counts:
            ranked=sorted(eligible[n],key=lambda key:hashlib.sha256(f'{REFERENCE_NAMESPACE}|{before[str(a.parent.resolve())]}|{n}|{key}'.encode()).digest())[:1024]
            if not ranked:raise ValueError('No TRAIN orbits for this requested composition')
            references[n]=1-np.asarray([[int(c) for c in key] for key in ranked],np.uint8)
            reference_rows.extend(dict(n_target=n,structure_id=str(data['structure_ids'][eligible[n][key]]),canonical_vacancy_key=key,source_row=eligible[n][key],source_role='train',rank=rank) for rank,key in enumerate(ranked))
        write(a.output_dir/'TRAIN_reference.csv',reference_rows)
        save(a.output_dir/'reference_protocol.json',dict(status='FROZEN_BEFORE_DISTANCE_COMPUTATION',namespace=REFERENCE_NAMESPACE,
            selection='Lowest SHA256(namespace|parent SHA|count|canonical vacancy key), unique TRAIN orbits only',requested_reference_size=1024,
            reference_orbits_per_count={n:len(x) for n,x in references.items()},full_TRAIN_unique_orbits_per_count={n:len(eligible[n]) for n in counts},
            TRAIN_reference_sha256=sha(a.output_dir/'TRAIN_reference.csv'),protected_training_data_sha256=sha(a.training_data),parent_sha256=sha(a.parent),energy_fields_read=0))
        unique={n:sorted({r['canonical_mask'] for r,v in zip(rows,valid) if v and target(r)==n}) for n in counts}
        lookup={};images={}
        with ThreadPoolExecutor(max_workers=min(3,len(counts))) as executor:
            futures={executor.submit(bounded_nearest,1-np.asarray([[int(c) for c in key] for key in keys],np.uint8).reshape(-1,sites),references[n],inverse,deadline=deadline):n for n,keys in unique.items()}
            for future in as_completed(futures):
                n=futures[future];distance,imagecount=future.result();lookup[n]=dict(zip(unique[n],map(int,distance)));images[n]=imagecount
                write(a.output_dir/f'distances_count_{n}.csv',[dict(n_target=n,canonical_mask=key,nearest_TRAIN_reference_orbit_Hamming=value) for key,value in lookup[n].items()])
                print('DISTANCES_COMPLETE',n,len(distance),imagecount,flush=True)
        seed_data={sha(a.training_data):data}
        for path in a.seed_source_data:seed_data[sha(path)]=load_identities(path)
        mutation_rows=[(int(i),r) for r,v,i in zip(rows,valid,indices) if v and r.get('seed_source_row')]
        mutation_index=np.asarray([i for i,r in mutation_rows],int)
        if time.monotonic()>deadline:raise TimeoutError('Declared wall-time bound reached before mutation verification')
        raw_mutation=np.count_nonzero(raw[mutation_index]!=seeds[mutation_index],axis=1)
        orbit_mutation=(popcount(pack(raw[mutation_index])[:,None]^orbit_codes(seeds[mutation_index],inverse)).min(1)
            if len(mutation_index) else np.asarray([],np.uint8))
        mutation_distance={int(i):(int(d),int(g)) for i,d,g in zip(mutation_index,raw_mutation,orbit_mutation)}
        checked=0
        for out,r,v,i in zip(attempt_rows,rows,valid,indices):
            n=target(r);out['TRAIN_reference_orbits']=len(references[n])
            if v:out['nearest_TRAIN_reference_orbit_Hamming']=lookup[n][r['canonical_mask']]
            if not v or not r.get('seed_source_row'):continue
            source=r.get('seed_source_sha256');original=seed_data.get(source)
            if original is None:raise ValueError('Mutation needs its explicit TRAIN seed-source file')
            j=int(r['seed_source_row']);k=int(r['swap_count']);seed=seeds[i];child=raw[i]
            if not 0<=j<len(original['split']) or original['split'][j]!='train' or original['structure_ids'][j]!=r['seed_structure_id'] or not np.array_equal(seed,original['occupancy'][j]):raise ValueError('Mutation seed lost its recorded TRAIN identity/orientation')
            rawdistance,minimum=mutation_distance[int(i)]
            removed=np.flatnonzero((seed==1)&(child==0));added=np.flatnonzero((seed==0)&(child==1))
            listed=lambda s:np.asarray([int(q) for q in s.split('|') if q!=''],int)
            if rawdistance!=2*k or len(removed)!=k or len(added)!=k or not np.array_equal(np.sort(listed(r['removed_sites'])),removed) or not np.array_equal(np.sort(listed(r['added_sites'])),added):raise ValueError('Mutation child does not implement recorded exact k original-seed edits')
            if minimum>rawdistance:raise AssertionError('Orbit minimum exceeds identity distance')
            out['raw_seed_child_Hamming']=rawdistance;out['symmetry_minimum_seed_child_Hamming']=minimum;checked+=1
        if time.monotonic()>deadline:raise TimeoutError('Declared wall-time bound reached after mutation verification')
        write(a.output_dir/'attempt_distances.csv',attempt_rows);cells,fits=summaries(attempt_rows)
        write(a.output_dir/'metrics_by_condition_seed.csv',cells);write(a.output_dir/'metrics_by_seed.csv',fits)
        (a.output_dir/'METHOD.md').write_text('For occupations a,b, d_G(a,b)=min_g sum_j |a_j-b_{g^-1(j)}| over ALL actual retained parent operations. This differs from Hamming distance between canonical representatives. For fixed TRAIN subset R, d_R(a)=min_b_in_R d_G(a,b)>=d_TRAIN(a). Thus a positive reported distance is only an upper bound on nearest full-TRAIN distance; it cannot prove full-TRAIN separation. References are 1024 deterministic SHA-ranked unique TRAIN orbits/count, or all if fewer. Hamming units are differing mobile-site occupations, not angstrom. For equal count, Hamming/2 is the minimum removal/insertion swap count under that alignment. Mutation raw 2k uses its original joint seed/child orientation; the orbit minimum can be smaller, even zero. Every attempt, repeat and invalid/failure row remains. Fit summaries average composition/bin cells equally; proposal rows are not independent model fits. No energy/force/stress members are opened and no novelty labels, model selection or weights are changed.\n')
        if before!={str(q.resolve()):sha(q) for q in sources}:raise ValueError('Protected input changed during audit')
        result=dict(status='PASS',attempts=len(rows),valid_attempts=int(valid.sum()),invalid_or_failed_attempts=int((~valid).sum()),all_attempts_preserved=True,
            unique_valid_orbits=sum(len(q) for q in unique.values()),actual_parent_operations=len(perms),sites=sites,reference_orbits_per_count={n:len(x) for n,x in references.items()},
            reference_images_per_count=images,reference_sha256=sha(a.output_dir/'TRAIN_reference.csv'),mutation_TRAIN_source_checks=checked,mutation_raw_distances_exact_2k=True,
            metric='Exact symmetry-aware minimum Hamming to deterministic TRAIN subset; UPPER bound on full-TRAIN nearest distance',energy_fields_read=0,
            input_hashes=before,inputs_unchanged=True,source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,maximum_seconds=a.max_seconds,
            job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),numpy=np.__version__,
            new_MLIP_calls=0,new_optimizer_updates=0,new_structures=0,selection_changed=False,
            output_hashes={q.name:sha(q) for q in a.output_dir.iterdir() if q.is_file()})
        result['elapsed_seconds']=time.monotonic()-started
        if result['elapsed_seconds']>a.max_seconds:raise TimeoutError('Declared wall-time bound reached while writing audit outputs')
        save(a.output_dir/'receipt.json',result);print(json.dumps(result,indent=2));return result
    except Exception as error:
        save(a.output_dir/'failure_receipt.json',dict(status='FAILED_PRESERVED',error_type=type(error).__name__,error=str(error),traceback=traceback.format_exc(),
            input_hashes=before,source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-started,new_MLIP_calls=0,new_optimizer_updates=0))
        raise
if __name__=='__main__':main()
