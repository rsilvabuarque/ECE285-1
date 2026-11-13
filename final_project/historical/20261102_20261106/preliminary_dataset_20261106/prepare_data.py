"""Generate canonical vacancy masks; geometry is reconstructed from explicit parent sites."""
import csv,hashlib,itertools,json,math,os,time
from pathlib import Path
import numpy as np
import spglib
from ase import Atoms
from ase.io import read,write
from scipy.optimize import linear_sum_assignment
HERE=Path(__file__).resolve().parent
def dump(path,data):Path(path).write_text(json.dumps(data,indent=2)+'\n')
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def symmetry(a,tol=1e-5):
    h=a.cell.array;f=a.get_scaled_positions();li=np.flatnonzero(a.numbers==3);lookup={int(k):i for i,k in enumerate(li)}
    sy=spglib.get_symmetry((h,f,a.numbers),symprec=tol);perms=[];residual=0.
    for r,t in zip(sy['rotations'],sy['translations']):
        transformed=(f@r.T+t)%1.;whole=np.zeros(len(a),int)
        for z in np.unique(a.numbers):
            ids=np.flatnonzero(a.numbers==z);d=transformed[ids,None]-f[None,ids];d-=np.rint(d)
            dist=np.linalg.norm(d@h,axis=-1);i,j=linear_sum_assignment(dist)
            residual=max(residual,float(dist[i,j].max()));whole[ids[i]]=ids[j]
        perms.append([lookup[int(whole[k])] for k in li])
    p=np.unique(perms,axis=0);keys={tuple(x) for x in p}
    assert tuple(range(len(li))) in keys and all(tuple(x[y]) in keys for x in p for y in p)
    return p,residual
def orbit_counts(perms):
    n=perms.shape[1];sums=[0]*(n+1)
    for p in perms:
        seen=set();lengths=[]
        for i in range(n):
            if i in seen:continue
            j=i;length=0
            while j not in seen:seen.add(j);length+=1;j=int(p[j])
            lengths.append(length)
        coeff=[0]*(n+1);coeff[0]=1
        for length in lengths:
            for k in range(n,length-1,-1):coeff[k]+=coeff[k-length]
        sums=[a+b for a,b in zip(sums,coeff)]
    assert all(x%len(perms)==0 for x in sums)
    return [x//len(perms) for x in sums]
def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    assert not (HERE/'data/mask_manifest.csv').exists()
    cfg=json.loads((HERE/'config.json').read_text());old=Path(cfg['fixed_pilot']);rng=np.random.default_rng(cfg['seed'])
    p48=read(old/'parent_POSCAR');oldsy=json.loads((old/'parent_symmetry.json').read_text())
    h=p48.cell.array/np.array([2,4,2])[:,None];f=np.mod(p48.get_scaled_positions()*[2,4,2],1.)
    seen=set();nums=[];fs=[]
    for z,s in zip(p48.numbers,f):
        s[np.isclose(s,1.,atol=1e-9,rtol=0)]=0.;key=(int(z),*np.round(s,9))
        if key not in seen:seen.add(key);nums.append(z);fs.append(s)
    base=Atoms(numbers=nums,scaled_positions=fs,cell=h,pbc=True);assert len(base)==12
    mapped_path=old.parent/'multimodel_20261104/structure_mapping/parents.json';mapped=json.loads(mapped_path.read_text())
    parents=[('O3_N48_pilot',p48,np.array(oldsy['permutations']),1e-5,oldsy['max_mapping_residual_A'],'Identical verified48-site pilot parent',512)]
    for name in ['Hu_O3_12','Historical_O3_24']:
        d=mapped[name];a=Atoms(numbers=d['numbers'],scaled_positions=d['fractional_sites'],cell=d['cell_A'],pbc=True)
        parents.append((name,a,np.array(d['li_permutations']),d['symprec_A'],d['max_species_symmetry_residual_A'],d['provenance'],128))
    for name,rep in [('O3_N36_compact',(3,4,1)),('O3_N48_compact',(4,4,1))]:
        path=HERE/'data/parents'/f'{name}.vasp';write(path,base.repeat(rep),format='vasp',sort=True,direct=True);a=read(path)
        perms,res=symmetry(a);parents.append((name,a,perms,1e-5,res,f'Canonical conventional repeat{rep}',128))
    previous={r['canonical_mask'] for r in csv.DictReader((old/'manifest.csv').open())};allrows=[];receipts=[]
    for name,a,perms,tol,res,provenance,quota in parents:
        n=int((a.numbers==3).sum());li=np.flatnonzero(a.numbers==3);inv=np.argsort(perms,axis=1)
        powers=np.left_shift(np.uint64(1),np.arange(n-1,-1,-1,dtype=np.uint64))
        def canonical(v):return int(np.min(v[inv].astype(np.uint64)@powers))
        capacities=orbit_counts(perms);masks=[];ids=[];acquisitions=[];start=time.time()
        for k,capacity in enumerate(capacities):
            target=min(quota,capacity);unique={};attempts=0
            if math.comb(n,k)<=30000:
                for comb in itertools.combinations(range(n),k):
                    v=np.zeros(n,np.uint8);v[list(comb)]=1;c=canonical(v)
                    if c not in unique:unique[c]=v.copy()
                assert len(unique)==capacity
                chosen=rng.choice(list(unique),size=target,replace=False) if len(unique)>target else list(unique)
                unique={int(c):unique[int(c)] for c in chosen};method='exhaustive_small_space'
            else:
                method='uniform_raw_fixed_count_with_orbit_rejection'
                while len(unique)<target:
                    attempts+=1;assert attempts<target*1000,(name,k,len(unique),target)
                    v=np.zeros(n,np.uint8);v[rng.choice(n,k,replace=False)]=1;c=canonical(v)
                    if c not in unique:unique[c]=v.copy()
            for c,v in sorted(unique.items()):
                sid=f'{name}_{c:0{(n+3)//4}x}';canonical_string=format(c,f'0{n}b')
                index=len(masks);masks.append(v);ids.append(sid);acquisitions.append(method)
                allrows.append(dict(structure_id=sid,parent=name,parent_index=index,n_sites=n,n_Li=n-k,x=(n-k)/n,n_atoms=len(a)-k,canonical_mask=canonical_string,raw_mask=''.join(map(str,v)),sampling=method,pilot_lineage_holdout=bool(name=='O3_N48_pilot' and canonical_string in previous),orbit_capacity=capacity))
        arr=np.array(masks,np.uint8)
        np.savez_compressed(HERE/'data/parents'/f'{name}.npz',masks=arr,structure_ids=np.array(ids),numbers=a.numbers,positions_A=a.positions,cell_A=a.cell.array,li_indices=li,permutations=perms,orbit_capacities=np.array(capacities,dtype=np.int64))
        write(HERE/'data/parents'/f'{name}.extxyz',a)
        rec=dict(parent=name,n_sites=n,n_structures=len(arr),permutations=len(perms),symprec_A=tol,max_species_symmetry_residual_A=res,provenance=provenance,orbit_capacity_by_vacancies=capacities,seconds=time.time()-start,sha256=sha(HERE/'data/parents'/f'{name}.npz'))
        receipts.append(rec);print('PARENT_PREPARED',name,len(arr),len(perms),flush=True)
        if name=='Hu_O3_12':assert sum(capacities)==87
    with (HERE/'data/mask_manifest.csv').open('w',newline='') as stream:
        w=csv.DictWriter(stream,fieldnames=list(allrows[0]));w.writeheader();w.writerows(allrows)
    dump(HERE/'data/preparation_receipt.json',dict(parents=receipts,total_structures=len(allrows),source_pilot=str(old),mapping_source=str(mapped_path),mapping_source_sha256=sha(mapped_path),mask_convention='1=vacancy; Li sites refer to saved parent order',scope='Parent lattice orbits; relaxed endpoints are not asserted symmetry-distinct or equilibrium phases'))
    tasks=[]
    for rec in receipts:
        for model in cfg['models']:
            for start in range(0,rec['n_structures'],128):
                tasks.append(dict(task_id=f"scf__{model}__{rec['parent']}__{start:06d}",stage='scf',model=model,parent=rec['parent'],start=start,stop=min(start+128,rec['n_structures'])))
    with (HERE/'data/scf_tasks.jsonl').open('w') as out:
        for t in tasks:out.write(json.dumps(t)+'\n')
    print('PREPARED',len(allrows),'structures',len(tasks),'SCF shards',flush=True)
if __name__=='__main__':main()
