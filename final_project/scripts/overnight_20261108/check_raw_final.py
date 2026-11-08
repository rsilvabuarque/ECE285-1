"""Independent saved-array FINAL audit; no NN forwards or energy members."""
import csv,json,hashlib,time
from pathlib import Path
from collections import Counter
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
ROOT=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project')
FINAL=ROOT/'runs/overnight_20261108/final';OUT=FINAL/'independent_raw_audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def canon(x,inverse):
    best=np.full(len(x),np.iinfo(np.uint64).max,np.uint64)
    for p in inverse:
        bits=np.packbits(1-x[:,p],axis=1,bitorder='big');padded=np.zeros((len(x),8),np.uint8);padded[:,:bits.shape[1]]=bits
        np.minimum(best,padded.view('>u8').ravel().astype(np.uint64),out=best)
    return best>>16
def main():
    require_compute_allocation();start=time.monotonic();OUT.mkdir(exist_ok=False)
    static=json.loads((FINAL/'static_manifest.json').read_text());plan=json.loads((FINAL/'paired_plan/plan.json').read_text())
    with np.load(static['inputs']['parent']['path'],allow_pickle=False) as z:perms=z['permutations'];host=len(z['numbers'])-48
    assert perms.shape==(192,48);inverse=np.argsort(perms,axis=1)
    hsha=static['inputs']['history']['sha256'];assert hsha==plan['inputs']['history']['sha256']
    assert sha(static['inputs']['history']['path'])==hsha
    expected={'A_graph_cvae':5,'B_spatial_ar':5,'C_dense_cvae':3,'D_nonphysical_ar':3,'A_radial_shells':5,'B_radial_shells':5,'B_remaining':3,'mutation':5,'uniform':5}
    actual=Counter(e['method'] for e in static['models']);assert actual==expected
    for method,count in expected.items():assert sorted(int(e['replicate_id']) for e in static['models'] if e['method']==method)==[2026110800+i for i in range(1,count+1)]
    pools=[('static',FINAL/'static_generated',static)]
    for name,stage in plan['stages'].items():pools.append((name,FINAL/'paired_generated/generated'/name,json.loads(Path(stage['manifest']['path']).read_text())))
    data_cache={};coupled_u={};coupled_z={};mutation_streams={};uniform_low={};receipts=[];endpoint_keys=set();all_model_samples=0
    for name,path,manifest in pools:
        receipt=json.loads((path/'receipt.json').read_text());protocol=json.loads((path/'protocol.json').read_text())
        assert receipt['status']=='FROZEN_BEFORE_SCORING' and receipt['protocol_sha256']==sha(path/'protocol.json')
        for file,digest in receipt['outputs'].items():assert sha(path/file)==digest
        assert protocol['inputs']['parent']['sha256']==static['inputs']['parent']['sha256'] and protocol['inputs']['history']['sha256']==hsha
        ns=manifest['namespace'];assert ns==protocol['namespace']
        assert (ns=='overnight-final-static-20261108-v1') if name=='static' else (ns=='overnight-final-paired-20261108-v1')
        count=128 if name=='static' else 64;rr=read(path/'attempts_all.csv');primary=[r for r in rr if r['diagnostic']=='primary']
        assert len(primary)==manifest['expected_primary_attempts']==receipt['primary_attempts']
        assert len(rr)==receipt['total_attempts'];assert len({r['attempt_id'] for r in rr})==len(rr)
        expected_diag=4992 if name=='static' else 0;assert len(rr)-len(primary)==expected_diag
        with np.load(path/'raw_attempts.npz',allow_pickle=False) as z:raw={k:z[k] for k in z.files}
        x=raw['occupancy'];assert x.shape==(len(rr),48) and np.isin(x,[0,1]).all()
        assert all(r['status']=='generated' and int(r['raw_row'])==i for i,r in enumerate(rr))
        assert np.array_equal(x.sum(1),np.array([int(r['n_target']) for r in rr]))
        keys=[format(int(c),'048b') for c in canon(x.astype(np.uint8),inverse)]
        assert keys==[r['canonical_mask'] for r in rr]
        ids=[f'mask48_{int(k,2):012x}' for k in keys];assert ids==[r['scoring_structure_id'] for r in rr]
        groups=Counter((r['method'],r['replicate_id'],int(r['n_target']),r['requested_bin'],r['diagnostic']) for r in rr)
        entries={(e['method'],str(e['replicate_id'])):e for e in manifest['models']}
        expected_groups={(e['method'],str(e['replicate_id']),int(n),b,'primary') for e in manifest['models'] for n in manifest['counts'] for b in manifest['requested_bins']}
        assert {(a,b,n,c,d) for a,b,n,c,d in groups if d=='primary'}==expected_groups and set(groups.values())=={count}
        if name!='static':assert all(str(e['replicate_id']) in {'0','1','2'} for e in manifest['models'])
        dsha=manifest['inputs']['data']['sha256'];dpath=Path(manifest['inputs']['data']['path']);assert sha(dpath)==dsha
        if dsha not in data_cache:
            with np.load(dpath,allow_pickle=False) as z:data_cache[dsha]={k:z[k] for k in ['occupancy','split','structure_ids']}
        data=data_cache[dsha];mutation_checks=0;diag_checks=0;low={}
        for i,r in enumerate(rr):
            method=r['method'];rep=r['replicate_id'];n=int(r['n_target']);j=int(r['attempt']);diagnostic=r['diagnostic'];entry=entries[method,rep]
            assert 0<=j<count and r['condition_schema_id']==protocol['condition_schema_id']
            assert r['parent_sha256']==static['inputs']['parent']['sha256'] and 'unrelaxed' in r['geometry_protocol']
            assert r['checkpoint_sha256']==entry.get('expected_checkpoint_sha256','not-applicable')
            assert r['fit_seed']==str(entry.get('fit_seed',''))
            coupling=(ns,rep,n,j);u=raw['uniforms'][i].tobytes()
            if coupling in coupled_u:assert coupled_u[coupling]==u
            else:coupled_u[coupling]=u
            dim=int(r['latent_dim']);assert dim in {0,4} and np.all(raw['latents'][i,dim:]==0)
            if dim:
                zkey=coupling+(diagnostic,);z=raw['latents'][i,:dim].tobytes()
                if zkey in coupled_z:assert coupled_z[zkey]==z
                else:coupled_z[zkey]=z
                if diagnostic=='primary' and r['requested_bin']=='low':low[method,rep,n,j]=z
                elif diagnostic=='latent_B':assert low[method,rep,n,j]!=z;diag_checks+=1
                all_model_samples+=1
            else:assert diagnostic=='primary'
            if entry.get('sampler')=='mutation':
                s=int(r['seed_source_row']);seed=raw['seed_occupancy'][i];k=int(r['swap_count']);assert 1<=k<=3
                assert r['seed_source_sha256']==dsha and str(data['split'][s])=='train' and str(data['structure_ids'][s])==r['seed_structure_id']
                assert np.array_equal(seed,data['occupancy'][s])
                occupied=np.flatnonzero(seed);vacant=np.flatnonzero(1-seed)
                take=occupied[np.argsort(raw['remove_uniforms'][i,occupied],kind='stable')[:k]];give=vacant[np.argsort(raw['add_uniforms'][i,vacant],kind='stable')[:k]]
                child=seed.copy();child[take]=0;child[give]=1
                assert np.array_equal(child,x[i]) and np.count_nonzero(child!=seed)==2*k
                assert r['removed_sites']=='|'.join(map(str,sorted(take))) and r['added_sites']=='|'.join(map(str,sorted(give)))
                stream=(k,raw['remove_uniforms'][i].tobytes(),raw['add_uniforms'][i].tobytes())
                if coupling in mutation_streams:assert mutation_streams[coupling]==stream
                else:mutation_streams[coupling]=stream
                mutation_checks+=1
            if method=='uniform' and r['requested_bin']=='low':uniform_low[name,rep,n,j]=(x[i].tobytes(),u,r['sampling_seed'],r['canonical_mask'])
        assert diag_checks==expected_diag
        for file,chosen in [('candidates.npz',rr),('primary_candidates.npz',primary),('diagnostic_candidates.npz',[r for r in rr if r['diagnostic']!='primary'])]:
            with np.load(path/file,allow_pickle=False) as z:
                wanted={r['scoring_structure_id'] for r in chosen};assert set(z['structure_ids'].astype(str))==wanted and len(z['structure_ids'])==len(wanted)
                assert np.array_equal(z['occupancy'],1-z['masks']) and np.array_equal(z['n_target'],z['occupancy'].sum(1))
                assert np.array_equal(z['n_atoms'],host+z['n_target'])
        if name!='static':endpoint_keys.update(keys)
        receipts.append(dict(stage=name,namespace=ns,primary_attempts=len(primary),diagnostic_attempts=len(rr)-len(primary),
            canonical_orbits=len(set(keys)),mutation_TRAIN_lineage_and_raw2k_checks=mutation_checks,latent_resample_pair_checks=diag_checks,
            generation_receipt_sha256=sha(path/'receipt.json'),manifest_primary_entries=len(entries),condition_groups=len(expected_groups)))
    for rep in ['0','1','2']:
        for n in [12,24,36]:
            for j in range(64):assert uniform_low['current_low',rep,n,j]==uniform_low['shared_uniform',rep,n,j]
    paired=json.loads((FINAL/'paired_generated/receipt.json').read_text());tasks=json.loads((FINAL/'paired_generated/endpoint_union_tasks.json').read_text())['tasks']
    assert paired['status']=='FROZEN_PROPOSALS_AND_ENDPOINT_TASKS_NO_SCFS' and paired['generated_attempts']==10944
    assert len(tasks)==len({t['physical_key'] for t in tasks})==paired['unique_physical_tasks']==len(endpoint_keys)==7616
    assert receipts[0]['primary_attempts']==44928 and receipts[0]['diagnostic_attempts']==4992
    assert sum(r['primary_attempts'] for r in receipts[1:])==10944
    result=dict(status='PASS',static_primary_attempts=44928,static_diagnostic_attempts=4992,endpoint_primary_attempts=10944,
        all_attempts_preserved=True,actual_parent_operations=192,history_sha256=hsha,stage_receipts=receipts,
        static_three_five_seed_budgets_verified=True,endpoint_zero_based_three_replicates_verified=True,
        common_actual_latents_and_AR_uniforms_verified=True,latent_resamples_change_actual_z_only=True,
        cross_endpoint_mutation_edit_streams_verified=True,endpoint_internal_uniform_LOW_exact_reuse_pairs=576,
        canonical_identities_and_query_unions_verified=True,mutation_seed_classes_not_recomputed_without_energy_lookup=True,
        energy_fields_read=0,new_NN_forwards=0,new_generation=0,new_MLIP_calls=0,new_optimizer_updates=0,
        source_sha256=sha(__file__),elapsed_seconds=time.monotonic()-start)
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
