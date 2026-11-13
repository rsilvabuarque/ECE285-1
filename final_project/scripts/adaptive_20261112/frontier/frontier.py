"""Bounded greedy versus energy-blind single-swap search, after required core completion."""
from pathlib import Path
import csv, importlib.util, json, time
import numpy as np
BASE=Path(__file__).parent.parent/'study_v2'
spec=importlib.util.spec_from_file_location('frozen_core_driver',BASE/'orchestrate.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
F=Path(__file__).parent
WORKER_OVERRIDE=BASE/'oracle_worker_override.json'
WORKER_CONFIG=json.loads(WORKER_OVERRIDE.read_text());W=Path(WORKER_CONFIG['source'])
ARMS=('greedy','random_walk','uniform');COUNTS=(12,24,36);BINS=('low','high');STEPS=4;STARTS=16;NEIGHBORS=8


def budget():
    d.assert_deadline()
    p=F/'protocol.json'
    if p.exists() and time.time()-json.loads(p.read_text())['started_unix']>48*60:raise TimeoutError('Optional48minute active budget exhausted; reserve two minutes for worker cleanup')
def taskfile(name,tasks):
    p=F/'taskfiles'/f'{name}.json';d.write(p,dict(status='FROZEN_OPTIONAL_FRONTIER_TASKS',created_unix=time.time(),protocol=d.binding(F/'protocol.json'),source=d.binding(__file__),tasks=tasks));return p

def score(name,pools,stages=None):
    from cathode_cvae.iterative_comparison import collect_oracle
    after=max(json.loads((p/'receipt.json').read_text())['created_unix'] for p in stages) if stages else json.loads((F/'seeds/receipt.json').read_text())['created_unix']
    expected=set()
    for pool in pools:
        with np.load(pool,allow_pickle=False) as z:expected.update(z['structure_ids'].astype(str).tolist())
    roots={m:F/name/('oracle_'+m) for m in ['uma','mace']}
    for m,root in roots.items():
        for pool in pools:d.oracle.register_pool(root,pool,d.PARENT,models=[m],batch_size=64)
    tasks=[]
    for m,root in roots.items():
        for i in range(9):
            t=d.template(f'{m}_{i:02d}',['-m','cathode_cvae.adaptive_oracle','worker','--root',root,'--model',m,'--checkpoint',d.CHECKPOINTS[m],'--threads','4'],[F/'protocol.json',root/'registry.json',W/'cathode_cvae/adaptive_oracle.py',W/'cathode_cvae/oracle_queue.py',WORKER_OVERRIDE,Path(WORKER_CONFIG['snapshot_receipt']['path'])],python=d.UMA_PY if m=='uma' else d.MACE_PY)
            t['env']=dict(t['env'],PYTHONPATH=str(W));tasks.append(t)
    assert d.ENV['PYTHONPATH']==str(d.SRC)
    p=taskfile(name,tasks);budget();d.start_tasks(p);pending=set(roots);last={}
    try:
        while pending:
            budget()
            for m in list(pending):
                root=roots[m];reg=json.loads((root/'registry.json').read_text());rr=list((root/'scores'/m).glob('*.json')) if (root/'scores'/m).exists() else []
                if list((root/'claims').glob('*/failure.json')):raise RuntimeError('Frontier oracle task failed')
                if any(json.loads(x.read_text())['status']!='complete' for x in rr):raise RuntimeError('Frontier structure evaluation failed')
                if last.get(m)!=len(rr):d.log('FRONTIER',name,m,len(rr),len(reg['tasks']));last[m]=len(rr)
                if len(rr)==len(reg['tasks']):(root/'STOP').write_text('All optional phase tasks complete.\n');pending.remove(m)
            time.sleep(5)
    except BaseException:
        for root in roots.values():(root/'STOP').write_text('Optional bounded search stopped; preserve incomplete outputs.\n')
        raise
    d.wait_tasks(p);scores={};verified={}
    for m,root in roots.items():
        out=F/name/('collected_'+m);r=d.oracle.collect(root,out)
        if r['status']!='PASS':raise ValueError('Frontier collection incomplete')
        v=F/name/('verified_'+m+'.json')
        if stages:d.ai.verify_oracle(stages,root,out/'receipt.json',v,m)
        else:
            ev=collect_oracle(root,out/'receipt.json',expected,d.sha256(d.PARENT),m,after)
            d.write(v,dict(status='VERIFIED_FRESH_FRONTIER_TRAIN_SEED_RESCORING',created_unix=time.time(),teacher=m,unique_structures=len(expected),source=d.binding(__file__),seed_receipt=d.binding(F/'seeds/receipt.json'),oracle_evidence=ev))
        with np.load(out/f'{m}.npz',allow_pickle=False) as z:scores[m]={str(s):float(e)/int(a) for s,e,a in zip(z['structure_ids'],z['energy_eV'],z['n_atoms'])}
        verified[m]=v
    return scores,verified


def seed_pool(boundary):
    _,_,protected=d.ai.load_boundary(boundary);perms,_=d._parent(d.PARENT)
    with np.load(BASE/'seed_bank/bank.npz',allow_pickle=False) as z:x=z['occupancy'];n=z['n_target'];b=z['energy_bin'].astype(str)
    keys=d._canonical_vacancy(x,perms);states={};rows=[];union=set()
    for seed in d.SEEDS:
        cells=[]
        for count in COUNTS:
            for label in BINS:
                eligible=np.asarray([i for i in np.flatnonzero((n==count)&(b==label)) if keys[i] not in protected],int)
                if len(eligible)<STARTS:raise ValueError('Insufficient original unprotected TRAIN seeds')
                rng=np.random.default_rng(d.ai.stream_seed('frontier_start_v1',seed,count,label));chosen=rng.choice(eligible,STARTS,replace=False)
                cells.append(np.asarray([[1-int(c) for c in keys[i]] for i in chosen],np.uint8))
                for trajectory,i in enumerate(chosen):
                    rows.append(dict(seed=seed,n_target=count,requested_bin=label,trajectory=trajectory,canonical_mask=str(keys[i]),structure_id=d.ai.identity(keys[i])));union.add(str(keys[i]))
        for arm in ('greedy','random_walk'):states[(arm,seed)]=np.asarray(cells,np.uint8)
    out=F/'seeds';out.mkdir(exist_ok=False);ordered=sorted(union);occ=np.asarray([[1-int(c) for c in k] for k in ordered],np.uint8)
    np.savez_compressed(out/'pool.npz',occupancy=occ,masks=1-occ,structure_ids=np.asarray([d.ai.identity(k) for k in ordered]),n_Li=occ.sum(1),n_target=occ.sum(1),parent_sha256=np.asarray(d.sha256(d.PARENT)))
    np.savez_compressed(out/'states.npz',**{f'{arm}_{seed}':v for (arm,seed),v in states.items()});d.ai.write_csv(out/'trajectories.csv',rows)
    d.write(out/'receipt.json',dict(status='FROZEN_FRONTIER_TRAIN_SEEDS_BEFORE_RESCORING',created_unix=time.time(),pool=d.binding(out/'pool.npz'),states=d.binding(out/'states.npz'),trajectories=d.binding(out/'trajectories.csv'),seed_bank=d.binding(BASE/'seed_bank/receipt.json'),boundary=d.binding(boundary/'receipt.json'),unique_seeds=len(union),trajectory_start_count=len(rows),all_original_unprotected_train=True,source=d.binding(__file__)))
    return states,rows


def make_stage(step,arm,seed,states,boundary):
    boundary_receipt,known,protected=d.ai.load_boundary(boundary);perms,_=d._parent(d.PARENT)
    out=F/f'step{step}'/'stages'/f'{arm}_{seed}';out.mkdir(parents=True,exist_ok=False)
    ns=f'frontier_v1_step{step}';attempts=STARTS*NEIGHBORS
    if arm!='uniform':np.savez_compressed(out/'states_before.npz',occupancy=states[(arm,seed)])
    protocol=dict(status='FROZEN_ADAPTIVE_PROPOSALS_BEFORE_GENERATION',created_unix=time.time(),parent=d.binding(d.PARENT),schema=d.binding(d.SCHEMA),schema_id=d.load_schema(d.SCHEMA)['identity'],boundary=d.binding(boundary/'receipt.json'),namespace=ns,seed=int(seed),method=arm,sampler='uniform' if arm=='uniform' else 'frontier_single_swap',checkpoint=None,seed_bank=d.binding(F/'seeds/receipt.json'),counts=list(COUNTS),bins=list(BINS),attempts_per_cell=attempts,queries_per_cell=attempts,latent_stream_width=32,selection='First eligible unique orbit; no energy filtering or rejection of query union.',source=d.binding(__file__),frontier_protocol=d.binding(F/'protocol.json'),states_before=None if arm=='uniform' else d.binding(out/'states_before.npz'),one_swap='one occupied Li site emptied and one vacant Li site filled, before canonical orientation',random_walk_selection='Per trajectory uniform neighbor index frozen now; cyclic first eligible fallback, independent of energies',greedy_selection='After complete queried union, choose best UMA energy per trajectory, ties canonical-mask lexicographic; no eligible neighbor -> keep state')
    d.write(out/'protocol.json',protocol);raw=[];zz=[];uu=[];rows=[];selected={};allkeys=set();choices=[]
    for cell,(count,label) in enumerate((n,b) for n in COUNTS for b in BINS):
        rng=np.random.default_rng(d.ai.stream_seed(ns,seed,count,label));u=rng.random((attempts,48)).astype(np.float32);z=rng.standard_normal((attempts,32)).astype(np.float32)
        blind=rng.integers(0,NEIGHBORS,size=STARTS).tolist();choices.append(blind)
        if arm=='uniform':
            x=np.zeros((attempts,48),np.uint8);order=np.argsort(u,axis=1,kind='stable');np.put_along_axis(x,order[:,:count],1,axis=1)
        else:
            x=np.repeat(states[(arm,seed)][cell],NEIGHBORS,axis=0).copy()
            for i in range(attempts):
                occupied=np.flatnonzero(x[i]);vacant=np.flatnonzero(1-x[i]);x[i,occupied[np.argmin(u[i,occupied])]]=0;x[i,vacant[np.argmin(u[i,vacant])]]=1
            before=np.repeat(states[(arm,seed)][cell],NEIGHBORS,axis=0)
            if not np.all(np.abs(x.astype(int)-before).sum(1)==2):raise ValueError('Frontier one-swap invariant failed')
        keys=d._canonical_vacancy(x,perms);seen=set();accepted=0
        for i,key in enumerate(keys):
            status='protected_boundary' if key in protected else 'known_boundary' if key in known else 'duplicate_within_cell' if key in seen else 'query_cap' if accepted>=attempts else 'selected'
            if status=='selected':accepted+=1;selected[str(key)]=d.ai.identity(key)
            seen.add(str(key));allkeys.add(str(key));j=len(raw);raw.append(x[i]);zz.append(z[i]);uu.append(u[i])
            rows.append(dict(attempt_id=f'{ns}|{arm}|{seed}|{count}|{label}|{i}',method=arm,seed=seed,n_target=count,requested_bin=label,raw_row=j,canonical_mask=str(key),structure_id=d.ai.identity(key),status=status,trajectory=i//NEIGHBORS,neighbor=i%NEIGHBORS,predeclared_blind_neighbor=blind[i//NEIGHBORS]))
    ordered=sorted(selected);occ=np.asarray([[1-int(c) for c in k] for k in ordered],np.uint8).reshape(-1,48)
    np.savez_compressed(out/'candidates.npz',occupancy=occ,masks=1-occ,structure_ids=np.asarray([selected[k] for k in ordered]),n_Li=occ.sum(1),n_target=occ.sum(1),parent_sha256=np.asarray(d.sha256(d.PARENT)))
    np.savez_compressed(out/'raw_attempts.npz',occupancy=np.asarray(raw,np.uint8),latents=np.asarray(zz,np.float32),uniforms=np.asarray(uu,np.float32),parent_sha256=np.asarray(d.sha256(d.PARENT)))
    d.ai.write_csv(out/'attempts.csv',rows);d.write(out/'all_generated_keys.json',sorted(allkeys));d.write(out/'blind_choices.json',choices)
    outputs={name:d.sha256(out/name) for name in ['candidates.npz','raw_attempts.npz','attempts.csv','all_generated_keys.json','blind_choices.json']}
    if arm=='random_walk':
        planned=[]
        for count,label in ((n,b) for n in COUNTS for b in BINS):
            for trajectory in range(STARTS):
                candidates=[r for r in rows if r['n_target']==count and r['requested_bin']==label and r['trajectory']==trajectory and r['status']=='selected']
                chosen=None
                if candidates:
                    blind=candidates[0]['predeclared_blind_neighbor'];chosen=min(candidates,key=lambda r:((r['neighbor']-blind)%NEIGHBORS,r['neighbor']))
                planned.append(dict(n_target=count,requested_bin=label,trajectory=trajectory,structure_id=None if chosen is None else chosen['structure_id'],canonical_mask=None if chosen is None else chosen['canonical_mask'],decision='Frozen before any energy scoring; first eligible in predeclared cyclic neighbor order'))
        d.write(out/'predeclared_moves.json',planned);outputs['predeclared_moves.json']=d.sha256(out/'predeclared_moves.json')
    if arm!='uniform':outputs['states_before.npz']=d.sha256(out/'states_before.npz')
    from collections import Counter
    d.write(out/'receipt.json',dict(status='FROZEN_ADAPTIVE_STAGE_BEFORE_SCORING',created_unix=time.time(),protocol=d.binding(out/'protocol.json'),outputs=outputs,attempts=len(rows),selected_unique=len(ordered),status_counts=dict(Counter(r['status'] for r in rows)),parent=protocol['parent'],schema=protocol['schema'],boundary=protocol['boundary'],no_energy_filtering=True))
    d.ai.validate_stage(out);return out


def next_states(step,stages,states,scores,all_scores,starts,best,metrics):
    perms,_=d._parent(d.PARENT);inv=np.argsort(perms,axis=1)
    for teacher in all_scores:all_scores[teacher].update(scores[teacher])
    for stage in stages:
        rows=d.ai.read_csv(stage/'attempts.csv');arm=rows[0]['method'];seed=int(rows[0]['seed'])
        for cell,(count,label) in enumerate((n,b) for n in COUNTS for b in BINS):
            rr=[r for r in rows if int(r['n_target'])==count and r['requested_bin']==label];selected=[r for r in rr if r['status']=='selected']
            sign=1 if label=='high' else -1
            if arm!='uniform':
                for trajectory in range(STARTS):
                    candidates=[r for r in selected if int(r['trajectory'])==trajectory]
                    if not candidates:continue
                    if arm=='greedy':chosen=min(candidates,key=lambda r:(-sign*scores['uma'][r['structure_id']],r['canonical_mask']))
                    else:
                        planned=json.loads((stage/'predeclared_moves.json').read_text())
                        move=next(r for r in planned if r['n_target']==count and r['requested_bin']==label and r['trajectory']==trajectory)
                        if move['structure_id'] is None:raise ValueError('Predeclared blind move missing despite eligible candidates')
                        chosen=next(r for r in candidates if r['structure_id']==move['structure_id'] and r['canonical_mask']==move['canonical_mask'])
                    states[(arm,seed)][cell,trajectory]=np.asarray([1-int(c) for c in chosen['canonical_mask']],np.uint8)
            if arm!='uniform':diversity_masks=states[(arm,seed)][cell]
            else:diversity_masks=np.asarray([[1-int(c) for c in r['canonical_mask']] for r in selected[:STARTS]],np.uint8).reshape(-1,48)
            transformed=diversity_masks[:,inv];distances=[]
            for i in range(len(diversity_masks)):
                for j in range(i+1,len(diversity_masks)):distances.append(float(np.count_nonzero(diversity_masks[i][None,:]!=transformed[j],axis=1).min()/2))
            diversity=float(np.mean(distances)) if distances else ''
            for teacher in ['uma','mace']:
                key=(teacher,arm,seed,count,label);values=[scores[teacher][r['structure_id']] for r in selected]
                if values:best[key]=(min if label=='low' else max)([best[key],*values])
                initial=starts[(teacher,seed,count,label)]
                current=[]
                if arm!='uniform':
                    keys=d._canonical_vacancy(states[(arm,seed)][cell],perms);current=[all_scores[teacher][d.ai.identity(k)] for k in keys]
                metrics.append(dict(step=step,teacher=teacher,arm=arm,seed=seed,n_target=count,requested_bin=label,attempts=len(rr),selected_query_incidences=len(selected),query_uniqueness_fraction=len(selected)/len(rr),duplicate_within_cell=sum(r['status']=='duplicate_within_cell' for r in rr),diversity_sample='current16paths' if arm!='uniform' else 'first16eligibleuniformproposals',diversity_sample_masks=len(diversity_masks),mean_symmetry_swap_distance=diversity,current_unique_orbits=len(set(keys.tolist())) if arm!='uniform' else '',known_excluded=sum(r['status']=='known_boundary' for r in rr),protected_excluded=sum(r['status']=='protected_boundary' for r in rr),best_initial_eV_atom=initial,best_so_far_eV_atom=best[key],desired_best_gain_meV_atom=sign*(best[key]-initial)*1000,mean_current_eV_atom=float(np.mean(current)) if current else '',best_current_eV_atom=(min(current) if label=='low' else max(current)) if current else ''))
    np.savez_compressed(F/f'step{step}/states_after.npz',**{f'{arm}_{seed}':v for (arm,seed),v in states.items()})
    d.ai.write_csv(F/f'step{step}/metrics.csv',[r for r in metrics if r['step']==step])


def main():
    d.require_compute_allocation()
    for item in [WORKER_CONFIG['snapshot_receipt'],WORKER_CONFIG['validation']]:d.binding(item['path'],item['sha256'])
    worker_manifest=json.loads(Path(WORKER_CONFIG['snapshot_receipt']['path']).read_text())
    assert worker_manifest['byte_reversal_recovers_original_oracle']
    for name,digest in worker_manifest['files'].items():d.binding(W/name,digest)
    if not (BASE/'completion.json').exists() or not (BASE/'analysis/receipt.json').exists():raise ValueError('Required core results and derived analysis must be ready first')
    boundary=BASE/'boundary_final'
    if (F/'input_boundary.json').exists():
        b=json.loads((F/'input_boundary.json').read_text());d.binding(b['path'],b['sha256']);boundary=Path(b['path']).parent
    d.ai.load_boundary(boundary)
    d.write(F/'protocol.json',dict(status='FROZEN_OPTIONAL_FRONTIER_BEFORE_SCORING',started_unix=time.time(),source=d.binding(__file__),queue_only_worker_override=d.binding(WORKER_OVERRIDE),core_completion=d.binding(BASE/'completion.json'),core_analysis=d.binding(BASE/'analysis/receipt.json'),initial_boundary=d.binding(boundary/'receipt.json'),steps=STEPS,seeds=d.SEEDS,counts=list(COUNTS),bins=list(BINS),starts_per_cell=STARTS,one_swap_neighbors=NEIGHBORS,attempts_and_querycap_per_cell_step=128,arms=list(ARMS),maximum_new_scored_geometry_model_pairs=55296,maximum_seed_rescoring_pairs=576,maximum_elapsed_minutes=48,hard_stop_minutes=50,maximum_node_hours=4,node_equivalent=4.5,seed_selection='Original unprotected H0TRAIN fixed-bin seeds; same16percell perreplicate forgreedy/random-walk, onefreshseedrescoringpool recordedseparately',greedy='Best measuredUMA neighbor, tie lexicalcanonicalmask; maymoveuphill; best-so-far retained',random_walk='Blindneighbor index frozen before energies; cyclic firsteligible fallback, noenergy-basedselection',uniform='Fixed-countuniform proposals equalattempt/querycap; noidentical editsemantics claim',known='Allprior core and allcompletedoptional rawattempts, allprotected unchanged; firsteligible canonicalunique; norepeatedquery topup',diversity='Mean pairwise minimum Li/vacancy-swap distance over savedparent symmetries (Hamming/2); current16walk masks or first16eligibleuniformproposals, no energy selection for diversity.',claims='Measuredfixed-lattice MLIP endpointsearch only; noglobalextremum,DFTorentropyclaim; no learnedneuraleditsemantics'))
    states,seedrows=seed_pool(boundary);score0,_=score('seed_rescoring',[F/'seeds/pool.npz']);all_scores={m:dict(v) for m,v in score0.items()};starts={};best={};metrics=[]
    for teacher in ['uma','mace']:
        for seed in d.SEEDS:
            for count in COUNTS:
                for label in BINS:
                    rows=[r for r in seedrows if r['seed']==seed and r['n_target']==count and r['requested_bin']==label];values=[score0[teacher][r['structure_id']] for r in rows];v=(min if label=='low' else max)(values);starts[(teacher,seed,count,label)]=v
                    for arm in ARMS:best[(teacher,arm,seed,count,label)]=v
    for step in range(1,STEPS+1):
        budget();stages=[make_stage(step,arm,seed,states,boundary) for arm in ARMS for seed in d.SEEDS]
        if step==1:
            checks=[]
            for seed in d.SEEDS:
                a=F/'step1/stages'/f'greedy_{seed}';b=F/'step1/stages'/f'random_walk_{seed}'
                with np.load(a/'raw_attempts.npz',allow_pickle=False) as za,np.load(b/'raw_attempts.npz',allow_pickle=False) as zb:
                    if any(not np.array_equal(za[k],zb[k]) for k in ['occupancy','latents','uniforms']):raise ValueError('First-step greedy and blind proposals differ despite identical states/streams')
                with np.load(a/'candidates.npz',allow_pickle=False) as za,np.load(b/'candidates.npz',allow_pickle=False) as zb:
                    if not np.array_equal(za['structure_ids'],zb['structure_ids']):raise ValueError('First-step oracle queries differ')
                checks.append(dict(seed=seed,greedy=d.binding(a/'receipt.json'),random_walk=d.binding(b/'receipt.json'),raw_and_query_equality=True))
            d.write(F/'first_step_paired_control_check.json',dict(status='EXACT_IDENTICAL_FIRST_STEP_PROPOSALS_AND_QUERIES',checks=checks))
        scores,_=score(f'step{step}',[p/'candidates.npz' for p in stages],stages)
        next_states(step,stages,states,scores,all_scores,starts,best,metrics)
        new=F/f'boundary_after_step{step}';d.ai.extend_boundary(boundary,stages,[],new);boundary=new
    d.ai.write_csv(F/'all_metrics.csv',metrics);d.write(F/'completion.json',dict(status='BOUNDED_FRONTIER_COMPLETE',created_unix=time.time(),protocol=d.binding(F/'protocol.json'),source=d.binding(__file__),metrics=d.binding(F/'all_metrics.csv'),boundary=d.binding(boundary/'receipt.json'),elapsed_seconds=time.time()-json.loads((F/'protocol.json').read_text())['started_unix']))
if __name__=='__main__':
    try:main()
    except BaseException as e:
        if not (F/'failure.json').exists():d.write(F/'failure.json',dict(status='OPTIONAL_FRONTIER_STOPPED',created_unix=time.time(),error_type=type(e).__name__,error=str(e)))
        raise
