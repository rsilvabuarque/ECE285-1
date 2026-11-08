"""Independent FINAL scalar/mask recount; no evaluator imports or NN calls."""
import csv,json,hashlib,math,time
from collections import defaultdict
from pathlib import Path
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
ROOT=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project')
R=ROOT/'runs/overnight_20261108';F=R/'final';OUT=F/'static_independent_metrics'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def write(p,rows):
    with p.open('x',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def mean(v):return math.fsum(v)/len(v)
def sd(v):
    m=mean(v);return math.sqrt(math.fsum((x-m)**2 for x in v)/(len(v)-1))
def truth(v):return v is True or v=='True'
def close(a,b):
    a=float(a);b=float(b)
    if not math.isfinite(a) or not math.isfinite(b):raise AssertionError('Nonfinite independently aggregated metric')
    if abs(a-b)>16*np.finfo(np.float64).eps*max(1,abs(a),abs(b)):raise AssertionError(('FP64 aggregate differs',a,b))
def canonical(x,perms):
    best=np.full(len(x),np.iinfo(np.uint64).max,np.uint64)
    for p in perms:
        bits=np.packbits(1-x[:,np.argsort(p)],axis=1,bitorder='big');pad=np.zeros((len(x),8),np.uint8);pad[:,:6]=bits
        np.minimum(best,pad.view('>u8').ravel().astype(np.uint64),out=best)
    return [format(int(v)>>16,'048b') for v in best]
def label(e,edges):return 'low' if e<=edges[0] else 'middle' if e<=edges[1] else 'high'
def main():
    require_compute_allocation();started=time.monotonic();OUT.mkdir()
    collected=json.loads((F/'static_collected/receipt.json').read_text());assert collected['status']=='PASS' and collected['tasks']==1464
    manifest=json.loads((F/'static_manifest.json').read_text());generation=json.loads((F/'static_generated/receipt.json').read_text())
    ev=F/'static_evaluation';evaluation=json.loads((ev/'receipt.json').read_text());assert evaluation['status']=='PASS'
    attempts=read(F/'static_generated/attempts_all.csv');assert len(attempts)==49920
    assert sha(F/'static_generated/raw_attempts.npz')==generation['outputs']['raw_attempts.npz']
    with np.load(F/'static_generated/raw_attempts.npz',allow_pickle=False) as z:x=z['occupancy'];latents=z['latents'];uniforms=z['uniforms']
    with np.load(manifest['inputs']['parent']['path'],allow_pickle=False) as z:perms=z['permutations']
    assert perms.shape==(192,48) and np.isin(x,[0,1]).all()
    keys=canonical(x.astype(np.uint8),perms);assert keys==[r['canonical_mask'] for r in attempts]
    assert np.array_equal(x.sum(1),np.array([int(r['n_target']) for r in attempts]))
    with np.load(manifest['inputs']['data']['path'],allow_pickle=False) as z:
        train=z['split'].astype(str)=='train';tx=z['occupancy'][train]
    trainkeys=set(canonical(tx.astype(np.uint8),perms))
    history_value=json.loads(Path(manifest['inputs']['history']['path']).read_text())
    history=set(history_value if isinstance(history_value,list) else history_value['keys']);assert len(history)==344001
    q20={(r['teacher'],int(r['n_Li'])):float(r['q20_eV_atom']) for r in read(ROOT/'artifacts/conditioning_20261107/data/generation_thresholds.csv') if r['source']=='main_train'}
    outcome_rows=read(ev/'attempt_outcomes.csv');actual_outcomes={(r['teacher'],r['attempt_id']):r for r in outcome_rows};assert len(outcome_rows)==len(actual_outcomes)==99840
    cells={};pair_source=defaultdict(dict);energies={};integer_checks=0
    for r in attempts:pair_source[r['method'],r['replicate_id'],int(r['n_target']),r['paired_id']][r['requested_bin'] if r['diagnostic']=='primary' else r['diagnostic']]=r
    for teacher in ['uma','mace']:
        schema=json.loads((ROOT/f'artifacts/overnight_20261107/data/condition_schema_{teacher}.json').read_text());body={k:v for k,v in schema.items() if k!='identity'}
        assert hashlib.sha256(json.dumps(body,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()==schema['identity']
        assert evaluation['score_sources'][teacher]['schema_sha256']==sha(ROOT/f'artifacts/overnight_20261107/data/condition_schema_{teacher}.json')
        with np.load(F/f'static_collected/{teacher}.npz',allow_pickle=False) as z:
            d={k:z[k] for k in ['structure_ids','energy_eV','n_atoms','n_Li','status','provenance']}
        assert len(set(d['structure_ids'].astype(str)))==len(d['structure_ids']);lookup={str(s):i for i,s in enumerate(d['structure_ids'])}
        for i,r in enumerate(attempts):
            n=int(r['n_target']);key=keys[i];sid=f'mask48_{int(key,2):012x}';assert sid==r['scoring_structure_id']
            j=lookup[sid];status=str(d['status'][j]);assert status=='success' or (status=='cached' and str(d['provenance'][j])=='cached')
            assert int(d['n_atoms'][j])==144+n and int(d['n_Li'][j])==n and np.isfinite(d['energy_eV'][j])
            e=float(d['energy_eV'][j])/int(d['n_atoms'][j]);b=label(e,schema['compositions'][str(n)]['edges_eV_atom'])
            novel_train=key not in trainkeys;novel_history=key not in history;novel=novel_train and novel_history;fixed=e<=q20[teacher,n]
            a=actual_outcomes[teacher,r['attempt_id']]
            assert truth(a['energy_known']) and a['measured_bin']==b and float(a['energy_eV_atom'])==e
            for field,v in [('novel_train',novel_train),('novel_history',novel_history),('novel_both',novel),('passes_fixed_q20',fixed)]:assert truth(a[field])==v;integer_checks+=1
            energies[teacher,r['attempt_id']]=e
            if r['diagnostic']!='primary':continue
            g=(teacher,r['method'],r['replicate_id'],n,r['requested_bin']);c=cells.setdefault(g,dict(attempts=0,distinct=set(),useful=set(),fixed=set(),hit=0,actual=defaultdict(int)))
            c['attempts']+=1;c['distinct'].add(key);c['actual'][b]+=1
            if b==r['requested_bin']:c['hit']+=1
            if novel and b==r['requested_bin']:c['useful'].add(key)
            if novel and fixed:c['fixed'].add(key)
    cell_rows=read(ev/'metrics.csv');actual_cells={(r['teacher'],r['method'],r['replicate_id'],int(r['n_Li']),r['requested_bin']):r for r in cell_rows}
    assert len(cell_rows)==len(actual_cells)==len(cells)==702 and set(actual_cells)==set(cells)
    seed_values=defaultdict(list);low_values=defaultdict(list);independent_cells=[]
    for g,c in cells.items():
        assert c['attempts']==128;a=actual_cells[g]
        for field,value in [('attempts',128),('scored_attempts',128),('distinct_orbits',len(c['distinct'])),('distinct_novel_inbin_orbits',len(c['useful'])),('distinct_novel_fixed_q20_orbits',len(c['fixed']))]:assert int(a[field])==value;integer_checks+=1
        y=len(c['useful'])/128;low=len(c['fixed'])/128;close(a['novel_inbin_yield_per_attempt'],y);close(a['bin_hit_rate_per_attempt'],c['hit']/128);close(a['novel_fixed_q20_yield_per_attempt'],low)
        seed_values[g[:3]].append(y)
        if g[4]=='low':low_values[g[:3]].append(low)
        independent_cells.append(dict(teacher=g[0],method=g[1],replicate_id=g[2],n_Li=g[3],requested_bin=g[4],attempts=128,distinct_novel_inbin_orbits=len(c['useful']),novel_inbin_yield=y,distinct_novel_original_q20_orbits=len(c['fixed']),original_q20_yield=low))
    confusion=read(ev/'confusion.csv');assert len(confusion)==702*6
    assert len({(r['teacher'],r['method'],r['replicate_id'],r['n_Li'],r['requested_bin'],r['measured_bin']) for r in confusion})==len(confusion)
    for r in confusion:
        g=(r['teacher'],r['method'],r['replicate_id'],int(r['n_Li']),r['requested_bin']);assert int(r['count'])==cells[g]['actual'].get(r['measured_bin'],0);integer_checks+=1
    seed_rows=read(ev/'seed_metrics.csv');actual_seeds={(r['teacher'],r['method'],r['replicate_id']):r for r in seed_rows};assert len(seed_rows)==len(actual_seeds)==78 and set(actual_seeds)==set(seed_values)
    independent_seeds=[];by_seed={}
    for key,values in seed_values.items():
        assert len(values)==9 and len(low_values[key])==3;y=mean(values);low=mean(low_values[key]);a=actual_seeds[key]
        close(a['macro_composition_bin_yield'],y);close(a['fixed_q20_LOW_request_yield'],low)
        by_seed[key]=(y,low);independent_seeds.append(dict(teacher=key[0],method=key[1],replicate_id=key[2],macro_yield=y,original_q20_LOW_yield=low))
    for a in read(ev/'summary.csv'):
        rows=[v for k,v in by_seed.items() if k[:2]==(a['teacher'],a['method'])];assert int(a['fit_or_sampler_replicates'])==len(rows)
        close(a['mean_novel_inbin_yield'],mean([v[0] for v in rows]));close(a['descriptive_seed_SD'],sd([v[0] for v in rows]));close(a['mean_fixed_q20_LOW_request_yield'],mean([v[1] for v in rows]))
    pair_checks=0
    for a in read(ev/'paired_attempts.csv'):
        group=pair_source[a['method'],a['replicate_id'],int(a['n_Li']),a['paired_id']]
        right={'condition_middle_minus_low':'middle','condition_high_minus_low':'high','alternate_latent_minus_low':'latent_B'}[a['contrast']]
        first,second=group['low'],group[right];i,j=int(first['raw_row']),int(second['raw_row'])
        expected=1000*(energies[a['teacher'],second['attempt_id']]-energies[a['teacher'],first['attempt_id']])
        assert truth(a['both_energies_known']) and float(a['energy_shift_meV_atom'])==expected
        assert int(a['raw_Hamming'])==np.count_nonzero(x[i]!=x[j]) and truth(a['same_orbit'])==(keys[i]==keys[j])
        assert np.array_equal(uniforms[i],uniforms[j]);assert truth(a['actual_z_equal'])==np.array_equal(latents[i],latents[j]);pair_checks+=1
    contrasts=[]
    comparisons=[('mechanism_first3','A_graph_cvae','B_spatial_ar',3),('mechanism_first3','C_dense_cvae','A_graph_cvae',3),('mechanism_first3','D_nonphysical_ar','B_spatial_ar',3),
        ('five_fit_shell_effect_CVAE','A_radial_shells','A_graph_cvae',5),('five_fit_shell_effect_AR','B_radial_shells','B_spatial_ar',5),
        ('five_fit_latent_effect_legacy','A_graph_cvae','B_spatial_ar',5),('five_fit_latent_effect_shell','A_radial_shells','B_radial_shells',5),
        ('remaining_control_first3','B_remaining','B_spatial_ar',3),('remaining_vs_selected_shell_first3','B_remaining','B_radial_shells',3),
        ('selected_CVAE_vs_mutation','A_radial_shells','mutation',5),('selected_AR_vs_mutation','B_radial_shells','mutation',5)]
    for teacher in ['uma','mace']:
        for scope,method,reference,count in comparisons:
            for i in range(1,count+1):
                rep=str(2026110800+i);a=by_seed[teacher,method,rep];b=by_seed[teacher,reference,rep]
                contrasts.append(dict(teacher=teacher,comparison=scope,method=method,reference=reference,replicate_id=rep,delta_macro_yield=a[0]-b[0],delta_original_q20_LOW_yield=a[1]-b[1]))
        for i in range(1,6):
            rep=str(2026110800+i);a=by_seed[teacher,'A_radial_shells',rep];b=by_seed[teacher,'A_graph_cvae',rep];c=by_seed[teacher,'B_radial_shells',rep];d=by_seed[teacher,'B_spatial_ar',rep]
            contrasts.append(dict(teacher=teacher,comparison='five_fit_2x2_interaction',method='shell_effect_CVAE',reference='shell_effect_AR',replicate_id=rep,delta_macro_yield=(a[0]-b[0])-(c[0]-d[0]),delta_original_q20_LOW_yield=(a[1]-b[1])-(c[1]-d[1])))
    grouped=defaultdict(list)
    for r in contrasts:grouped[r['teacher'],r['comparison'],r['method'],r['reference']].append(r)
    contrast_summary=[]
    for key,rows in grouped.items():
        contrast_summary.append(dict(teacher=key[0],comparison=key[1],method=key[2],reference=key[3],independent_replicate_pairs=len(rows),mean_delta_macro_yield=mean([r['delta_macro_yield'] for r in rows]),descriptive_SD_delta_macro_yield=sd([r['delta_macro_yield'] for r in rows]),mean_delta_original_q20_LOW_yield=mean([r['delta_original_q20_LOW_yield'] for r in rows]),descriptive_SD_delta_original_q20_LOW_yield=sd([r['delta_original_q20_LOW_yield'] for r in rows])))
    for name,value in [('independent_metrics.csv',independent_cells),('independent_seed_metrics.csv',independent_seeds),('perfit_contrasts.csv',contrasts),('contrast_summary.csv',contrast_summary)]:write(OUT/name,value)
    result=dict(status='PASS',source_sha256=sha(__file__),evaluated_primary_attempts_per_teacher=44928,separate_diagnostic_attempts_per_teacher=4992,teachers=['uma','mace'],integer_boolean_checks=integer_checks,paired_energy_mask_checks=pair_checks,
        canonical_operations=192,all_outcome_rows_checked=99840,metrics_cells_checked=702,all_integer_counts_and_labels_exact=True,
        float_allowance='Only independently aggregated FP64 mean/SD/rates permit16eps*max(1,abs(a),abs(b)); per-row energies and bin labels are exact',
        fit_uncertainty='Descriptive fit/sampler replicate spread; no proposal-row p-values or equivalence claims',
        protected_assessment='No feedback, model selection, training or new scoring after FINAL outcomes',elapsed_seconds=time.monotonic()-started,
        collected_receipt_sha256=sha(F/'static_collected/receipt.json'),evaluation_receipt_sha256=sha(ev/'receipt.json'),history_sha256=sha(manifest['inputs']['history']['path']),new_NN_forwards=0,new_MLIP_calls=0,new_optimizer_updates=0)
    (OUT/'receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
