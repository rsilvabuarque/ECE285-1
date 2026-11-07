"""Observed unpaired cross-seed ranges for both declared temperature transports.

For observed values a_i (reference arrangement) and b_j (other arrangement),
the range is [min(b)-max(a), max(b)-min(a)]. All b_j-a_i combinations share the
same small set of trajectories: four combinations are not four independent
replicates or a confidence interval. No thermodynamic populations are computed.
"""
import argparse,csv,hashlib,json
from pathlib import Path
import numpy as np

SOURCE=Path(__file__).read_bytes()

def values_summary(reference,candidate):
    a=np.asarray(reference,float);b=np.asarray(candidate,float)
    if not len(a) or not len(b) or not np.isfinite(np.r_[a,b]).all():raise ValueError('Need finite observed repeat values')
    differences=(b[:,None]-a[None,:]).ravel()
    low=float(b.min()-a.max());high=float(b.max()-a.min())
    assert np.isclose(differences.min(),low,rtol=0,atol=1e-10)
    assert np.isclose(differences.max(),high,rtol=0,atol=1e-10)
    return dict(observed_min_delta_eV=low,observed_max_delta_eV=high,
                difference_of_observed_means_eV=float(b.mean()-a.mean()),
                reference_trajectories=len(a),candidate_trajectories=len(b),
                correlated_cross_combinations=len(differences),
                observed_range_contains_zero=bool(low<=0<=high),
                observed_cross_differences_change_sign=bool(low<0<high))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--transport-csv',type=Path)
    p.add_argument('--reference-basin',default='basin_1');p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--selftest',action='store_true');a=p.parse_args()
    a.output_dir.mkdir(parents=True,exist_ok=False);(a.output_dir/'analysis_source.py').write_bytes(SOURCE)
    if a.selftest:
        result=values_summary([0.,2.],[1.,3.]);assert result['observed_min_delta_eV']==-1 and result['observed_max_delta_eV']==3
        assert result['observed_cross_differences_change_sign'] and np.array_equal(np.array([1.,3.])-np.array([0.,2.]),[1.,1.])
        receipt=dict(status='PASS',synthetic=True,test='Paired gaps both+1 but unpaired observed range[-1,+3]',result=result)
    else:
        if a.transport_csv is None:raise ValueError('Provide --transport-csv')
        with a.transport_csv.open() as f:records=list(csv.DictReader(f))
        groups={}
        for r in records:
            key=(r['native_file'],r['scope'],r['mode'],r['treatment'],r['target_temperature_K'])
            group=groups.setdefault(key,{})
            bucket=group.setdefault(r['basin_id'],{})
            if r['repeat_id'] in bucket:raise ValueError('Repeated trajectory identity in a transport stratum')
            bucket[r['repeat_id']]=r
        summary=[];cross=[]
        for (native,scope,mode,treatment,target),by_basin in groups.items():
            if a.reference_basin not in by_basin:raise ValueError('Reference arrangement absent')
            reference=by_basin[a.reference_basin]
            for basin,candidate in by_basin.items():
                if basin==a.reference_basin:continue
                for transport,field in [('direct_first_order','F_direct_first_order_eV'),('harmonic_reference_residual_first_order','F_harmonic_reference_transport_eV')]:
                    info=dict(native_file=native,scope=scope,mode=mode,treatment=treatment,target_temperature_K=float(target),transport=transport,reference_basin_id=a.reference_basin,candidate_basin_id=basin)
                    reference_values=[float(r[field]) for r in reference.values()];candidate_values=[float(r[field]) for r in candidate.values()]
                    summary.append({**info,**values_summary(reference_values,candidate_values)})
                    for reference_repeat,r in reference.items():
                        for candidate_repeat,c in candidate.items():
                            cross.append(dict(**info,reference_repeat_id=reference_repeat,candidate_repeat_id=candidate_repeat,delta_eV=float(c[field])-float(r[field]),independent_replicate=False))
        def write(name,rows):
            with (a.output_dir/name).open('w') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        if not summary:raise ValueError('Need at least two observed arrangements')
        write('observed_unpaired_ranges.csv',summary);write('correlated_cross_combinations.csv',cross)
        receipt=dict(status='CALCULATED_OBSERVED_CROSS_SEED_RANGES',strata=len(summary),cross_combinations=len(cross),transport_csv=str(a.transport_csv),transport_csv_sha256=hashlib.sha256(a.transport_csv.read_bytes()).hexdigest(),limitations=['Observed min-max sensitivity over the available trajectories, not a confidence interval','Cross combinations reuse trajectories and are not independent replicates','No validated equilibrium or thermodynamic weight inferred','A sign change means the observed ordering depends on cross-seed pairing; no sign change is not a population-level guarantee'])
    receipt['source_sha256']=hashlib.sha256(SOURCE).hexdigest()
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status=receipt['status'],output=str(a.output_dir))))

if __name__=='__main__':main()
