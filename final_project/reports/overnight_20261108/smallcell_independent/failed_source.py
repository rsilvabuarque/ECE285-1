"""Independent raw geometry/probability checks; qualify ranks by approximate orbit.

Does not call either production thermodynamic routine or the canonical-key
helper. Floating comparisons are roundoff checks, not physical tolerances.
"""
import argparse
import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr,kendalltau
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation

KB=8.617333262145e-5


def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--prepared',required=True,type=Path)
    p.add_argument('--scores',required=True,type=Path);p.add_argument('--analysis',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path);args=p.parse_args();require_compute_allocation()
    args.output.mkdir(parents=True,exist_ok=False);protocol=json.loads((args.prepared/'protocol.json').read_text())
    parent=load(protocol['parent']['path']);states=load(args.prepared/'states.npz');x=states['occupancy'];groups=states['canonical_orbit_keys'].astype(str)
    indices=parent['li_indices'];inverses=np.argsort(parent['permutations'],axis=1);checks=0;maxdifference=0.
    def check(actual,expected):
        nonlocal checks,maxdifference
        a=np.asarray(actual,float);b=np.asarray(expected,float);difference=float(np.max(np.abs(a-b)))
        tolerance=128*np.finfo(float).eps*max(1,float(np.max(np.abs(b))))
        if difference>tolerance:raise AssertionError(f'Arithmetic mismatch {difference:g}>{tolerance:g}')
        checks+=a.size;maxdifference=max(maxdifference,difference)
    for n,total in [(3,220),(6,924),(9,220)]:
        actual={tuple(row) for row in x[x.sum(1)==n]};expected=set()
        for ii in itertools.combinations(range(12),n):
            row=[0]*12
            for i in ii:row[i]=1
            expected.add(tuple(row))
        if actual!=expected or len(actual)!=total:raise AssertionError('Raw combinatorial completeness mismatch')
    for i,row in enumerate(x):
        mask=1-row;expected=min(''.join(map(str,mask[inv].tolist())) for inv in inverses)
        if expected!=groups[i]:raise AssertionError('Independent approximate-orbit key mismatch')
        keep=np.ones(48,bool);keep[indices[row==0]]=False;h=hashlib.sha256()
        for value in [parent['numbers'][keep].astype('<i8'),parent['positions_A'][keep].astype('<f8'),parent['cell_A'].astype('<f8'),np.ones(3,np.uint8)]:h.update(value.tobytes())
        if h.hexdigest()!=str(states['geometry_sha256'][i]):raise AssertionError('Literal raw geometry hash mismatch')
    e={m:np.full(len(states['label_ids']),np.nan) for m in ['uma','mace']};seen={m:set() for m in e};sources=[]
    for file in args.scores.rglob('chunk_*.npz'):
        metadata=json.loads(file.with_suffix('.json').read_text());m=metadata['binding']['model']
        if sha256(file)!=metadata['sha256']:raise AssertionError('Raw teacher score chunk changed')
        z=load(file)
        for j,label in enumerate(z['label_index']):
            label=int(label)
            if label in seen[m] or str(z['status'][j])!='success':raise AssertionError('Duplicate or failed raw label')
            if str(z['label_id'][j])!=str(states['label_ids'][label]):raise AssertionError('Actual label ID mismatch')
            seen[m].add(label);e[m][label]=z['energy_eV'][j]
        sources.append(dict(path=str(file.resolve()),sha256=sha256(file)))
    for m in e:
        if len(seen[m])!=1376 or not np.isfinite(e[m]).all():raise AssertionError('Missing fresh literal-state labels')
    reported=list(csv.DictReader((args.analysis/'thermodynamics.csv').open()))
    probabilities=load(args.analysis/'raw_probabilities.npz');representatives=json.loads((args.prepared/'representatives.json').read_text())
    for r in reported:
        m=r['teacher'];n=int(r['n_Li']);T=float(r['temperature_K']);ii=np.flatnonzero(x.sum(1)==n);local=e[m][ii]
        weight=np.exp(-(local-local.min())/(KB*T));Z=weight.sum();probability=weight/Z
        entropy=-KB*np.dot(probability[probability>0],np.log(probability[probability>0]));mean=np.dot(probability,local)
        free=local.min()-KB*T*np.log(Z)
        check(float(r['mean_energy_eV_cell']),mean);check(float(r['free_energy_eV_cell']),free)
        check(float(r['configurational_entropy_eV_K_cell']),entropy);check(free,mean-T*entropy)
        check(probabilities[f'{m}_n{n}_T{str(T).replace(".","p")}'],probability)
        local_keys=groups[ii];rep_raw=[];conditional_entropy=0.;P=[];g=[]
        for key in sorted(set(local_keys)):
            member=np.flatnonzero(local_keys==key);mass=probability[member].sum();P.append(mass);g.append(len(member))
            conditional=probability[member]/mass
            conditional_entropy+=mass*(-KB*np.dot(conditional[conditional>0],np.log(conditional[conditional>0])))
            raw_id=representatives[key]['raw_id'];j=int(np.flatnonzero(states['raw_ids'][ii].astype(str)==raw_id)[0])
            rep_raw.extend([local[j]]*len(member))
        P=np.asarray(P);g=np.asarray(g);entropy_of_orbits=-KB*np.dot(P[P>0],np.log(P[P>0]))
        check(entropy,entropy_of_orbits+conditional_entropy)
        rep_raw=np.asarray(rep_raw);w=np.exp(-(rep_raw-rep_raw.min())/(KB*T));rp=w/w.sum()
        rs=-KB*np.dot(rp[rp>0],np.log(rp[rp>0]));rf=rep_raw.min()-KB*T*np.log(w.sum());re=np.dot(rp,rep_raw)
        check(float(r['representative_entropy_error_eV_K']),rs-entropy)
        check(float(r['representative_free_energy_error_eV']),rf-free);check(float(r['representative_mean_energy_error_eV']),re-mean)
    rank=[]
    for n in [3,6,9]:
        ii=np.flatnonzero(x.sum(1)==n);a=e['uma'][ii];b=e['mace'][ii];key=groups[ii];unique=sorted(set(key))
        ma=np.array([a[key==k].mean() for k in unique]);mb=np.array([b[key==k].mean() for k in unique])
        ia,ja=np.triu_indices(len(ii),1);inter=key[ia]!=key[ja];within=~inter
        da=np.sign(a[ia]-a[ja]);db=np.sign(b[ia]-b[ja])
        rank.append(dict(n_Li=n,raw_states=len(ii),approximate_orbits=len(unique),
            raw_Spearman=float(spearmanr(a,b).statistic),raw_Kendall_tau_b=float(kendalltau(a,b).statistic),
            mean_orbit_Spearman=float(spearmanr(ma,mb).statistic),mean_orbit_Kendall_tau_b=float(kendalltau(ma,mb).statistic),
            UMA_lowest_mean_orbit=unique[int(ma.argmin())],MACE_lowest_mean_orbit=unique[int(mb.argmin())],
            inter_orbit_raw_pair_sign_agreement=float(np.mean(da[inter]==db[inter])),
            within_orbit_raw_pair_sign_agreement=float(np.mean(da[within]==db[within])),
            UMA_raw_energy_range_eV_cell=float(np.ptp(a)),MACE_raw_energy_range_eV_cell=float(np.ptp(b))))
    with (args.output/'rank_scope.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rank[0]));w.writeheader();w.writerows(rank)
    receipt=dict(status='PASS',source_sha256=sha256(__file__),independent_raw_geometry_checks=1364,
        fresh_labels_per_teacher=1376,arithmetic_field_checks=checks,maximum_absolute_float_difference=maxdifference,
        roundoff_check='128*float64_eps*max(1,absolute expected value); engineering arithmetic check, not a physical tolerance',
        production_thermodynamic_or_canonical_helper_calls=0,source_chunks=sources,
        outputs={'rank_scope.csv':sha256(args.output/'rank_scope.csv')},
        limitations='Exact static finite ensemble per teacher/count only; different lowest mean-energy orbit for every count demonstrates teacher disagreement, not DFT error.')
    (args.output/'receipt.json').write_text(json.dumps(receipt,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:v for k,v in receipt.items() if k not in ['source_chunks','outputs']},indent=2))


if __name__=='__main__':main()
