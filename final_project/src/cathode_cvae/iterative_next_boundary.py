"""Freeze the next development boundary using raw and assessed identities only.

All successfully generated orbits become known; only eligible assessment
candidates become newly protected. No score or energy member is read.
"""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from datetime import datetime,timezone
import numpy as np
from cathode_cvae.data import sha256
from cathode_cvae.iterative_acquisition import _canonical_vacancy,_parent
from cathode_cvae.iterative_boundary import binding,verify_boundary,write
from cathode_cvae.runtime import require_compute_allocation


def canonical_set(keys,perms):
    a=np.asarray([[int(c) for c in k] for k in sorted(keys)],dtype=np.uint8).reshape(-1,48)
    return set(_canonical_vacancy(1-a,perms).tolist())


def pool_keys(path,parent_hash,perms):
    with np.load(path,allow_pickle=False) as z:
        masks=z['masks'];ids=z['structure_ids'].astype(str)
        if str(z['parent_sha256'].item())!=parent_hash:raise ValueError('Candidate pool parent differs')
    if masks.shape!=(len(ids),48) or len(set(ids))!=len(ids) or not np.isin(masks,[0,1]).all():
        raise ValueError('Invalid candidate pool')
    keys=_canonical_vacancy(1-masks,perms)
    if any(''.join(map(str,mask))!=key or sid!=f'mask48_{int(key,2):012x}' for mask,key,sid in zip(masks,keys,ids)):
        raise ValueError('Candidate masks are not canonical or IDs disagree')
    return set(keys.tolist())


def read_rows(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))


def build(boundary_dir,h0_path,pairs,output_dir,known_raw=()):
    boundary,old_known,old_protected=verify_boundary(boundary_dir)
    parent_hash=boundary['parent']['sha256'];perms,_=_parent(boundary['parent']['path'])
    known=set(old_known);protected=set(old_protected)
    old_known_orbits=canonical_set(old_known,perms);old_protected_orbits=canonical_set(old_protected,perms)
    if not old_protected_orbits<=old_known_orbits:raise ValueError('Old protected orbit missing from history')
    # Historical H0 roles are inspected and bound, never rewritten or relabeled.
    if 'data' in boundary and boundary['data']['sha256']!=sha256(h0_path):
        raise ValueError('H0 differs from original boundary dataset')
    with np.load(h0_path,allow_pickle=False) as z:
        roles=z['split'].astype(str);h0keys=_canonical_vacancy(z['occupancy'],perms)
    if len(roles)!=len(h0keys) or set(roles)-{'train','val','test'}:raise ValueError('Invalid H0 roles')
    if not set(h0keys.tolist())<=old_known_orbits:raise ValueError('Old history omits H0 identities')
    if not set(h0keys[np.isin(roles,['val','test'])].tolist())<=old_protected_orbits:
        raise ValueError('Old boundary does not protect H0 VAL/TEST')
    if not pairs:raise ValueError('At least one raw and assessment-safe pair is required')
    sources=[];raw_union=set();eligible_union=set();seen=set()
    for raw_dir,safe_dir in pairs:
        raw=Path(raw_dir).resolve();safe=Path(safe_dir).resolve()
        if (str(raw),str(safe)) in seen:raise ValueError('Repeated proposal pair')
        seen.add((str(raw),str(safe)))
        rr=json.loads((raw/'receipt.json').read_text());sr=json.loads((safe/'receipt.json').read_text())
        if rr.get('status')!='FROZEN_BEFORE_SCORING' or not rr.get('no_energy_filtering') or sr.get('status')!='FROZEN_BEFORE_SCORING':
            raise ValueError('Raw or assessment-safe bundle is not frozen')
        binding(raw/'protocol.json',rr['protocol_sha256']);binding(safe/'protocol.json',sr['protocol_sha256'])
        if sr['original_generation_receipt']['sha256']!=sha256(raw/'receipt.json'):
            raise ValueError('Assessment-safe receipt does not bind this raw generation')
        if sr['boundary_receipt']['sha256']!=sha256(Path(boundary_dir)/'receipt.json'):
            raise ValueError('Assessment-safe bundle used another prior boundary')
        for name in ('generator_source','runtime_loader_source','adapter_source','manifest','boundary_receipt','original_generation_receipt'):
            binding(sr[name]['path'],sr[name]['sha256'])
        for name,digest in rr['outputs'].items():binding(raw/name,digest)
        for name,digest in sr['outputs'].items():binding(safe/name,digest)
        proto=json.loads((raw/'protocol.json').read_text())
        if proto['inputs']['parent']['sha256']!=parent_hash or proto['inputs']['history']['sha256']!=boundary['history']['sha256']:
            raise ValueError('Raw generation used another parent/history')
        if sha256(raw/'protocol.json')!=sha256(safe/'protocol.json'):
            raise ValueError('Safe and raw proposal protocols differ')
        if sha256(raw/'raw_attempts.npz')!=sha256(safe/'raw_attempts.npz'):
            raise ValueError('Safe bundle did not preserve raw arrays')
        rows=read_rows(raw/'attempts_all.csv');safe_rows=read_rows(safe/'attempts_all.csv')
        if (len(rows)!=rr['total_attempts'] or len(rows)!=sr['total_attempts'] or len(rows)!=len(safe_rows)
            or len({r['attempt_id'] for r in rows})!=len(rows)):
            raise ValueError('Raw and safe attempt identity/count differs')
        with np.load(raw/'raw_attempts.npz',allow_pickle=False) as z:occ=z['occupancy']
        if occ.shape!=(len(rows),48):raise ValueError('Raw occupancy shape differs')
        valid=[i for i,r in enumerate(rows) if r['status']=='generated']
        actual=_canonical_vacancy(occ[valid],perms);raw_keys=set(actual.tolist())
        for i,key in zip(valid,actual):
            r=rows[i];sid=f'mask48_{int(key,2):012x}'
            if (int(r['raw_row'])!=i or r['canonical_mask']!=key or r['scoring_structure_id']!=sid
                or int(r['n_target'])!=int(occ[i].sum())):raise ValueError('Raw orbit/count/ID differs')
        if raw_keys!=pool_keys(raw/'candidates.npz',parent_hash,perms):
            raise ValueError('Raw candidate pool omits or adds generated orbits')
        eligible=set()
        for i,(r,s) in enumerate(zip(rows,safe_rows)):
            if (r['attempt_id']!=s['attempt_id'] or r['raw_row']!=s['raw_row']
                or r['canonical_mask']!=s['canonical_mask'] or r['status']!=s['original_status']):
                raise ValueError('Safe attempt does not preserve raw identity/status')
            if s['status']=='generated':
                if r['status']!='generated' or s['boundary_status']!='eligible':raise ValueError('Invalid eligible assessment row')
                eligible.add(r['canonical_mask'])
            elif s.get('boundary_status')=='eligible':raise ValueError('Eligible row was not generated')
        safe_keys=pool_keys(safe/'candidates.npz',parent_hash,perms)
        if safe_keys!=eligible or not safe_keys<=raw_keys:
            raise ValueError('Safe candidate pool differs from eligible raw attempts')
        if safe_keys & old_known_orbits or safe_keys & old_protected_orbits:
            raise ValueError('Newly assessable candidates intersect prior known/protected orbits')
        raw_union.update(raw_keys);eligible_union.update(safe_keys)
        sources.append({'raw_receipt':binding(raw/'receipt.json'),'raw_protocol':binding(raw/'protocol.json'),
            'safe_receipt':binding(safe/'receipt.json'),'raw_candidates':binding(raw/'candidates.npz'),
            'safe_candidates':binding(safe/'candidates.npz'),'raw_attempts':binding(raw/'attempts_all.csv'),
            'safe_attempts':binding(safe/'attempts_all.csv'),'raw_orbits':len(raw_keys),'assessable_orbits':len(safe_keys)})
    for raw_dir in known_raw:
        raw=Path(raw_dir).resolve()
        if any(raw==Path(pair[0]).resolve() for pair in pairs):
            raise ValueError('The same raw bundle is already paired with an assessment')
        rr=json.loads((raw/'receipt.json').read_text())
        if rr.get('status')!='FROZEN_BEFORE_SCORING' or not rr.get('no_energy_filtering'):
            raise ValueError('Known-only raw generation is not frozen')
        binding(raw/'protocol.json',rr['protocol_sha256'])
        for name,digest in rr['outputs'].items():binding(raw/name,digest)
        proto=json.loads((raw/'protocol.json').read_text())
        if proto['inputs']['parent']['sha256']!=parent_hash or proto['inputs']['history']['sha256']!=boundary['history']['sha256']:
            raise ValueError('Known-only raw generation used another parent/history')
        binding(proto['manifest']['path'],proto['manifest']['sha256'])
        rows=read_rows(raw/'attempts_all.csv')
        if len(rows)!=rr['total_attempts'] or len({r['attempt_id'] for r in rows})!=len(rows):
            raise ValueError('Known-only attempt identity/count differs')
        with np.load(raw/'raw_attempts.npz',allow_pickle=False) as z:occ=z['occupancy']
        if occ.shape!=(len(rows),48):raise ValueError('Known-only raw occupancy shape differs')
        valid=[i for i,r in enumerate(rows) if r['status']=='generated']
        actual=_canonical_vacancy(occ[valid],perms);raw_keys=set(actual.tolist())
        for i,key in zip(valid,actual):
            r=rows[i];sid=f'mask48_{int(key,2):012x}'
            if (int(r['raw_row'])!=i or r['canonical_mask']!=key or r['scoring_structure_id']!=sid
                or int(r['n_target'])!=int(occ[i].sum())):raise ValueError('Known-only raw orbit/count/ID differs')
        if raw_keys!=pool_keys(raw/'candidates.npz',parent_hash,perms):
            raise ValueError('Known-only pool differs from raw generated orbits')
        raw_union.update(raw_keys)
        sources.append({'scope':'known_only','raw_receipt':binding(raw/'receipt.json'),
            'raw_protocol':binding(raw/'protocol.json'),'raw_candidates':binding(raw/'candidates.npz'),
            'raw_attempts':binding(raw/'attempts_all.csv'),'raw_orbits':len(raw_keys)})
    known.update(raw_union);protected.update(eligible_union);known.update(protected)
    out=Path(output_dir);out.mkdir(parents=True,exist_ok=False)
    write(out/'known_history.json',sorted(known));write(out/'protected.json',sorted(protected))
    h0_train=set(h0keys[roles=='train'].tolist())
    r={'status':'FROZEN_ITERATIVE_KEY_ONLY_BOUNDARY','created_utc':datetime.now(timezone.utc).isoformat(),
       'parent':boundary['parent'],'base_data':binding(h0_path),'previous_boundary_receipt':binding(Path(boundary_dir)/'receipt.json'),
       'history':binding(out/'known_history.json'),'protected':binding(out/'protected.json'),'proposal_sources':sources,
       'known_history_orbits':len(known),'protected_orbits':len(protected),'raw_union_orbits':len(raw_union),
       'newly_protected_assessable_orbits':len(eligible_union),'inherited_h0_train_protected_orbits':len(h0_train & old_protected_orbits),
       'newly_protected_h0_train_orbits':len(h0_train & eligible_union),
       'role_policy':'All H0 roles unchanged. Raw proposals are known; only previously unknown eligible assessment candidates are newly protected.',
       'source':binding(__file__),'no_protected_energy_members_read':True,'new_MLIP_calls':0}
    write(out/'receipt.json',r);return r


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--boundary-dir',type=Path,required=True);p.add_argument('--h0',type=Path,required=True)
    p.add_argument('--pair',action='append',required=True,help='RAW_GENERATION_DIR=ASSESSMENT_SAFE_DIR; repeat for all proposal bundles')
    p.add_argument('--known-raw',type=Path,action='append',default=[],help='Additional acquisition-generation directory; its identities become known only')
    p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();require_compute_allocation()
    pairs=[item.split('=',1) for item in a.pair]
    if any(len(pair)!=2 or not all(pair) for pair in pairs):raise ValueError('Each pair must have RAW=SAFE format')
    r=build(a.boundary_dir,a.h0,pairs,a.output_dir,a.known_raw)
    print(json.dumps({k:r[k] for k in ('status','raw_union_orbits','newly_protected_assessable_orbits','newly_protected_h0_train_orbits')},indent=2))
if __name__=='__main__':main()
