"""Promote the previously assessed pool to adaptation data, freeze a new audit boundary."""
import csv, hashlib, json, os
from pathlib import Path
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def load(path):
    with np.load(path) as z:return {k:z[k] for k in z.files}
def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    original=load(ROOT/'data/learning_O3_N48_pilot_uma.npz')
    parent=load(ROOT/'data/parents/O3_N48_pilot.npz')
    prior=load(ROOT/'data/generated_scalar_uma.npz')
    generated=load(ROOT/'data/generated_candidates.npz')
    manifest={r['structure_id']:r for r in csv.DictReader((ROOT/'data/mask_manifest.csv').open()) if r['parent']=='O3_N48_pilot'}
    splits={r['structure_id']:r['split'] for r in csv.DictReader((ROOT/'splits/lineage_splits.csv').open())}
    assert np.array_equal(generated['structure_ids'],prior['structure_ids'])
    ids=[];keys=[];occupancy=[];energies=[];split=[];source=[]
    for i,sid in enumerate(original['structure_ids']):
        if int(original['n_Li'][i]) not in [12,24,36]:continue
        key=manifest[str(sid)]['canonical_mask']
        ids.append(str(sid));keys.append(key);occupancy.append(1-np.array(list(key),np.uint8));energies.append(float(original['energy_eV'][i]/original['n_atoms'][i]));split.append(splits[str(sid)]);source.append('original')
    baseline_count=len(ids)
    for i,sid in enumerate(prior['structure_ids']):
        if int(prior['n_Li'][i]) not in [12,24,36]:continue
        key=format(int(str(sid).split('_')[1],16),'048b')
        fraction=int(hashlib.sha256(('adaptation_split_v1:'+key).encode()).hexdigest()[:12],16)/16**12
        label='train' if fraction<.8 else 'val' if fraction<.9 else 'test'
        ids.append(str(sid));keys.append(key);occupancy.append(1-np.array(list(key),np.uint8));energies.append(float(prior['energy_eV'][i]/prior['n_atoms'][i]));split.append(label);source.append('prior_assessment_now_adaptation')
    assert len(keys)==len(set(keys))
    x=np.array(occupancy,np.uint8);n=x.sum(1).astype(int);split=np.array(split);source=np.array(source)
    common=dict(occupancy=x,n_Li=n,energy_eV_atom=energies,split=split,weight=np.ones(len(x),np.float32),structure_ids=ids,canonical_masks=keys,source=source)
    np.savez_compressed(HERE/'training_round0.npz',**common)
    np.savez_compressed(HERE/'training_original.npz',**{k:np.array(v)[:baseline_count] for k,v in common.items()})
    known={r['canonical_mask'] for r in manifest.values()}|{format(int(str(s).split('_')[1],16),'048b') for s in prior['structure_ids']}
    (HERE/'known_keys_round0.json').write_text(json.dumps({'canonical_masks':sorted(known),'scope':'All initial catalog plus all previously scored proposals, including adaptation validation/test; frozen before new generation.'})+'\n')
    cache={}
    for teacher in ['uma','mace']:
        old=load(ROOT/f'data/learning_O3_N48_pilot_{teacher}.npz');new=load(ROOT/f'data/generated_scalar_{teacher}.npz')
        values={manifest[str(sid)]['canonical_mask']:float(old['energy_eV'][i]) for i,sid in enumerate(old['structure_ids'])}
        values.update({format(int(str(sid).split('_')[1],16),'048b'):float(new['energy_eV'][i]) for i,sid in enumerate(new['structure_ids'])})
        kk=sorted(values);np.savez_compressed(HERE/f'data/cache_{teacher}.npz',canonical_masks=kk,structure_ids=[f'mask48_{int(k,2):012x}' for k in kk],energy_eV=[values[k] for k in kk],n_atoms=[192-k.count('1') for k in kk])
    selected={}
    for teacher in ['uma','mace']:
        old=load(ROOT/f'data/learning_O3_N48_pilot_{teacher}.npz')
        selected[teacher]={}
        for count in [12,24,36]:
            ii=[i for i,s in enumerate(old['structure_ids']) if int(old['n_Li'][i])==count and splits[str(s)]=='train']
            vv=old['energy_eV'][ii]/old['n_atoms'][ii];q=np.sort(vv)[int(np.ceil(.2*len(vv)))-1]
            selected[teacher][str(count)]=[manifest[str(old['structure_ids'][i])]['canonical_mask'] for i,e in zip(ii,vv) if e<=q]
    (HERE/'data/original_low_training_seeds.json').write_text(json.dumps(selected)+'\n')
    receipt={'training_rows':len(x),'original_rows':baseline_count,'split_counts':{s:int((split==s).sum()) for s in sorted(set(split))},'training_sha256':sha(HERE/'training_round0.npz'),'original_sha256':sha(HERE/'training_original.npz'),'known_keys':len(known),'known_keys_sha256':sha(HERE/'known_keys_round0.json'),'sources':{s:int((source==s).sum()) for s in sorted(set(source))},'scope':'Prior 54,752-orbit oracle pool is now adaptation data, no longer final assessment. Original held-out splits preserved; fresh final proposals will be scored after freezing models.'}
    (HERE/'data/preparation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt),flush=True)
if __name__=='__main__':main()
