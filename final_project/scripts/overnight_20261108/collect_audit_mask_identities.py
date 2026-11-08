"""Inventory only saved actual audit masks; no sampling or energy arrays."""
import json,hashlib,re
from pathlib import Path
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
BASE=Path(__file__).resolve().parent
SOURCES=[BASE/'trained_initial_audit',BASE/'screen_controls_dev_v1/trained_checkpoint_audit',
    BASE/'confirmation_controls_dev_v1/trained_checkpoint_audit',BASE/'final_readiness_audit_v1/final_only8',
    BASE/'final_readiness_audit_v1/paired_deletion3',BASE/'final_readiness_audit_v1/paired_endpoint3']
PARENT_SHA='ae8f8bc8d7e767165b0172b396e3c723b15e09f4b79429ac01b7c78fd4741e95'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    require_compute_allocation();out=BASE/'audit_mask_identities_v1';out.mkdir()
    occupancy=[];source_npz=[];source_key=[];source_row=[];origins=[];checkpoints=[];inventory=[]
    for stem in SOURCES:
        npz=stem.with_suffix('.npz');metadata=stem.with_suffix('.json');r=json.loads(metadata.read_text());digest=sha(npz)
        if r['status']!='PASS' or r['parent_sha256']!=PARENT_SHA or r['raw_sha256']!=digest:raise ValueError('Unverified or different-parent saved audit')
        with np.load(npz,allow_pickle=False) as z:
            keys=sorted([k for k in z.files if re.fullmatch(r'model\d+_sample',k)],key=lambda k:int(k.split('_')[0][5:]))
            if len(keys)!=len(r['models']):raise ValueError('Checkpoint result/sample arrays are not aligned')
            details=[]
            for key,model in [('occupancy',None),*zip(keys,r['models'])]:
                x=z[key]
                if x.shape!=(9,48) or not np.isin(x,[0,1]).all():raise ValueError('Unexpected saved audit occupancy')
                if model is not None:
                    if model['status']!='PASS' or not model['reproducible'] or model['encoder_calls']!=0:raise ValueError('Incomplete or failed recorded sample probe')
                    index=key.split('_')[0];counts=np.rint(z[index+'_conditions'][:,0]*48).astype(int)
                    if not np.array_equal(x.sum(1),counts):raise ValueError('Saved sample lost its hard-count condition')
                origin='saved_completed_prior_draw' if model is not None else 'saved_validation_probe_input'
                for i,row in enumerate(x):
                    occupancy.append(row.astype(np.uint8));source_npz.append(str(npz.resolve()));source_key.append(key);source_row.append(i);origins.append(origin)
                    checkpoints.append('' if model is None else model['checkpoint'])
                details.append(dict(key=key,shape=list(x.shape),origin=origin,checkpoint=None if model is None else model['checkpoint']))
        inventory.append(dict(path=str(npz.resolve()),sha256=digest,receipt_path=str(metadata.resolve()),receipt_sha256=sha(metadata),arrays=details))
        if sha(npz)!=digest:raise ValueError('Saved audit changed during identity-only extraction')
    x=np.asarray(occupancy,np.uint8)
    if len(x)!=486 or origins.count('saved_completed_prior_draw')!=432:raise ValueError('Not all48 checkpoint instances and six input arrays were collected')
    output=out/'occupancy.npz'
    np.savez_compressed(output,occupancy=x,n_target=x.sum(1).astype(np.int16),parent_sha256=np.asarray(PARENT_SHA),
        source_npz=np.asarray(source_npz),source_key=np.asarray(source_key),source_row=np.asarray(source_row,np.int16),
        mask_origin=np.asarray(origins),checkpoint=np.asarray(checkpoints))
    receipt=dict(status='SAVED_AUDIT_IDENTITIES_ONLY',output=dict(path=str(output.resolve()),sha256=sha(output),rows=len(x)),
        parent_sha256=PARENT_SHA,checkpoint_instances=48,saved_generated_sample_rows=432,saved_input_rows=54,
        counts_preserved=True,sources=inventory,source_sha256=sha(__file__),energy_fields_read=0,new_model_calls=0,
        new_MLIP_calls=0,new_optimizer_updates=0,new_generation_performed=False,
        repeated_second_draw_scope='Second sampling call was verified exactly equal to the saved first call in every PASS audit; no additional distinct identity follows.',
        future_permutation_probe_masks_serialized=False,
        future_probe_limit='The helper flipped x[:, i:] at sites0/12/24/36/47 and saved gradients/invariance booleans but not the altered occupancy arrays. No future-flip identities were reconstructed or invented here.',
        no_more_sampling_audits_after_this_inventory=True)
    (out/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({k:receipt[k] for k in ['status','output','saved_generated_sample_rows','saved_input_rows','future_permutation_probe_masks_serialized']},indent=2))
if __name__=='__main__':main()
