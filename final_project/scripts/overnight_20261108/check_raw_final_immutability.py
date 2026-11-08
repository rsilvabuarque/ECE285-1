"""Post-audit byte bindings only; no array members or model forwards."""
import json,hashlib,shutil
from pathlib import Path
from cathode_cvae.runtime import require_compute_allocation
ROOT=Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285_overnight_20261108/final_project');F=ROOT/'runs/overnight_20261108/final';OUT=F/'independent_raw_audit'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    require_compute_allocation();plan=json.loads((F/'paired_plan/plan.json').read_text());pairs=[(F/'static_generated',F/'static_manifest.json')]
    pairs += [(F/'paired_generated/generated'/name,Path(s['manifest']['path'])) for name,s in plan['stages'].items()]
    checked={};checkpoint_checks=0
    for generated,manifestpath in pairs:
        manifest=json.loads(manifestpath.read_text());protocol=json.loads((generated/'protocol.json').read_text());receipt=json.loads((generated/'receipt.json').read_text())
        assert sha(manifestpath)==protocol['manifest']['sha256']
        assert sha(generated/'protocol.json')==receipt['protocol_sha256']
        for name,digest in receipt['outputs'].items():assert sha(generated/name)==digest;checked[str(generated/name)]=digest
        for key,b in manifest['inputs'].items():
            assert sha(b['path'])==b['sha256'];assert b['sha256']==protocol['inputs'][key]['sha256'];checked[b['path']]=b['sha256']
        for entry in manifest['models']:
            if entry.get('checkpoint'):
                assert sha(entry['checkpoint'])==entry['expected_checkpoint_sha256'];checkpoint_checks+=1
                checked[entry['checkpoint']]=entry['expected_checkpoint_sha256']
        checked[str(manifestpath)]=sha(manifestpath)
    source=Path(__file__).with_name('check_raw_final.py');audit=json.loads((OUT/'receipt.json').read_text());assert sha(source)==audit['source_sha256']
    target=OUT/'executed_checker.py'
    if target.exists():assert target.read_bytes()==source.read_bytes()
    else:shutil.copy2(source,target)
    result=dict(status='PASS',generated_stages=5,checkpoint_binding_checks=checkpoint_checks,checked_files=checked,
        scope='Byte hashes checked again after the independent raw audit; manifests, parent, data, schemas, history, checkpoint files and proposal artifacts match their frozen bindings',
        source_sha256=sha(__file__),energy_fields_read=0,new_NN_forwards=0,new_generation=0,new_MLIP_calls=0,new_optimizer_updates=0)
    (OUT/'post_audit_immutability.json').write_text(json.dumps(result,indent=2)+'\n');print('PASS post-audit immutable byte bindings',len(checked))
if __name__=='__main__':main()
