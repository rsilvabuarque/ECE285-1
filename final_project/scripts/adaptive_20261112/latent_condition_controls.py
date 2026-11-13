from pathlib import Path
import hashlib,json
import numpy as np
import torch
from cathode_cvae.runtime import require_compute_allocation
from cathode_cvae.learning.runtime import trusted_checkpoint
require_compute_allocation();import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();b=p.parent;rows=[];sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
for w in [10,100]:
 for seed in range(2026111301,2026111306):
  base=b/'initial_fits'/f'd2_{seed}'/'last_update.pt';new=p/'fits'/f'lambda{w}_condition_d2_{seed}'/'last_update.pt';x=trusted_checkpoint(base,'cpu');y=trusted_checkpoint(new,'cpu');sx,sy=x['state_dict'],y['state_dict'];same=sx.keys()==sy.keys() and all(torch.equal(sx[k],sy[k]) for k in sx)
  dr=p/'diagnostics'/f'd2_{seed}'/'raw_intervention_proposals.npz';nr=p/'diagnostics'/f'lambda{w}_condition_d2_{seed}'/'raw_intervention_proposals.npz'
  with np.load(dr,allow_pickle=False) as z:raw={k:z[k] for k in z.files}
  with np.load(nr,allow_pickle=False) as z:matched=all(np.array_equal(raw[k],z[k]) for k in raw)
  rows.append(dict(weight=w,seed=seed,generator_state_exact=same,raw_masks_and_streams_exact=matched,checkpoints=[dict(path=str(q),sha256=sha(q)) for q in [base,new]],raw_bindings=[dict(path=str(q),sha256=sha(q)) for q in [dr,nr]]))
receipt=dict(status='PASS' if all(x['generator_state_exact'] and x['raw_masks_and_streams_exact'] for x in rows) else 'DIFFERENCES_REQUIRING_INVESTIGATION',rows=rows,source_sha256=sha(__file__),scope='Full4000-update equality and all12672rawattempts perfit; conditionhead gradient cannot alter generator even understrength10/100')
(p/'condition_control_equality.json').write_text(json.dumps(receipt,indent=2)+'\n');print(receipt['status'],flush=True)
