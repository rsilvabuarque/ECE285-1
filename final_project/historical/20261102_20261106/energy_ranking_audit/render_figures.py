"""Regenerate the two reference figures from saved predictions; no new inference."""
import json
import os
import pandas as pd
from analyze import HERE, reference_figure

if not os.environ.get('SLURM_JOB_ID') or not os.environ.get('SLURMD_NODENAME','').startswith('nid'):
    raise RuntimeError('Render in a compute allocation')
out=HERE/'results'
df=pd.read_csv(out/'frame_metrics.csv').drop_duplicates(['dataset','geometry_key'])
for name in ('historical','hu'):
    reference_figure(df[df.dataset==name].copy(),out,name,name=='historical')
(out/'render_receipt.json').write_text(json.dumps(dict(job_id=os.environ['SLURM_JOB_ID'],
    change='Colorbar moved outside data axes; identical saved predictions and scientific estimators',
    new_inference_calls=0),indent=2)+'\n')
print('Reference figures regenerated from saved data; no inference.')
