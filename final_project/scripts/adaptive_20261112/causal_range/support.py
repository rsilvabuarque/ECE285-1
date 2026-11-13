from pathlib import Path
import importlib.util,json
import numpy as np
C=Path(__file__).parent;BASE=C.parent/'study_v2'
spec=importlib.util.spec_from_file_location('core',BASE/'orchestrate.py');d=importlib.util.module_from_spec(spec);spec.loader.exec_module(d)
d.require_compute_allocation()
with np.load(d.H0,allow_pickle=False) as z:ii=np.flatnonzero(z['split'].astype(str)=='train');nn=z['n_Li'][ii];ee=z['energy_eV_atom'][ii]
rows=[]
for family in d.FAMILIES:
 for seed in d.SEEDS:
  schema=d.load_schema(BASE/'round1/schemas'/f'r1_{family}_range_{seed}'/'condition_schema.json')
  for count in [12,24,36]:
   labels=np.searchsorted(schema['compositions'][str(count)]['edges_eV_atom'],ee[nn==count],side='left')
   rows.append(dict(family=family,fit_seed=seed,n_target=count,low_rows=int((labels==0).sum()),middle_rows=int((labels==1).sum()),high_rows=int((labels==2).sum())))
d.ai.write_csv(C/'H0_support_under_R1_edges.csv',rows)
d.write(C/'H0_support_receipt.json',dict(status='ACTUAL_H0_TRAIN_SUPPORT_AUDITED',data=d.binding(d.H0),support=d.binding(C/'H0_support_under_R1_edges.csv'),source=d.binding(__file__),all_cells_nonempty=all(min(r[k] for k in ['low_rows','middle_rows','high_rows'])>0 for r in rows),no_TEST_classification=True))
print('ALL_CELLS_NONEMPTY',all(min(r[k] for k in ['low_rows','middle_rows','high_rows'])>0 for r in rows),flush=True)
