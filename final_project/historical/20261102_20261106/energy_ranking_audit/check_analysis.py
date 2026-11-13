"""Small independent examples for the tie-aware ordering and regret estimators."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
from analyze import inversion_counts, group_metrics

examples = [([0,1,2],[0,1,2],(0,3)),
            ([0,1,2],[2,1,0],(3,3)),
            ([0,0,1],[0,1,0],(1,1)),
            ([0,1,2,3],[0,2,1,3],(1,6))]
for dft,mlip,expected in examples:
    assert inversion_counts(np.array(dft),np.array(mlip))[:2] == expected
g=pd.DataFrame(dict(dft_energy_eV=[0.,2.,4.],uma_energy_eV=[0.,0.,4.],n_Co=[1]*3,n_atoms=[1]*3))
assert group_metrics(g)['selection_regret_meV_atom'] == 1000.
result=dict(status='PASS',checks=['identity','complete reversal','ties in either ranking',
                                'one inversion','uniform selection among tied minima'])
(Path(__file__).parent/'results/unit_validation.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result))
