"""Check that the training export preserves every selected coordinate and label."""
import os
for key in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[key]='1'
import numpy as np
import map_frames as m

if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Use compute allocation')
with np.load(m.AUDIT/'reference_frames.npz') as z:ref={k:z[k] for k in z.files}
with np.load(m.HERE/'joint_training_candidates.npz') as z:out={k:z[k] for k in z.files}
ix=out['reference_index']
atoms=np.concatenate([np.arange(ref['offsets'][i],ref['offsets'][i+1]) for i in ix])
tests={}
for dest,source in [('numbers','numbers'),('positions_A','positions'),('dft_forces_eV_A','forces')]:
    tests[dest]=bool(np.array_equal(out[dest],ref[source][atoms]))
for dest,source in [('cell_A','cell'),('dft_energy_eV','energy'),('dft_stress_eV_A3','stress')]:
    tests[dest]=bool(np.array_equal(out[dest],ref[source][ix]))
assert all(tests.values())
assert len(ix)==6481 and len(set(ix))==len(ix)
m.dump(m.HERE/'export_validation.json',dict(status='PASS',n_frames=len(ix),n_atoms=len(atoms),exact_equal_checks=tests))
print('EXACT_EXPORT_CHECK_PASS',len(ix),len(atoms),flush=True)
