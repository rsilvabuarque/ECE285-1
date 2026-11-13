"""Independent high-precision spots and a cancellation-free Taylor envelope."""
from pathlib import Path
from decimal import Decimal,localcontext
import json,hashlib
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
require_compute_allocation();import argparse
parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--run-root',type=Path,required=True);args=parser.parse_args();p=args.run_root.resolve();rows=[]
for n,name in enumerate(['D2','Sampled auxiliary lambda10','Mean auxiliary lambda10']):
 with np.load(p/f'fixed_condition_embedding/embedding_{n}.npz',allow_pickle=False) as z:mu=z['posterior_mu'].astype(float);lv=z['posterior_logvar'].astype(float)
 delta=lv[:,None,:]-lv[None,:,:];dm=mu[:,None,:]-mu[None,:,:];stable=.5*np.sum(np.expm1(delta)-delta+dm**2*np.exp(-lv[None,:,:]),axis=-1)
 # Taylor's theorem: exp(d)-1-d <= d² exp(|d|)/2. This variance
 # envelope uses no subtraction of nearly equal floating-point numbers.
 envelope=.5*np.sum(.5*delta**2*np.exp(np.abs(delta))+dm**2*np.exp(-lv[None,:,:]),axis=-1)
 pairs=[(0,0),(0,1),(int(mu[:,0].argmin()),int(mu[:,0].argmax())),(int(lv[:,0].argmin()),int(lv[:,0].argmax()))];checks=[]
 for i,j in pairs:
  with localcontext() as ctx:
   ctx.prec=80;value=Decimal(0)
   for k in range(mu.shape[1]):
    d=Decimal.from_float(float(lv[i,k]))-Decimal.from_float(float(lv[j,k]));m=Decimal.from_float(float(mu[i,k]))-Decimal.from_float(float(mu[j,k]));value+=(d.exp()-Decimal(1)-d+m*m*(-Decimal.from_float(float(lv[j,k]))).exp())/Decimal(2)
   reference=float(value)
  checks.append(dict(i=i,j=j,Decimal80_reference=str(value),float64_stable=float(stable[i,j]),absolute_error=abs(float(stable[i,j])-reference),float64_Taylor_envelope=float(envelope[i,j])))
 rows.append(dict(model=name,maximum_abs_logvariance_difference=float(np.abs(delta).max()),mean_pair_KL_float64=float(stable.mean()),mean_pair_Taylor_upper_envelope_float64=float(envelope.mean()),spot_checks=checks,interpretation='Use rounded order of magnitude, not certifiedlastdigits. Taylor envelope is an independent analytic upper expression for the exact variance term; Decimal80spotsvalidate implementation nearzero.'))
receipt=dict(status='CHECKS_COMPLETED',source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),rows=rows,scope='No arbitrary relative-pass tolerance imposed on nearzero KL. The mathematicalJensen/Taylor inequalities are exact; displayed averages use floating-point arithmetic, not intervalcertifiedbounds. Scientificinterpretation uses orders of magnitude and fixedfiniteuniformVALscope.')
(p/'posterior_information_bound_highprecision_validation.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2),flush=True)
