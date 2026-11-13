"""Synthetic six-site enumeration verifies the conditional sampler's normalization."""
import itertools,json,os
from pathlib import Path
import numpy as np
from sampling import conditional_count
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
rng=np.random.default_rng(83);logit=np.array([-.8,.2,1.1,-.4,.7,-.1]);masks=np.array([a for a in itertools.product([0,1],repeat=6) if sum(a)==3]);weights=np.exp(masks@logit);prob=weights/weights.sum()
draws,logz=conditional_count(np.tile(logit,(20000,1)),3,rng,True)
assert np.max(abs(logz-np.log(weights.sum())))<1e-12
observed=np.array([(draws==v).all(axis=1).sum() for v in masks]);bounds=5*np.sqrt(20000*prob*(1-prob))+2
assert (abs(observed-20000*prob)<=bounds).all()
out={'status':'PASS','synthetic':True,'sites':6,'count':3,'draws':20000,'log_normalizer_error':float(np.max(abs(logz-np.log(weights.sum())))),'maximum_standardized_frequency_error':float(np.max(abs(observed-20000*prob)/np.sqrt(20000*prob*(1-prob)))),'meaning':'Exactnormalizeragainstexplicit20-stateenumeration; seededMonteCarlofrequencycheck5sigma+2countbounds, notmaterialdatasetperformance.'}
(Path(__file__).resolve().parent/'sampling_validation.json').write_text(json.dumps(out,indent=2)+'\n');print(out,flush=True)
