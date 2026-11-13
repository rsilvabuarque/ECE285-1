"""Sample a frozen 48-site vacancy decoder, retaining raw and projected counts."""
import argparse,json
from pathlib import Path
import numpy as np
import torch
from model import MaskCVAE
from sampling import top_count,conditional_count
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--checkpoint',type=Path,required=True);ap.add_argument('--n-li',type=int,required=True);ap.add_argument('--count',type=int,default=100);ap.add_argument('--condition',type=int,choices=[0,1],default=1);ap.add_argument('--sampling',choices=['topk','conditional'],default='conditional');ap.add_argument('--seed',type=int,default=20261106);ap.add_argument('--device',choices=['cpu','cuda'],default='cpu');ap.add_argument('--output',type=Path,required=True);a=ap.parse_args();assert 0<=a.n_li<=48
    torch.set_num_threads(4);cfg=json.loads((a.checkpoint.parent/'configuration.json').read_text());ck=torch.load(a.checkpoint,map_location=a.device,weights_only=False)
    model=MaskCVAE(condition_dim=cfg['condition_dim']).to(a.device);model.load_state_dict(ck['state_dict']);model.eval();rng=np.random.default_rng(a.seed)
    c=np.tile([a.n_li/48,a.condition],(a.count,1))[:,:cfg['condition_dim']].astype(np.float32);z=rng.normal(size=(a.count,8)).astype(np.float32)
    with torch.no_grad():logits=model.decode(torch.tensor(z,device=a.device),torch.tensor(c,device=a.device)).cpu().numpy()
    occupancy=(top_count if a.sampling=='topk' else conditional_count)(logits,a.n_li,rng)
    parent=Path(__file__).resolve().parents[1]/'data/parents/O3_N48_pilot.npz'
    with np.load(parent) as z:inv=np.argsort(z['permutations'],axis=1)
    bits=np.left_shift(np.uint64(1),np.arange(47,-1,-1,dtype=np.uint64));keys=[];witness=[]
    for o in occupancy:
        values=((1-o)[inv].astype(np.uint64)*bits).sum(1);j=int(values.argmin());keys.append(f'{int(values[j]):048b}');witness.append(j)
    np.savez_compressed(a.output,occupancy=occupancy,canonical_vacancy_mask=keys,canonicalizing_inverse_permutation_index=witness,raw_logits=logits,raw_expected_count=(1/(1+np.exp(-logits))).sum(1),n_Li=a.n_li,seed=a.seed)
    print('WROTE',a.output,'rows',len(occupancy),'fixed_count',a.n_li,'unique_symmetry_orbits',len(set(keys)),'(deduplication and fresh oracle evaluation remain separate)')
if __name__=='__main__':main()
