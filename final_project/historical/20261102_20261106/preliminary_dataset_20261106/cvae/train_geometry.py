"""Factorized joint prototype p(occupancy|c) p(displacement|occupancy,cell,c)."""
import argparse,csv,hashlib,json,os,random,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from geometry_model import DisplacementModel
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--kind',choices=['cvae','regressor'],default='cvae');ap.add_argument('--seed',type=int,default=20261400);ap.add_argument('--epochs',type=int,default=150);a=ap.parse_args()
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid') and torch.cuda.is_available()
    torch.set_num_threads(4);torch.backends.cuda.matmul.allow_tf32=False;torch.use_deterministic_algorithms(True);random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed)
    source=ROOT/'data/geometry_uma_fixed.npz'
    with np.load(source) as z:d={k:z[k] for k in z.files}
    keep=d['localized'];d={k:v[keep] for k,v in d.items()};N=192;cohorts=['deletion','fixed_intermediate','fixed_endpoint'];cohort=np.array([cohorts.index(s) for s in d['cohort']])
    train=np.flatnonzero(d['split']=='train');val=np.flatnonzero(d['split']=='val');assert len(set(d['structure_ids'][train]) & set(d['structure_ids'][val]))==0
    presence=np.repeat(d['presence'],3,axis=1).astype(np.float32);u=d['displacement_A'].reshape(-1,576).astype(np.float32);weights=d['frame_weight'].astype(np.float32)
    scale=float(np.sqrt((weights[train,None]*u[train]**2*presence[train]).sum()/(weights[train,None]*presence[train]).sum()));assert scale>0
    raw=np.stack([d['energy_eV']/d['presence'].sum(1),np.log1p(d['force_component_rms_eV_A']),d['configurational_pressure_GPa'],np.log1p(d['deviatoric_stress_GPa'])],axis=1)
    stats={};props=np.zeros_like(raw);target_conditions={}
    for n in sorted(set(d['n_Li'])):
        for cidx,cname in enumerate(cohorts):
            ids=np.flatnonzero((d['n_Li']==n)&(cohort==cidx)&(d['split']=='train'));assert len(ids)
            mean=raw[ids].mean(0);std=np.maximum(raw[ids].std(0),1e-8);key=f'{int(n)}|{cname}';stats[key]=dict(mean=mean.tolist(),std=std.tolist(),train_frames=len(ids),train_parents=len(set(d['structure_ids'][ids])))
            apply=(d['n_Li']==n)&(cohort==cidx);props[apply]=(raw[apply]-mean)/std
            if cname=='fixed_endpoint':
                candidates=ids[d['force_component_rms_eV_A'][ids]<=np.median(d['force_component_rms_eV_A'][ids])];chosen=candidates[np.argmin(d['energy_eV'][candidates])]
                target_conditions[int(n)]=dict(raw=raw[chosen].tolist(),standardized=props[chosen].tolist(),source_training_parent=str(d['structure_ids'][chosen]),rule='AttainedtrainingendpointwithlowestenergyamongforceRMS-below-medianendpoints; noheldoutoracleconditionused.')
    ctx=np.concatenate([d['occupancy'],(d['n_Li']/48)[:,None],np.eye(3)[cohort],d['deformation'].reshape(-1,9),props],axis=1).astype(np.float32);assert ctx.shape[1]==65
    device='cuda';U=torch.tensor(u/scale,device=device);C=torch.tensor(ctx,device=device);P=torch.tensor(presence,device=device);W=torch.tensor(weights/weights[train].mean(),device=device)
    model=DisplacementModel(kind=a.kind).to(device);optimizer=torch.optim.Adam(model.parameters(),lr=1e-3);ids_train=torch.tensor(train,device=device);ids_val=torch.tensor(val,device=device)
    out=HERE/'geometry_runs'/f'{a.kind}_{a.seed}';out.mkdir(parents=True,exist_ok=False);history=[];best=float('inf');best_epoch=0;started=time.perf_counter()
    metadata=dict(kind=a.kind,seed=a.seed,teacher='uma',source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),model_source_sha256=hashlib.sha256((HERE/'geometry_model.py').read_bytes()).hexdigest(),trainer_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),displacement_scale_A=scale,condition_stats=stats,target_conditions=target_conditions,context_order=['occupancy48','nLi/48','cohort_onehot3','deformation_rowmajor9','energy_peratom_z','log1p(forceRMS/1eVA)_z','pressure_GPa_z','log1p(devstress/1GPa)_z'],cohorts=cohorts,train_parents=len(set(d['structure_ids'][train])),val_parents=len(set(d['structure_ids'][val])),geometry_frame='Canonicalparentframe; affine/globaltranslationremoved, noLi-onlysymmetryaugmentation',likelihood='IndependentGaussianinmaskednormalizeddisplacement, fixedphysicalscale; MSE/2summedoverpresentcoordinates. ELBOomitsconstantsharedacrossfits.',scope='Fixedcellprototype; suppliedcell, notpredictedlattice. Conditionsaretrain-onlyanddonotguaranteezeroforce/stress.')
    (out/'configuration.json').write_text(json.dumps(metadata,indent=2)+'\n');eps=torch.randn((len(val),16),device=device)
    for epoch in range(a.epochs):
        model.train();order=ids_train[torch.randperm(len(train),device=device)];tot=0.;kl_total=0.;beta=min(1.,(epoch+1)/20)
        for ids in order.split(128):
            pred,mu,lv=model(U[ids],C[ids]);recon=.5*((pred-U[ids]).square()*P[ids]).sum(-1);kl=.5*(mu.square()+lv.exp()-1-lv).sum(-1) if mu is not None else torch.zeros_like(recon)
            loss=((recon+beta*kl)*W[ids]).mean();optimizer.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);optimizer.step();tot+=float(loss)*len(ids);kl_total+=float(kl.mean())*len(ids)
        model.eval()
        with torch.no_grad():
            if a.kind=='cvae':mu,lv=model.encode(U[ids_val],C[ids_val]);pred=model.decode(mu+torch.exp(.5*lv)*eps,C[ids_val]);kl=.5*(mu.square()+lv.exp()-1-lv).sum(-1)
            else:pred=model.decode(None,C[ids_val]);kl=torch.zeros(len(val),device=device)
            rec=.5*((pred-U[ids_val]).square()*P[ids_val]).sum(-1);vl=float(((rec+kl)*W[ids_val]).sum()/W[ids_val].sum());rmse=float(torch.sqrt(((pred-U[ids_val]).square()*P[ids_val]*W[ids_val,None]).sum()/(P[ids_val]*W[ids_val,None]).sum())*scale)
        row=dict(epoch=epoch+1,train_objective=tot/len(train),train_KL=kl_total/len(train),val_objective=vl,val_coordinate_RMSE_A=rmse);history.append(row)
        if epoch+1>=20 and vl<best-1e-4:best=vl;best_epoch=epoch+1;torch.save({'state_dict':model.state_dict(),'configuration':metadata,'epoch':epoch+1},out/'best.pt')
        if (epoch+1)%10==0:print('GEOMETRY_EPOCH',a.kind,row,flush=True)
        if epoch+1>=50 and epoch+1-best_epoch>=25:break
    with (out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    (out/'summary.json').write_text(json.dumps(dict(best_epoch=best_epoch,best_val_objective=best,training_seconds=time.perf_counter()-started,epochs=len(history),validation_note='Posteriorconditionedcoordinatefit; actualprior-decodedEFSandrelaxationsavingsmustbeevaluatedseparately.'),indent=2)+'\n');print('GEOMETRY_TRAINED',out,flush=True)
if __name__=='__main__':main()
