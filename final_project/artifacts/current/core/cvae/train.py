"""Reproducible grouped-mask CVAE/VAE training; conditions fitted on train only."""
import argparse,csv,hashlib,json,math,os,random,time
from pathlib import Path
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
import numpy as np
import torch
from model import MaskCVAE,loss_terms
HERE=Path(__file__).resolve().parent;ROOT=HERE.parent
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--teacher',default='uma');ap.add_argument('--kind',choices=['cvae','composition','vae'],default='cvae');ap.add_argument('--seed',type=int,default=20261106);ap.add_argument('--epochs',type=int,default=150);ap.add_argument('--split',choices=['main','composition'],default='main');ap.add_argument('--data-root',type=Path,default=Path(__file__).resolve().parents[1]);ap.add_argument('--output-dir',type=Path,required=True);ap.add_argument('--device',choices=['auto','cpu','cuda'],default='auto');ap.add_argument('--threads',type=int,default=4);ap.add_argument('--beta-final',type=float,default=1.);a=ap.parse_args();global ROOT;ROOT=a.data_root.resolve();device=('cuda' if torch.cuda.is_available() else 'cpu') if a.device=='auto' else a.device;assert device!='cuda' or torch.cuda.is_available()
    torch.set_num_threads(a.threads);torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=False;torch.use_deterministic_algorithms(True)
    random.seed(a.seed);np.random.seed(a.seed);torch.manual_seed(a.seed);torch.cuda.manual_seed_all(a.seed) if device=='cuda' else None
    out=a.output_dir;out.mkdir(parents=True,exist_ok=False)
    source=ROOT/'data'/f'learning_O3_N48_pilot_{a.teacher}.npz'
    with np.load(source) as z:data={k:z[k] for k in z.files}
    split_path=ROOT/'splits/lineage_splits.csv';splits={r['structure_id']:r['split' if a.split=='main' else 'composition_split'] for r in csv.DictReader(split_path.open())}
    labels=np.array([splits[s] for s in data['structure_ids']]);train=np.flatnonzero(labels=='train');val=np.flatnonzero(labels=='val');test=np.flatnonzero(labels=='test')
    assert len(train)>1000 and len(val)>100
    E=data['energy_eV']/data['n_atoms'];nLi=data['n_Li'];condition_stats={};low=np.zeros(len(E),np.float32)
    for n in sorted(set(nLi[train])):
        values=E[train[nLi[train]==n]];q=float(np.sort(values)[math.ceil(.2*len(values))-1]);condition_stats[int(n)]=dict(n_train=len(values),mean_eV_atom=float(values.mean()),std_eV_atom=float(values.std()),low_quantile_eV_atom=q,low_fraction_train=float((values<=q).mean()))
        low[nLi==n]=(E[nLi==n]<=q)
    conditions=np.stack([nLi/48,low],axis=1).astype(np.float32)
    if a.kind=='composition':conditions=conditions[:,:1]
    if a.kind=='vae':conditions=conditions[:,:0]
    x=torch.tensor(data['occupancy'],dtype=torch.float32,device=device);c=torch.tensor(conditions,device=device)
    with np.load(ROOT/'data/parents/O3_N48_pilot.npz') as z:inv=torch.tensor(np.argsort(z['permutations'],axis=1),device=device)
    model=MaskCVAE(condition_dim=conditions.shape[1]).to(device);opt=torch.optim.Adam(model.parameters(),lr=1e-3)
    train_t=torch.tensor(train,device=device);val_t=torch.tensor(val,device=device)
    val_noise=torch.randn((4,len(val),8),device=device);history=[];best=float('inf');best_epoch=-1;started=time.perf_counter()
    meta=dict(teacher=a.teacher,kind=a.kind,seed=a.seed,split=a.split,train_rows=len(train),val_rows=len(val),test_rows=len(test),condition_dim=conditions.shape[1],latent_dim=8,n_sites=48,hidden_encoder=[128,64],hidden_decoder=[64,128],beta_final=a.beta_final,beta_anneal_epochs=20,batch_size=512,learning_rate=.001,logvar_numerical_clip=[-12,8],augmentation='uniform parent site permutation on train rows only, same source-row weight',energy_class='bottom20%oftrainingenergieswithincomposition; exacttiesretained',occupancy_convention='1=presentLi',projection='top-nLi occupancy logits; constrainedMAP, not Bernoulli sampling',source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),split_sha256=hashlib.sha256(split_path.read_bytes()).hexdigest(),condition_stats=condition_stats,torch=torch.__version__,gpu=torch.cuda.get_device_name() if device=='cuda' else 'CPU',dtype='float32',tf32=False)
    (out/'configuration.json').write_text(json.dumps(meta,indent=2)+'\n')
    for epoch in range(a.epochs):
        model.train();perm=train_t[torch.randperm(len(train_t),device=device)];tot=np.zeros(3);nb=0;beta=a.beta_final*min(1.,(epoch+1)/20)
        for ids in perm.split(512):
            xx=x[ids];ops=inv[torch.randint(len(inv),(len(ids),),device=device)];xx=xx.gather(1,ops)
            logits,mu,logvar=model(xx,c[ids]);recon,kl=loss_terms(logits,xx,mu,logvar);loss=(recon+beta*kl).mean()
            opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),10.);opt.step()
            tot+=np.array([loss.item(),recon.mean().item(),kl.mean().item()])*len(ids);nb+=len(ids)
        model.eval()
        with torch.no_grad():
            xv,cv=x[val_t],c[val_t];mu,logvar=model.encode(xv,cv);reconstruction=0.
            for eps in val_noise:
                logits=model.decode(mu+torch.exp(.5*logvar)*eps,cv);rec,kl=loss_terms(logits,xv,mu,logvar);reconstruction+=float(rec.mean())/4
            val_kl=float(kl.mean());val_elbo=reconstruction+val_kl
        row=dict(epoch=epoch+1,beta=beta,train_objective=float(tot[0]/nb),train_reconstruction=float(tot[1]/nb),train_KL=float(tot[2]/nb),val_negative_ELBO=val_elbo,val_reconstruction=reconstruction,val_KL=val_kl);history.append(row)
        if epoch+1>=min(20,a.epochs) and val_elbo<best-1e-4:
            best=val_elbo;best_epoch=epoch+1;torch.save({'state_dict':model.state_dict(),'configuration':meta,'epoch':epoch+1,'val_negative_ELBO':best},out/'best.pt')
        if (epoch+1)%10==0:print('EPOCH',row,flush=True)
        if epoch+1>=40 and epoch+1-best_epoch>=25:break
    checkpoint=torch.load(out/'best.pt',map_location=device,weights_only=False);model.load_state_dict(checkpoint['state_dict']);model.eval()
    with torch.no_grad():
        mu,logvar=model.encode(x[val_t],c[val_t]);logits=model.decode(mu,c[val_t]);pred=torch.zeros_like(logits)
        for n in sorted(set(nLi[val])):
            selected=np.flatnonzero(nLi[val]==n);ii=torch.tensor(selected,device=device);indices=logits[ii].topk(int(n),dim=1).indices;block=torch.zeros_like(logits[ii]);block.scatter_(1,indices,1.);pred[ii]=block
        hamming=float((pred!=x[val_t]).float().mean());raw_count_error=float((logits.sigmoid().round().sum(1)-torch.tensor(nLi[val],device=device)).abs().mean());variance=mu.var(0).cpu().tolist();kl_per_dim=(.5*(mu.square()+logvar.exp()-1-logvar)).mean(0).cpu().tolist()
    summary=dict(best_epoch=best_epoch,best_val_negative_ELBO=best,val_projected_hamming_fraction=hamming,val_unprojected_mean_absolute_Li_count_error=raw_count_error,posterior_mu_variance_per_dimension=variance,mean_KL_per_dimension=kl_per_dim,training_seconds=time.perf_counter()-started,completed_epochs=len(history),checkpoint_sha256=hashlib.sha256((out/'best.pt').read_bytes()).hexdigest(),generation_quality='Requires independent frozenMLIP scoring; reconstruction/ELBO do notprove discovery utility.')
    with (out/'history.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(history[0]));w.writeheader();w.writerows(history)
    (out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print('TRAINING_COMPLETE',out,summary,flush=True)
if __name__=='__main__':main()
