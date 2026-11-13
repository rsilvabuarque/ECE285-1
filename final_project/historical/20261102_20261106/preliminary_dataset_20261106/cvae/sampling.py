"""Exact fixed-count sampling under a factorized Bernoulli decoder."""
import numpy as np
def top_count(logits,n,rng):
    order=np.lexsort((rng.random(logits.shape),-logits),axis=1);out=np.zeros(logits.shape,np.uint8)
    np.put_along_axis(out,order[:,:n],1,axis=1);return out
def conditional_count(logits,n,rng,return_logZ=False):
    """p(o|sum(o)=n) proportional exp(sum_i logits_i*o_i); no Gumbel approximation."""
    logits=np.asarray(logits,float);b,N=logits.shape;assert 0<=n<=N
    z=np.full((b,N+1,n+1),-np.inf);z[:,N,0]=0.
    for i in range(N-1,-1,-1):
        z[:,i,0]=0.
        for k in range(1,min(n,N-i)+1):z[:,i,k]=np.logaddexp(z[:,i+1,k],logits[:,i]+z[:,i+1,k-1])
    out=np.zeros((b,N),np.uint8);need=np.full(b,n,int)
    for i in range(N):
        ids=np.flatnonzero(need>0)
        if not len(ids):break
        k=need[ids];prob=np.exp(logits[ids,i]+z[ids,i+1,k-1]-z[ids,i,k]);take=(rng.random(len(ids))<np.clip(prob,0,1)) | (k==N-i)
        selected=ids[take];out[selected,i]=1;need[selected]-=1
    assert not need.any() and (out.sum(axis=1)==n).all()
    return (out,z[:,0,n]) if return_logZ else out
