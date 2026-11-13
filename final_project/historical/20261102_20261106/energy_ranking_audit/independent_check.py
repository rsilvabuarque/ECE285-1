"""Independent standard-library check of the screenshot's cached energy ranks."""
import csv
import math
import statistics
from predict import SLIDE


def average_ranks(values):
    order=sorted(range(len(values)),key=lambda i:values[i]);out=[0.]*len(values);start=0
    while start<len(values):
        end=start+1
        while end<len(values) and values[order[end]]==values[order[start]]:end+=1
        for i in order[start:end]:out[i]=(start+1+end)/2
        start=end
    return out


def cached_reference():
    with SLIDE.open() as f:rows=list(csv.DictReader(f))
    result=[]
    for x in sorted({float(r['lithiation_pct']) for r in rows}):
        g=[r for r in rows if float(r['lithiation_pct'])==x]
        d=[float(r['vasp_energy_eV']) for r in g];u=[float(r['uma_energy_eV']) for r in g];n=len(g)
        dr,ur=average_ranks(d),average_ranks(u);k=math.ceil(.2*n)
        ds={i for i in range(n) if d[i]<=sorted(d)[k-1]};us={i for i in range(n) if u[i]<=sorted(u)[k-1]}
        pairs=[(i,j) for i in range(n) for j in range(i) if d[i]!=d[j] and u[i]!=u[j]]
        inv=sum((d[i]-d[j])*(u[i]-u[j])<0 for i,j in pairs)
        best=[i for i in range(n) if u[i]==min(u)]
        result.append(dict(lithiation_pct=x,n=n,rho=statistics.correlation(dr,ur),exact_order=dr==ur,
                           strict_pairs=len(pairs),inversions=inv,inversion_fraction=inv/len(pairs),
                           bottom20_precision=len(ds&us)/len(us),
                           selection_regret_meV_atom=(statistics.mean(d[i] for i in best)-min(d))/int(g[0]['natoms'])*1000))
    return result
