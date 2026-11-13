"""Additional cheap timestep checks cover the planned production composition extremes."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
tasks=[]
for m in ['uma','mace']:
    rows=[json.loads(p.read_text()) for p in (HERE/'data/confirm'/m).glob('*.json')]
    for n in [12,46]:
        r=next(r for r in rows if r['n_Li']==n);assert r['converged'];path=HERE/'data/confirm'/m/(r['task_id']+'.npz')
        tasks.append(dict(task_id=f"md_extra_gate__{m}__{r['structure_id']}",stage='md_extra_gate',protocol='gate',model=m,structure_id=r['structure_id'],parent=r['parent'],n_Li=n,seed_npz=str(path),seed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=20261206+n,warm_ps=.5,nve_ps=1.))
(HERE/'data/md_extra_gate_tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in tasks))
(HERE/'data/md_extreme_design.json').write_text(json.dumps({'reason':'OriginalNVEgatescoverx.5/.75; productionalsoincludes.25/.875/.958. Addworst-Ttestsat.25/.958 beforeclaimingproductionrangecoverage.','change':'FourindependentextraUMA/MACEseedcases,0.5pswarm+paired1psNVE, samephysicalcriterionanddamping; preserveoriginalgate.','cost':'~14000systemsteps,wellbelow1GPU-nodehbeforeloading','decision':'Choose1fsforproductiononlyifbothoriginalandextreme1fs/halfbranchespass; otherwiseuse.5fsifallhalfbranchespass,holdifnot.'},indent=2)+'\n')
print('EXTRA_GATES_PREPARED',len(tasks),flush=True)
