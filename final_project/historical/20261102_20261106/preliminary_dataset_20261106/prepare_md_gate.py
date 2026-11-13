"""Freeze paired-timestep assays from physically converged model-specific cells."""
import hashlib,json,os
from pathlib import Path
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
cfg=json.loads((HERE/'config.json').read_text());rows=[json.loads(f.read_text()) for f in (HERE/'data/confirm').glob('*/*.json')];assert len(rows)==100 and all(r['converged'] for r in rows)
tasks=[]
for m in cfg['models']:
    for n in [24,36]:
        r=next(r for r in rows if r['model']==m and r['n_Li']==n);path=HERE/'data/confirm'/m/(r['task_id']+'.npz')
        tasks.append(dict(task_id=f'md_gate__{m}__{r["structure_id"]}',stage='md_gate',protocol='gate',model=m,structure_id=r['structure_id'],parent=r['parent'],n_Li=n,seed_npz=str(path),seed_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),seed=20261106+n,warm_ps=1.,nve_ps=2.))
for stage,ts in [('md_gate',tasks),('md_smoke',[dict(tasks[0],task_id='md_smoke__uma',stage='md_smoke',warm_ps=.01,nve_ps=.02)])]:
    (HERE/'data'/f'{stage}_tasks.jsonl').write_text(''.join(json.dumps(t)+'\n' for t in ts))
design={'hypothesis':'Conservativefoundationmodels plusresolvedtimestep give bounded NVEenergy error beforethermaldata generation.','controls':'Twoindependent model-specific relaxedmaskstates permodel,493.15K;1psBussiwarm,cloneexactstate,paired1fs/.5fs2psNVE; fixedCOM3N-3 inbothbranches.','physicalunits':'ASEinternal time; dt_fs*ase.units.fs. Velocities exported asget_velocities()*ase.units.fs inA/fs. PressuresMPa*1e6/1.602176634e11 eV/A3.','provisional_numerical_gate':'NVEtotalenergyrange andenddrift peratom <=1%ofthermalkineticenergy peratom(3N-3)kBT/(2N) over2ps; reportdriftvsdt andFP32repeats. Thisisobservable-basednumericaltarget, notuniversalMDstandard.','dt_selection':'Use1fs ifbothstatespassboth1/.5fs branches; otherwise.5fsifbothhalfstepstatespass; otherwiseholdmodelMDanddiagnose.','resourceguards':'Volume0.25-4xseed,principalstretches0.25-3,nearestseparation>=0.6A; broadcomputationalguardsnotphysicalphasevalidation. Retainfailedstates.','npt_convention':'IndependentunconstrainedMaxwellinitializationperT/P,3NDOFs;1psBussiwarmthenMTKNPTwithtdamp100fs,pdamp1000fs; noCOMprojection orcallbackstateedits. Measuretotalpressureincludingkineticstress.','production_proposal':'20independentparents,5T/Pconditions,UMA+MACE,5psNPTafter1pswarm; preliminarygeometrylabels, notequilibriumthermodynamicclaims.','MD_targets':cfg['MD_targets'],'cost_estimate_four_gpu_node_hours':'~0.6summedtasktimeforNVEgatesbeforeloading; initial4hallocationscombined36nodehmax, wholeobjectivecap200.','source_pressure_bug':'HistoricalTorchSimscreenunitpathmissing1e6; newcodeusesdirectSIconversion. HistoricalLAMMPSbarconversioncorrect; do notattributealloldfailures tothisbug.'}
(HERE/'data/md_design.json').write_text(json.dumps(design,indent=2)+'\n')
print('MD_GATES_PREPARED',len(tasks),flush=True)
