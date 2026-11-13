"""Independent requirement checks and compact delivery package after all labels exist."""
import csv,hashlib,json,os,subprocess,tarfile,time
from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from package_core import build_package
HERE=Path(__file__).resolve().parent;PACK=HERE/'collaborator_package'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,d):
    path=HERE/name;tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(d,indent=2)+'\n');tmp.replace(path)
def main():
    assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
    checks=[]
    for name in ['masks','scf_identity','calibration','relaxations_logical','generated','diagnostics','graph_symmetry_smoke','portable','portable_gpu','decoder_latency','diversity','relaxation_objective','md_gate','md_production_gate','md_export','MD_mapping_review','core_archive_content']:
        path=HERE/'validation'/f'{name}.json';d=json.loads(path.read_text());assert d.get('status') in ['PASS','COMPLETE'],(name,d.get('status'));checks.append(dict(check=name,receipt=str(path.relative_to(HERE)),sha256=sha(path)))
    mainparents={}
    with np.load(HERE/'data/parents/O3_N48_pilot.npz') as z:parent={k:z[k] for k in z.files}
    splitmap={r['structure_id']:r['split'] for r in csv.DictReader((HERE/'splits/lineage_splits.csv').open())}
    framecounts={};weights={};localized={}
    for model in ['uma','mace','sevennet','chgnet','m3gnet']:
        for cohort in ['fixed','variable']+(['md'] if model in ['uma','mace'] else []):
            path=HERE/'data'/f'geometry_{model}_{cohort}.npz'
            with np.load(path) as z:
                assert np.array_equal(z['presence'][:,parent['li_indices']].sum(1),z['n_Li'])
                assert (z['forces_parent_eV_A'][~z['presence']]==0).all() and (z['displacement_A'][~z['presence']]==0).all()
                assert np.isfinite(z['displacement_A']).all() and np.isfinite(z['forces_parent_eV_A']).all()
                assert all(splitmap[str(s)]==sp for s,sp in zip(z['structure_ids'],z['split']))
                for sid in np.unique(z['structure_ids']):
                    total=float(z['frame_weight'][z['structure_ids']==sid].sum());assert abs(total-1)<1e-5,(model,cohort,sid,total)
                if cohort=='md':
                    assert np.array_equal(z['ancestral_occupancy'].astype(bool),z['presence'][:,parent['li_indices']]) and (z['observed_occupancy'].sum(1)==z['n_Li']).all()
                framecounts[f'{model}:{cohort}']=len(z['structure_ids']);localized[f'{model}:{cohort}']=int(z['localized'].sum())
    assert sum(v for k,v in framecounts.items() if not k.endswith(':md'))==54810
    assert sum(v for k,v in framecounts.items() if k.endswith(':md'))==4000
    checks.append(dict(check='all_derived_frames',frames=framecounts,localized=localized,scope='Presenceghostmasking,finiteU/F,frozenlineages,perparent/model/cohorttotalweight1; MDancestral/currentoccupancyseparate.'))
    with np.load(HERE/'validation/portable_cpu_smoke/final_package_samples.npz') as z:
        assert z['occupancy'].shape==(100,48) and (z['occupancy'].sum(1)==24).all();assert len(set(z['canonical_vacancy_mask']))==100
    for name in ['uma_cvae_main_20261106','uma_cvae_beta01_main_20261106']:
        assert sha(PACK/'checkpoints'/name/'best.pt')==sha(HERE/'cvae/runs'/name/'best.pt')
    for name in ['cvae_20261400','regressor_20261400']:assert sha(PACK/'checkpoints'/name/'best.pt')==sha(HERE/'cvae/geometry_runs'/name/'best.pt')
    assert (PACK/'requirements.txt').read_text().splitlines()==['numpy>=1.26','torch>=2.2']
    checks.append(dict(check='final_portable_package',CPU_samples=100,unique_orbits=100,Li_count=24,checkpoint_copies='ExactSHA256match;4frozenfits',scope='CPU/GPUtwoepochoperability, notscientificconvergence'))
    # Per-frame archive errors retain source/cohort; raw energy offsets are not calibrated away.
    manifest=list(csv.DictReader((HERE/'data/phase_context_manifest.csv').open()));phase=[]
    with np.load(HERE/'data/phase_context_candidates.npz') as z:p={k:z[k] for k in z.files}
    for model in ['uma','mace','sevennet','chgnet','m3gnet']:
        tasks=[t for t in map(json.loads,(HERE/'data/phase_context_tasks.jsonl').read_text().splitlines()) if t['model']==model]
        for t in tasks:
            with np.load(HERE/'data/phase_context'/model/(t['task_id']+'.npz')) as z:
                for j,i in enumerate(z['indices']):
                    keep=p['atom_presence'][i];N=int(keep.sum());df=z['forces_parent_eV_A'][j,keep]-p['dft_forces_parent_eV_A'][i,keep];ds=(z['stress_eV_A3'][j]-p['dft_stress_eV_A3'][i])*160.21766208
                    phase.append(dict(manifest[i],teacher=model,raw_energy_error_meV_atom=float((z['energy_eV'][j]-p['dft_energy_eV'][i])/N*1000),force_component_RMSE_eV_A=float(np.sqrt((df*df).mean())),stress_component_RMSE_GPa=float(np.sqrt((ds*ds).mean()))))
    with (HERE/'reports/phase_context_per_frame.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(phase[0]));w.writeheader();w.writerows(phase)
    # Short-block means are plotted as diagnostics, without equilibrium error-bar claims.
    md=list(csv.DictReader((HERE/'reports/MD_trajectory_summary.csv').open()));fig,axs=plt.subplots(2,2,figsize=(10,7),sharex=True)
    for col,model in enumerate(['uma','mace']):
        for T in sorted({float(r['target_temperature_K']) for r in md}):
            rr=[r for r in md if r['model']==model and float(r['target_temperature_K'])==T];ns=sorted({int(r['n_Li']) for r in rr})
            values=[[r for r in rr if int(r['n_Li'])==n] for n in ns]
            axs[0,col].plot(np.array(ns)/48,[np.mean([float(r['mean_temperature_K'])-T for r in a]) for a in values],'.-',label=f'{T:.0f}K')
            axs[1,col].plot(np.array(ns)/48,[np.mean([float(r['mean_total_pressure_MPa'])-float(r['target_pressure_MPa']) for r in a]) for a in values],'.-')
        axs[0,col].set_title(model);axs[0,col].set_ylabel('Last-half mean T − target (K)');axs[1,col].set_ylabel('Last-half mean P − target (MPa)');axs[1,col].set_xlabel('Li fraction');axs[0,col].legend(fontsize=7)
        for ax in axs[:,col]:ax.axhline(0,color='gray',ls='--');ax.grid(alpha=.2)
    fig.suptitle('5 ps NPT exploratory traces: four parent masks per composition\nNo established equilibration or independent block uncertainty',fontsize=11);fig.tight_layout();fig.savefig(HERE/'reports/MD_operating_points.pdf');fig.savefig(HERE/'reports/MD_operating_points.png',dpi=170);plt.close(fig)
    archive=build_package()
    resource=json.loads((HERE/'resources.json').read_text())
    ids={r['job_id'] for key in ['initial_allocations','subsequent_allocations'] for r in resource[key]}
    assert {r['job_id'] for r in resource['last_observed']['jobs']}==ids
    assert resource['last_observed']['allocated_four_gpu_node_equivalent_hours']<=resource['gpu_node_hour_limit']
    delivery=dict(status='PASS',scope='AuthorizedovernightpreliminaryMLIPdatasetandinitialCVAEexploration;noDFTsubmitted.',checks=checks,archive=str(archive),archive_bytes=archive.stat().st_size,archive_sha256=sha(archive),thermal_lineages=20,thermal_trajectories=200,thermal_frames=4000,relaxation_frames=54810,limitations=['MLIPoraclesnotnewDFTtruth','Noexhaustivegroundstateguarantee','Fixed48sitearchitecture,notmixedsizetransfer','ShortMDnotadequateequilibriumthermodynamics','GeometryCVAEhasnomatchedadvantageoverregressor','MaskCVAEdoesnotbeatstrongmutationbaseline'],remaining='Build/checkfinalPDF; obtainfinalterminalallocationaccounting; rootreviewbeforegoalcompletion.')
    save('validation/requirement_audit.json',delivery);print('REQUIREMENT_AUDIT_PASS',archive.stat().st_size,flush=True)
if __name__=='__main__':main()
