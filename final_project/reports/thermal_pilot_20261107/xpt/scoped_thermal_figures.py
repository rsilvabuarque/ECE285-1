"""Scoped plots of completed saved analyses; no MLIP, MD, or xPT rerun.

Full20ps data contain assignment events. First10ps were selected after that
audit and are exploratory, not independently established equilibrium samples.
Observed cross ranges reuse2+2trajectories; they are not confidence intervals.
"""
from pathlib import Path
import csv,hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
def read(p):
    with Path(p).open() as f:return list(csv.DictReader(f))
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def spectrum(row):
    p=Path(row['native_prefix']).parent/'spectrum.npz'
    with np.load(p,allow_pickle=False) as d:return {k:d[k] for k in d.files}
def save(fig,folder,name):
    fig.savefig(folder/(name+'.pdf'));fig.savefig(folder/(name+'.png'),dpi=180);plt.close(fig)
def write(p,rows):
    with Path(p).open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)

events_path=ROOT.parent/'numerics/md_audit_events.json'
event_audit=json.loads(events_path.read_text());assert event_audit['primary_runs_initial_assignments_stable_through10ps'] is True
events={r['identity']:r['phases']['production']['first_atom_assignment_change_ps'] for r in event_audit['runs']}
sources=[events_path]

# All20ps remains a mixed/event spectral diagnostic, never a basin F plot.
folder=ROOT/'full20_diagnostic';sources.append(folder/'native_results.csv')
rows=[r for r in read(folder/'native_results.csv') if r['mode']=='3PT']
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for r in rows:
    d=spectrum(r);identity=r['basin_id']+'/'+r['repeat_id'];event=events[identity]
    suffix='no observed event' if event is None else f'event {event:.3f} ps'
    label=f"start {r['basin_id'][-1]}, {r['repeat_id']}: {suffix}"
    axes[0].plot(d['frequency_THz'],d['dos_per_cm_cell']/.0299792458,label=label,linewidth=.8)
    axes[1].plot(d['lag_ps'],d['normalized_VACF'],label=label,linewidth=.8)
axes[0].set(xlim=(0,25),xlabel='Frequency (THz)',ylabel='Modes per THz per cell')
axes[1].set(xlim=(0,1),xlabel='Lag time (ps)',ylabel='Mass-weighted VACF / VACF(0)')
axes[0].legend(fontsize=7);[ax.grid(alpha=.2) for ax in axes]
fig.suptitle('Full 20 ps: two trajectories contain persistent Li-site assignment events\nMixed/event diagnostics only; these are not initial-basin vibrational free energies',fontsize=11)
fig.tight_layout(rect=[0,0,1,.86]);save(fig,folder,'full20_assignment_event_scoped')

# Two declared target-T transports and all observed cross-seed combinations.
folder=ROOT/'prefix10_analysis';ranges_path=ROOT/'prefix10_cross_seed_ranges/observed_unpaired_ranges.csv';cross_path=ROOT/'prefix10_cross_seed_ranges/correlated_cross_combinations.csv'
sources.extend([ranges_path,cross_path,folder/'difference_summary.csv'])
ranges=read(ranges_path);cross=read(cross_path);reference=read(folder/'difference_summary.csv')[0]
fig,axes=plt.subplots(1,2,figsize=(11,5),sharey=True)
transports=[('direct_first_order','Direct linear transport'),('harmonic_reference_residual_first_order','Exact harmonic + linear residual')]
for ax,treatment in zip(axes,['classical','quantum']):
    for j,(transport,label) in enumerate(transports):
        for i,mode in enumerate(['1PT','2PT','3PT']):
            row=next(r for r in ranges if r['treatment']==treatment and r['transport']==transport and r['mode']==mode)
            x=i+(j-.5)*.24;low=float(row['observed_min_delta_eV']);high=float(row['observed_max_delta_eV'])
            ax.vlines(x,low,high,color=f'C{j}',linewidth=2,label=label if i==0 else None)
            ax.plot([x-.04,x+.04],[low,low],color=f'C{j}');ax.plot([x-.04,x+.04],[high,high],color=f'C{j}')
            vals=[float(r['delta_eV']) for r in cross if r['treatment']==treatment and r['transport']==transport and r['mode']==mode]
            ax.scatter(np.full(len(vals),x),vals,facecolors='none',edgecolors=f'C{j}',s=24,zorder=3)
            ax.scatter(x,float(row['difference_of_observed_means_eV']),color=f'C{j}',s=24,zorder=4)
    ax.axhline(0,color='black',linewidth=.7)
    ax.axhline(float(reference['delta_static_eV']),color='black',linestyle='--',linewidth=.8,label='Static minimum gap')
    ax.axhline(float(reference[f'delta_harmonic_target_{treatment}_eV']),color='gray',linestyle=':',linewidth=1.1,label='Finite-cell harmonic reference')
    ax.set_xticks(range(3),['1PT','2PT','3PT']);ax.set_title(treatment.capitalize());ax.grid(axis='y',alpha=.2)
axes[0].set_ylabel(f"Estimate at {float(ranges[0]['target_temperature_K']):.2f} K: arrangement 2 minus 1 (eV/cell)")
axes[0].legend(fontsize=8,loc='lower left')
fig.suptitle('Post hoc first 10 ps: assignment-stable at 2 fs inspection\nExploratory estimates; equilibrium and thermodynamic populations unvalidated',fontsize=11)
fig.text(.5,.025,'Segments: observed 2 × 2 cross-seed ranges, not confidence intervals. Open dots share 2 + 2 trajectories; solid dot is difference of means.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.08,1,.86]);save(fig,folder,'prefix10_transport_ranges_scoped')

# Different physical/control trajectories; match only observation window/grid.
folder=ROOT/'matched5ps_controls';folder.mkdir(exist_ok=True)
cases=[('Baseline NVE, dt = 0.5 fs',ROOT/'baseline_prefix5_matched_controls/native_results.csv'),('Bussi NVT, dt = 0.5 fs',ROOT/'nvt_control_native/native_results.csv'),('Half-dt NVE, dt = 0.25 fs',ROOT/'half_dt_observation05/native_results.csv')]
transport_path=ROOT/'matched5ps_transport_sensitivity/transports_per_repeat.csv';partition_path=ROOT/'completed_partition_diagnostics/partition_diagnostics.csv'
sources.extend([transport_path,partition_path]);trans=read(transport_path);parts=read(partition_path);table=[]
fig,axes=plt.subplots(1,2,figsize=(11,4.5))
for label,path in cases:
    sources.append(path);members=read(path);row=next(r for r in members if r['mode']=='3PT')
    for r in members:
        assert np.isclose(float(r['saved_interval_fs']),.5) and np.isclose(float(r['duration_ps']),5.) and np.isclose(float(r['maximum_lag_ps']),1.25)
        part=next(v for v in parts if Path(v['native_csv']).resolve()==path.resolve() and v['mode']==r['mode'])
        entry=dict(control=label,mode=r['mode'],scope=r['scope'],observed_temperature_K=float(r['observed_temperature_K']),saved_interval_fs=float(r['saved_interval_fs']),physical_maximum_lag_ps=float(r['maximum_lag_ps']),closure_fluidicity=float(r['fluidicity']),integrated_assigned_gas_modes=float(part['integrated_assigned_gas_modes']),cage_f_from_gas_over_3N=float(part['cage_f_from_gas_over_3N']),cage_gate_formula_value=float(part['cage_gate_formula_value']),S_cage_quantum_eV_K=float(r['S_cage_quantum_eV_K']))
        for treatment in ['classical','quantum']:
            v=next(v for v in trans if Path(v['native_file']).resolve()==path.resolve() and v['mode']==r['mode'] and v['treatment']==treatment)
            entry[f'F_{treatment}_native_eV']=float(r[f'F_{treatment}_MD_anchored_eV'])
            entry[f'F_{treatment}_direct_transport_eV']=float(v['F_direct_first_order_eV'])
            entry[f'F_{treatment}_harmonic_reference_transport_eV']=float(v['F_harmonic_reference_transport_eV'])
        table.append(entry)
    d=spectrum(row);legend=f"{label}, {float(row['observed_temperature_K']):.2f} K"
    axes[0].plot(d['frequency_THz'],d['dos_per_cm_cell']/.0299792458,label=legend,linewidth=.8)
    axes[1].plot(d['lag_ps'],d['normalized_VACF'],label=legend,linewidth=.8)
axes[0].set(xlim=(0,25),xlabel='Frequency (THz)',ylabel='Modes per THz per cell');axes[0].legend(fontsize=7)
axes[1].set(xlim=(0,1.25),xlabel='Lag time (ps)',ylabel='Mass-weighted VACF / VACF(0)')
[ax.grid(alpha=.2) for ax in axes]
fig.suptitle('Matched observation: 5 ps window, 0.5 fs saved spacing, 1.25 ps lag\nDifferent native temperatures/warmup paths: no isolated thermostat or timestep effect',fontsize=11)
fig.tight_layout(rect=[0,0,1,.86]);save(fig,folder,'matched5ps_spectra_scoped');write(folder/'matched5ps_control_table.csv',table)

verification={}
for name in ['full20_diagnostic','prefix10_analysis','baseline_prefix5_matched_controls']:
    path=ROOT/name;records=read(path/'native_results.csv');receipt=json.loads((path/'receipt.json').read_text());sources.extend([path/'native_results.csv',path/'receipt.json'])
    assert sha(path/'analysis_source.py')==receipt['script_sha256']=='1019fb739af11cf107dae2967fd39fd1ce3dbb00d7789661d4539a220707a894'
    verification[name]=dict(main_analyses=len(records),sensitivity_analyses=receipt['sensitivity_analyses'],scope=sorted({r['scope'] for r in records}),maximum_mode_integral_error=max(abs(float(r['mode_integral_error'])) for r in records),maximum_energy_anchor_error_eV=max(abs(float(r['energy_anchor_roundtrip_error_eV'])) for r in records),maximum_quantum_identity_error_eV=max(abs(float(r['quantum_thermodynamic_identity_error_eV'])) for r in records),maximum_dt_Omega0=max(float(r['saved_dt_times_Omega0']) for r in records))
result=dict(status='SCOPED_DIAGNOSTIC_FIGURES_AND_TABLES',source_sha256=sha(__file__),inputs={str(p):sha(p) for p in sources},verification=verification,interpretation=['No thermodynamic weights computed','Full20ps hopping/event trajectories are not initial-basin free energies','First10ps is a posthoc restricted-window diagnostic, not independently validated equilibrium','Cross ranges are observed correlated2×2combinations, not confidence intervals','Matched observation does not match nativeT or equilibration paths'])
(ROOT/'scoped_figures_receipt.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(dict(status=result['status'],verification=verification)))
