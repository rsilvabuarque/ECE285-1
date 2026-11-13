"""Publish reviewed template-applicability diagnostics, retaining all raw MD frames."""
import csv,json,os
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
rows=list(csv.DictReader((HERE/'reports/MD_mapping_groups.csv').open()));diag=list(csv.DictReader((HERE/'reports/MD_mapping_diagnostic.csv').open()));counts={}
for m in ['uma','mace']:
    with np.load(HERE/'data'/f'geometry_{m}_md.npz') as z:
        local=z['localized'];changed=z['occupancy_changed'];assert not (local&changed).any()
        counts[m]=dict(frames=len(local),reference_localized_frames=int(local.sum()),ancestral_mask_changed_frames=int(changed.sum()))
assert len(diag)==48 and len({r['task_id'] for r in diag})==12
assert all(int(r['large_fractional_snapshot_jumps'])==0 for r in diag)
fig,axs=plt.subplots(1,2,figsize=(9,3.6),sharey=True);ns=[12,24,36,42,46];Ts=[298.15,393.15,433.15,473.15,493.15]
for ax,m in zip(axs,['uma','mace']):
    values=np.array([[float(next(r['localized_fraction'] for r in rows if r['model']==m and int(r['n_Li'])==n and float(r['T_K'])==T)) for n in ns] for T in Ts])
    im=ax.imshow(values,vmin=0,vmax=1,cmap='viridis',origin='lower',aspect='auto')
    for i in range(5):
        for j in range(5):ax.text(j,i,f'{values[i,j]:.0%}',ha='center',va='center',color='black' if values[i,j]>.6 else 'white',fontsize=8)
    ax.set_title(m);ax.set_xticks(range(5),[f'{n/48:.3f}' for n in ns]);ax.set_yticks(range(5),[f'{T:.0f}' for T in Ts]);ax.set_xlabel('Li fraction')
axs[0].set_ylabel('Target temperature (K)')
fig.subplots_adjust(left=.08,right=.82,top=.79,bottom=.18,wspace=.18)
coloraxis=fig.add_axes([.86,.20,.025,.56]);fig.colorbar(im,cax=coloraxis,label='Reference-localized fraction')
fig.suptitle('Template applicability of thermal geometries\n80 capped frames per cell; same force/stress labels retained for every frame',fontsize=11);fig.savefig(HERE/'reports/MD_template_applicability.pdf');fig.savefig(HERE/'reports/MD_template_applicability.png',dpi=180);plt.close(fig)
old=json.loads((HERE/'validation/MD_mapping_review.json').read_text());old.update(status='PASS',purpose='Gauge/ancestry diagnostic and template-applicability review; not phase classification or DFT validation.',all_frame_counts=counts,all_reference_localized_frames_preserve_ancestral_occupancy=True,raw_periodic_jump_check='No fractional snapshot increment above half a cell coordinate in the12 examined trajectories.',interpretation='Low-lithiation examples retain large internal motion after independent periodic Co/O host registration; high-lithiation controls remain much closer to parent sites. This is model-predicted motion, not an established O1/spinel transition.',training_decision='Keep initial dense geometry CVAE on the packaged UMA fixed-cell cohort. Reference-localized MD frames can be considered with lineage/ensemble controls; other frames require a representation/reassignment review. Preserve every raw frame and its flags.',source_tables=['reports/MD_mapping_groups.csv','reports/MD_mapping_diagnostic.csv'])
path=HERE/'validation/MD_mapping_review.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(old,indent=2)+'\n');tmp.replace(path);print(json.dumps(old),flush=True)
