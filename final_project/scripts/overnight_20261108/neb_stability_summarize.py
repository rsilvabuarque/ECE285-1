"""Summarize every completed/bounded route without selecting convenient spans."""
import argparse
import csv
import json
from pathlib import Path
import sys
import numpy as np
from cathode_cvae.runtime import require_compute_allocation
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'conditioning_20261107'))
from neb_common import digest,load_npz,read_json,write_json


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--runs',nargs='+',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--profiles',nargs='*',type=Path,default=[])
    args=p.parse_args();require_compute_allocation();args.output.mkdir(parents=True,exist_ok=False)
    rows=[];sources=[];bands={}
    import matplotlib;matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,2,figsize=(11,4))
    for run in args.runs:
        receipt=read_json(run/'receipt.json');binding=receipt['binding'];event=binding['event'];route=binding['route']
        final=run/'final_band.npz';path=final if final.exists() else run/'last_state.npz';data=load_npz(path)
        display_route=route.replace('_interrupted_continuation',' recovery').replace('_',' ')
        finite=bool(np.isfinite(data['energy_eV']).all());label=f"{display_route}, {binding['interior_images']} interior"
        if not receipt['converged']:label+=' [step cap]'
        row=dict(event=event,route=route,interior_images=binding['interior_images'],target_eV_A=binding['target_projected_Fmax_eV_A'],
            status=receipt['status'],numerically_converged=receipt['converged'],steps=receipt.get('optimizer_steps',''),
            projected_Fmax_eV_A=receipt.get('projected_Fmax_eV_A',''),
            forward_span_eV_hop=receipt.get('discrete_forward_span_eV',''),reverse_span_eV_hop=receipt.get('discrete_reverse_span_eV',''),
            endpoint_start_sha256=binding['endpoint_geometry_sha256'][0],endpoint_end_sha256=binding['endpoint_geometry_sha256'][1],
            EFS_attempts=receipt['call_accounting']['actual_calculator_EF_attempts'],elapsed_seconds=receipt['call_accounting']['elapsed_seconds'],
            path_sha256=digest(path))
        rows.append(row);sources.append(dict(path=str((run/'receipt.json').resolve()),sha256=digest(run/'receipt.json')))
        if finite:
            xyz=data['positions_A'];arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0).reshape(len(xyz)-1,-1),axis=1))]
            x=arc/arc[-1];ax=axes[0 if 'basin_1' in event else 1]
            ax.plot(x,data['energy_eV']-data['energy_eV'][0],'-o' if receipt['converged'] else '--o',label=label)
            bands[(event,route,binding['interior_images'])]=(data,binding,receipt)
    comparisons=[]
    keys=list(bands)
    for i,a in enumerate(keys):
        for b in keys[i+1:]:
            if a[0]!=b[0]:continue
            da,ba,ra=bands[a];db,bb,rb=bands[b]
            fixed=ba['endpoint_geometry_sha256']==bb['endpoint_geometry_sha256']
            if not fixed:continue
            delta=float(rb['discrete_forward_span_eV']-ra['discrete_forward_span_eV'])
            both=ra['converged'] and rb['converged']
            comparison=dict(event=a[0],route_A=a[1],images_A=a[2],route_B=b[1],images_B=b[2],
                same_endpoints=True,both_numerically_converged=both,
                common_force_target=ba['target_projected_Fmax_eV_A']==bb['target_projected_Fmax_eV_A'],
                forward_span_B_minus_A_eV_hop=delta,
                reverse_span_B_minus_A_eV_hop=float(rb['discrete_reverse_span_eV']-ra['discrete_reverse_span_eV']),
                practical_numerical_target_eV_hop=.02,within_practical_target=bool(both and abs(delta)<=.02),
                same_image_path_RMS_atom_A='')
            if da['positions_A'].shape==db['positions_A'].shape:
                comparison['same_image_path_RMS_atom_A']=float(np.sqrt(np.mean(np.sum((da['positions_A']-db['positions_A'])**2,axis=2))))
            comparisons.append(comparison)
    for profile in args.profiles:
        r=read_json(profile/'receipt.json');d=load_npz(profile/'profile.npz');e=d['energy_eV'];xyz=d['positions_A']
        arc=np.r_[0,np.cumsum(np.linalg.norm(np.diff(xyz,axis=0).reshape(len(xyz)-1,-1),axis=1))]
        # Filename event inference is merely a display label; exact source hash
        # and path in the profile protocol determine the scientific cohort.
        source=r['binding']['band']['path'];ax=axes[0 if 'basin_1' in source or 'event1' in source else 1]
        ax.plot(arc/arc[-1],e-e[0],'--s',label=f"{r['binding']['model'].upper()} fixed UMA path")
        sources.append(dict(path=str((profile/'receipt.json').resolve()),sha256=digest(profile/'receipt.json')))
    for j,ax in enumerate(axes):
        ax.set(title=f'Event {j+1}: saved route families',xlabel='Normalized Cartesian 3N path arclength',ylabel='Potential energy relative to start (eV/cell)');ax.grid(alpha=.2)
        if ax.lines:ax.legend(fontsize=7)
    fig.tight_layout();fig.savefig(args.output/'route_profiles.pdf');fig.savefig(args.output/'route_profiles.png',dpi=160);plt.close(fig)
    for name,table in [('bands',rows),('comparisons',comparisons)]:
        if not table:continue
        with (args.output/f'{name}.csv').open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(table[0]));w.writeheader();w.writerows(table)
    write_json(args.output/'receipt.json',dict(status='COMPLETE_SUMMARY',source_sha256=digest(__file__),sources=sources,
        total_runs=len(rows),numerically_converged=sum(r['numerically_converged'] for r in rows),
        numerical_stability_target_eV_hop=.02,scope='MLIP potential spans per hop; force convergence alone is not a saddle/path validation. Fixed-coordinate MACE profiles are not MACE optimized barriers.',
        outputs={p.name:digest(p) for p in args.output.iterdir() if p.is_file()}))


if __name__=='__main__':main()
