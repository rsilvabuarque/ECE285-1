"""Assemble completed research observations; no new models, forces or tests."""
import argparse
import csv
import json
from pathlib import Path
from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--research-run',type=Path,required=True)
    p.add_argument('--candidate-review',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();require_compute_allocation();args.output.mkdir(parents=True,exist_ok=False)
    files=[];bands=[];endpoints=[]
    def read(file):
        files.append(dict(path=str(file.resolve()),sha256=sha256(file)));return json.loads(file.read_text())
    for group,root in [('fixed_original_endpoints',args.research_run/'neb'),('new_teacher_tightened_endpoints',args.research_run/'neb_new_endpoints')]:
        for file in sorted(root.glob('*/receipt.json')):
            r=read(file)
            if 'binding' not in r or 'interior_images' not in r['binding']:continue
            b=r['binding'];bands.append(dict(case=file.parent.name,group=group,model=b.get('model','uma'),
                interior_images=b['interior_images'],force_target_eV_A=b['target_projected_Fmax_eV_A'],status=r['status'],
                steps=r.get('optimizer_steps',''),projected_Fmax_eV_A=r.get('projected_Fmax_eV_A',''),
                forward_span_eV=r.get('discrete_forward_span_eV',''),reverse_span_eV=r.get('discrete_reverse_span_eV',''),
                reaction_energy_eV=r.get('discrete_forward_span_eV',0)-r.get('discrete_reverse_span_eV',0),
                climbing_image=r.get('climbing_image',''),endpoint_geometry_sha256=';'.join(b['endpoint_geometry_sha256']),
                logged_EFS_attempts=r['call_accounting']['actual_calculator_EF_attempts'],source=str(file.resolve())))
    for file in sorted((args.research_run/'endpoint_tightening').glob('*/receipt.json')):
        r=read(file)
        for e in r['endpoints']:
            endpoints.append(dict(teacher=r['binding']['model'],endpoint=e['index'],status=r['status'],steps=e['steps'],
                physical_Fmax_eV_A=e['physical_Fmax_eV_A'],energy_change_eV=e['energy_change_eV'],
                host_aligned_RMS_A=e['geometry_change_from_original']['identity_preserving_host_aligned_RMS_A'],
                maximum_atom_difference_A=e['geometry_change_from_original']['maximum_atom_difference_A'],
                marked_event_retained=r['marked_event_retained'],band_gate=r['band_gate'],source=str(file.resolve())))
    profile=read(args.research_run/'neb_mace/event2_uma7/receipt.json');candidate=read(args.candidate_review/'receipt.json')
    for name,rows in [('bands.csv',bands),('endpoints.csv',endpoints)]:
        with (args.output/name).open('w',newline='') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    fixed=[b for b in bands if b['case'] in ['basin_2_repeat1_saved_5','basin_2_repeat1_saved_7','basin_2_repeat1_saved_9']]
    if len(fixed)!=3 or len({b['endpoint_geometry_sha256'] for b in fixed})!=1:raise ValueError('Fixed-original event2 image comparison is incomplete')
    original7=next(b for b in fixed if b['interior_images']==7)
    tight=next(b for b in bands if b['case']=='basin_2_repeat1_fixed7_tight01')
    new_uma=next(b for b in bands if b['group']=='new_teacher_tightened_endpoints' and b['model']=='uma')
    delta=candidate['displacement_size_comparison'];overlap=candidate['mode_route_overlaps']
    summary=dict(status='COMPLETE_READ_ONLY_PHYSICAL_SUMMARY',source_sha256=sha256(__file__),sources=files,
        event2_fixed_original_common_target_forward_spread_eV=max(b['forward_span_eV'] for b in fixed)-min(b['forward_span_eV'] for b in fixed),
        event2_fixed7_tolerance_0p01_minus_0p02_forward_eV=tight['forward_span_eV']-original7['forward_span_eV'],
        event2_new_UMA_endpoints_0p02_minus_original_7_0p01_forward_eV=new_uma['forward_span_eV']-tight['forward_span_eV'],
        combined_protocol_change_exceeds_practical_target=abs(new_uma['forward_span_eV']-tight['forward_span_eV'])>.02,
        combined_change_scope='Both endpoint coordinates and band force target differ; demonstrates combined protocol sensitivity, not an isolated endpoint-error estimate. Restricted image/tolerance checks do not establish overall barrier stability.',
        numerical_stability_target_eV_hop=.02,numerical_target_scope='Chosen practical numerical target, not DFT or physical accuracy',
        saddle_candidate_geometry_sha256=delta['candidate_geometry_sha256'],saddle_displacement_sensitivity=delta,
        mode_route_overlaps=overlap,MACE_on_original_UMA7=dict(forward_span_eV=profile['discrete_forward_span_eV'],
            reverse_span_eV=profile['discrete_reverse_span_eV'],peak_image=profile['peak_image'],physical_endpoint_Fmax_eV_A=profile['physical_endpoint_Fmax_eV_A']),
        later_DFT_candidate_handoff=str((args.candidate_review/'handoff_manifest.json').resolve()),
        exposure_history_snapshot=str((args.research_run/'research_exposure_snapshots/receipt.json').resolve()),
        unresolved=['Event1doesnotmeetdeclaredforcecriterion','Stablelocalindexonecurvaturedoesnotverifythecompleteoriginalelementaryhop',
            'SameRAWassignedmaskdoesnotidentifycontinuousPESbasin','Offsiteforce-smallstateisnotcertifiedalocalminimum',
            'TeacherindependentnewendpointbandsarenotDFTvalidatedbarriers'],
        interrupted_call_accounting='Externallykilledrefinedworkerhas3020completedloggedcalls; atmostoneadditionalserialcalculateinvocationcouldhavebeeninflightwithoutapersistedrow. Do notclaimneuralbackbone-forwardcounts. Allocatedtimeincludingthisfailureisaccountedbyrootresourceaudit.',
        new_MLIP_calls=0,new_training_updates=0,new_DFT_calls=0)
    (args.output/'receipt.json').write_text(json.dumps(summary,indent=2)+'\n')
    (args.output/'FINDINGS.md').write_text('''# Research findings

Event 2 passes only the restricted fixed-original-endpoint image and tolerance checks: the 5/7/9-interior-image forward-span spread is 0.00465 eV, and the fixed-seven-image 0.01 target changes the forward span by −0.00784 eV. However, changing to the tighter UMA endpoints with a 0.02 band target raises the forward span from 0.169182 to 0.195626 eV: +0.026444 eV, above the chosen 0.02 practical target. This is a combined endpoint/force-target change, so it is not an isolated endpoint-error estimate. Overall barrier stability is not established. Event 1 remains nonconverged under its declared caps; its terminal spans are not validated barriers.

At one retained event 2 climbing image, raw central-force reconstruction confirms one negative internal curvature at both 0.01 and 0.005 Angstrom displacement. The negative modes agree to 0.999999994 absolute overlap. However, the two downhill branches do not recover both intended original endpoints. The off-site plus branch and end-like minus branch share the exact same raw assigned vacancy mask while retaining different coordinates, coordination and energies. This supports a local index-one candidate, not a verified single elementary marked hop. The off-site state's local minimum character remains unverified. Unweighted force-derivative noise and mass-weighted eigenvalues have different units; displacement-size agreement and mode overlap provide the numerical evidence.

Separate UMA/MACE endpoint tightening retains the marked event and reaches 0.01 eV/Angstrom. The resulting new-endpoint seven-image bands have forward/reverse spans 0.195626/0.177294 eV (UMA) and 0.199585/0.251953 eV (MACE). Similar forward heights do not establish a common physical barrier: reaction energies differ in sign and peak indices differ (5 versus 3). These new endpoint identities are separate from the fixed-original image/tolerance comparison.

Every viewed coordinate has an exact NPZ, source hash and continuous local-to-parent atom mapping. POSCAR/CIF files are viewing exports. The exact-coordinate handoff and all accepted trajectory/FD exposure snapshots preserve the off-site state. No new force, DFT, MD or model-training call is made by this summary. Final charged resources come from the integrating root audit, not sums of overlapping process times.
''')
    print(json.dumps({k:summary[k] for k in ['status','event2_fixed_original_common_target_forward_spread_eV','event2_fixed7_tolerance_0p01_minus_0p02_forward_eV','new_MLIP_calls']},indent=2))


if __name__=='__main__':main()
