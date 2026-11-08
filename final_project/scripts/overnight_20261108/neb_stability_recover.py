"""One bounded geometric continuation after externally interrupted NEB.

Uses only the unspent original step/call/time budget, preserves the interrupted
run, restores its last complete saved coordinates without resampling, and
declares that FIRE velocity/memory is restarted rather than exactly restored.
"""
import time
PROCESS_START=time.monotonic()
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np
from cathode_cvae.runtime import require_compute_allocation

spec=importlib.util.spec_from_file_location('neb_stability_run',Path(__file__).with_name('neb_stability_run.py'))
driver=importlib.util.module_from_spec(spec);spec.loader.exec_module(driver)


def interruption_facts(run,reason):
    protocol=driver.read_json(run/'protocol.json')
    if (run/'receipt.json').exists():raise ValueError('Do not recover or relabel a terminal scientific run')
    progress=[json.loads(line) for line in (run/'progress.jsonl').read_text().splitlines() if line]
    calls=[json.loads(line) for line in (run/'calculator_calls.jsonl').read_text().splitlines() if line]
    if not progress or not calls:raise ValueError('Recovery requires recorded complete progress and physical calls')
    last=progress[-1];snapshot=run/'latest_band.npz';data=driver.load_npz(snapshot)
    matching=[r for r in progress if r['step']%10==0 and np.allclose(r['energy_eV'],data['energy_eV'],rtol=0,atol=1e-10)]
    if len(matching)!=1 or not np.all(data['labels_match_geometry']):raise ValueError('Cannot bind last complete snapshot to one recorded optimizer step')
    after=[r for r in calls if r['call_index']>last['actual_EFS_attempts']]
    # A subsequent calculator call proves an additional optimizer position
    # update began, even if its whole-band force evaluation was interrupted.
    charged_steps=int(last['step'])+int(bool(after))
    elapsed_upper_reservation=math.ceil(last['elapsed_seconds']+sum(r['elapsed_seconds'] for r in after))+1
    remaining_steps=int(protocol['maximum_total_optimizer_steps'])-charged_steps
    remaining_calls=int(protocol['maximum_EFS_calls'])-len(calls)
    remaining_seconds=float(protocol['maximum_seconds_including_imports_loading'])-elapsed_upper_reservation
    return dict(status='EXTERNALLY_INTERRUPTED_PARTIAL',operational_reason=reason,
        original_run=str(run.resolve()),original_protocol_sha256=driver.digest(run/'protocol.json'),
        last_complete_progress=last,last_complete_snapshot_step=int(matching[0]['step']),
        snapshot=dict(path=str(snapshot.resolve()),sha256=driver.digest(snapshot)),
        progress_sha256=driver.digest(run/'progress.jsonl'),calls_sha256=driver.digest(run/'calculator_calls.jsonl'),
        saved_EFS_attempts=len(calls),successful_EFS=sum(r['status']=='success' for r in calls),
        failed_EFS=sum(r['status']!='success' for r in calls),inflight_unreported_calls=len(after),
        conservative_charged_optimizer_steps=charged_steps,remaining_steps=remaining_steps,
        previous_elapsed_seconds_reserved=elapsed_upper_reservation,remaining_seconds=remaining_seconds,remaining_EFS_calls=remaining_calls,
        physical_timer_scope='Ceiling(last recorded process elapsed + subsequent completed call times)+1s, reserved against original45min cap; any scheduler overhead remains separate allocation cost',
        restart_scope='Restore exact last complete saved geometry; FIRE internal velocity/memory reset. Any rollback/repeated physical work remains charged. No full rerun.')


def exact_restore(data,interior,initialization='retain_coordinates'):
    if len(data['positions_A'])!=interior+2:raise ValueError('Recovery must retain image count')
    images=[driver.atoms_from(data,row) for row in data['positions_A']]
    return images,[driver.geometry_hash(images[0]),driver.geometry_hash(images[-1])]


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--interrupted-run',type=Path,required=True)
    p.add_argument('--interruption-reason',required=True);p.add_argument('--audit-output',type=Path,required=True)
    p.add_argument('--output',type=Path);p.add_argument('--execute',action='store_true')
    p.add_argument('--device',default='cuda');p.add_argument('--threads',type=int,default=4)
    args=p.parse_args();require_compute_allocation();facts=interruption_facts(args.interrupted_run,args.interruption_reason)
    args.audit_output.mkdir(parents=True,exist_ok=False);driver.write_json(args.audit_output/'receipt.json',facts)
    if not args.execute:print(json.dumps(facts,indent=2));return 0
    if args.output is None:raise ValueError('A fresh output directory is required for execution')
    if min(facts['remaining_steps'],facts['remaining_EFS_calls'],facts['remaining_seconds'])<=0:raise ValueError('Original bounded budget is exhausted')
    old=driver.read_json(args.interrupted_run/'protocol.json');data=driver.load_npz(facts['snapshot']['path'])
    images,hashes=exact_restore(data,old['interior_images'])
    if hashes!=old['endpoint_geometry_sha256']:raise ValueError('Saved endpoints changed')
    original_binding=driver.request_binding
    def bind(request,current_images,endpoint_hashes):
        result=original_binding(request,current_images,endpoint_hashes)
        result.update(recovery=facts,recovery_source_sha256=driver.digest(__file__),
            initialization='Exact saved coordinates from externally interrupted route; NO arclength resampling; FIRE memory reset')
        return result
    driver.prepare_band=exact_restore;driver.request_binding=bind;driver.PROCESS_START=PROCESS_START
    argv=['--root-protocol',old['root_protocol']['path'],'--band',facts['snapshot']['path'],
        '--parent',old['parent']['path'],'--checkpoint',old['checkpoint']['path'],'--event',old['event'],'--route',old['route']+'_interrupted_continuation',
        '--output',str(args.output),'--interior',str(old['interior_images']),'--fmax',str(old['target_projected_Fmax_eV_A']),
        '--initialization','retain_coordinates',
        '--steps',str(facts['remaining_steps']),'--max-calls',str(facts['remaining_EFS_calls']),
        '--max-seconds',str(facts['remaining_seconds']),'--device',args.device,'--threads',str(args.threads)]
    return driver.main(argv)


if __name__=='__main__':raise SystemExit(main())
