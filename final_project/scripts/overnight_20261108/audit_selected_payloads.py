"""Verify recorded named checkpoint selections and label/proposal semantics.

This inspects trusted local checkpoints and validation-selection metadata only.
It does not select another checkpoint or read any structure energy labels.
"""
import argparse,csv,json
from pathlib import Path
from cathode_cvae.data import sha256
from cathode_cvae.learning.runtime import trusted_checkpoint
from cathode_cvae.runtime import require_compute_allocation

DELETION_PROTOCOL='pristine-derived deletion; unrelaxed; exact canonical parent coordinates'


def validate_payload(filename,summary,payload,history,expected_updates=4000):
    if summary.get('status')!='FIT_COMPLETE' or summary['optimizer_updates']!=expected_updates:
        raise ValueError('The training trajectory did not complete its prescribed update budget')
    named=[(key,row) for key,row in summary['selection_checkpoints'].items() if row['filename']==filename]
    if len(named)!=1:raise ValueError('Checkpoint is not a unique named selection in the terminal summary')
    key,selection=named[0];beta=float(key);cfg=payload['configuration']
    if filename!='best_validation_beta_0p1.pt' or beta!=.1:
        raise ValueError('This readiness check requires the predeclared named beta=0.1 checkpoint')
    if payload.get('selection_beta')!=beta or cfg.get('selected_validation_beta')!=beta:
        raise ValueError('Payload/configuration selection beta disagrees with the named summary row')
    if payload.get('epoch')!=selection['epoch'] or payload.get('val_objective')!=selection['val_objective']:
        raise ValueError('Payload epoch/objective disagrees with its named selection, irrespective of primary best_epoch')
    selected_history=[r for r in history if int(r['epoch'])==payload['epoch']]
    if len(selected_history)!=1:raise ValueError('Selected epoch is not unique in validation history')
    row=selected_history[0];column='val_objective_beta_'+format(beta,'.12g').replace('.','p')
    if (int(row['optimizer_updates'])!=payload['optimizer_updates'] or
            float(row[column])!=payload['val_objective'] or not 0<payload['optimizer_updates']<=expected_updates):
        raise ValueError('Named selection does not match the executed history row/update count')
    target=cfg.get('energy_target','deletion')
    label=cfg.get('label_energy_protocol',cfg.get('geometry_protocol'))
    proposal=cfg.get('proposal_geometry_protocol',cfg.get('geometry_protocol'))
    if proposal!=DELETION_PROTOCOL:raise ValueError('The generator must restore pristine-derived unrelaxed deletion coordinates')
    if cfg.get('geometry_protocol')!=label:raise ValueError('Legacy label-protocol alias disagrees with the explicit label protocol')
    if target=='endpoint':
        if label==proposal or 'endpoint' not in label:raise ValueError('Endpoint labels were confused with the generated proposal geometry')
    elif target=='deletion':
        if label!=DELETION_PROTOCOL:raise ValueError('Deletion labels and deletion proposal geometry disagree')
    else:raise ValueError('Unknown label target in this frozen comparison')
    return dict(selected_beta=beta,selected_epoch=payload['epoch'],selected_optimizer_updates=payload['optimizer_updates'],
        completed_optimizer_updates=expected_updates,primary_beta1_best_epoch=summary['best_epoch'],
        named_selection_matches_payload_and_history=True,energy_target=target,label_energy_protocol=label,
        proposal_geometry_protocol=proposal,proposal_is_unrelaxed_deletion=True,
        endpoint_label_is_not_claimed_as_generated_relaxation=(target=='endpoint'),fit_seed=cfg['seed'])


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--checkpoint',type=Path,action='append',required=True)
    p.add_argument('--output',type=Path,required=True);p.add_argument('--expected-updates',type=int,default=4000)
    a=p.parse_args(argv);require_compute_allocation()
    if a.output.exists():raise FileExistsError('Preserve the previous selection audit')
    results=[]
    for path in a.checkpoint:
        before=sha256(path);summarypath=path.with_name('summary.json');historypath=path.with_name('history.csv')
        summary=json.loads(summarypath.read_text())
        with historypath.open() as f:history=list(csv.DictReader(f))
        payload=trusted_checkpoint(path,'cpu')
        result=validate_payload(path.name,summary,payload,history,a.expected_updates)
        named=summary['selection_checkpoints']['0.1']
        if named['sha256']!=before or sha256(path)!=before:raise ValueError('Named checkpoint bytes changed or do not match the terminal summary')
        result.update(checkpoint=str(path.resolve()),checkpoint_sha256=before,checkpoint_unchanged=True,
            summary_sha256=sha256(summarypath),history_sha256=sha256(historypath))
        results.append(result)
    a.output.parent.mkdir(parents=True,exist_ok=True)
    receipt=dict(status='PASS',models=results,checkpoint_count=len(results),source_sha256=sha256(__file__),
        scope='Verify the recorded named selection and its history; do not substitute summary.best_epoch or change the writer improvement rule',
        energy_fields_read=0,TEST_records_read=0,new_model_forward_calls=0,new_MLIP_calls=0,new_optimizer_updates=0)
    a.output.write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(dict(status='PASS',checkpoint_count=len(results))))
    return receipt


if __name__=='__main__':main()
