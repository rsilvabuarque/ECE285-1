"""Named beta0.1 metadata must not be replaced by primary beta1 best_epoch."""
import copy,importlib.util
from pathlib import Path
import pytest

p=Path(__file__).resolve().parents[1]/'scripts/overnight_20261108/audit_selected_payloads.py'
s=importlib.util.spec_from_file_location('selected_payloads_test',p);m=importlib.util.module_from_spec(s);s.loader.exec_module(m)

def fixture(target='endpoint'):
    summary=dict(status='FIT_COMPLETE',optimizer_updates=4000,best_epoch=12,selection_checkpoints={'0.1':dict(filename='best_validation_beta_0p1.pt',epoch=37,val_objective=1.234)})
    label='paired fixed-cell UMA FIRE endpoint; synthetic test' if target=='endpoint' else m.DELETION_PROTOCOL
    cfg=dict(selected_validation_beta=.1,energy_target=target,label_energy_protocol=label,geometry_protocol=label,proposal_geometry_protocol=m.DELETION_PROTOCOL,seed=1)
    payload=dict(configuration=cfg,selection_beta=.1,epoch=37,optimizer_updates=296,val_objective=1.234)
    history=[dict(epoch='37',optimizer_updates='296',val_objective_beta_0p1='1.234')]
    return summary,payload,history

def test_named_payload_uses_its_own_epoch_and_keeps_endpoint_label_separate():
    summary,payload,history=fixture()
    r=m.validate_payload('best_validation_beta_0p1.pt',summary,payload,history)
    assert r['selected_epoch']==37 and r['primary_beta1_best_epoch']==12
    assert r['selected_optimizer_updates']==296 and r['completed_optimizer_updates']==4000
    assert r['proposal_is_unrelaxed_deletion'] and r['energy_target']=='endpoint'

@pytest.mark.parametrize('field,value',[('epoch',12),('selection_beta',1.),('optimizer_updates',4000)])
def test_wrong_primary_selection_or_update_metadata_is_rejected(field,value):
    summary,payload,history=fixture();payload[field]=value
    with pytest.raises(ValueError):m.validate_payload('best_validation_beta_0p1.pt',summary,payload,history)

def test_endpoint_label_cannot_be_used_as_proposal_geometry():
    summary,payload,history=fixture();payload['configuration']['proposal_geometry_protocol']=payload['configuration']['label_energy_protocol']
    with pytest.raises(ValueError):m.validate_payload('best_validation_beta_0p1.pt',summary,payload,history)
