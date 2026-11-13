"""Small synthetic checks; descriptor conventions are not physics tolerances."""
import json
import numpy as np
import pytest
import torch

from cathode_cvae.graph import build_periodic_graph
from cathode_cvae.parent_spec import ParentSpec
from cathode_cvae.learning.physical_auxiliary import (vacancy_pair_values,
    prepare_targets, make_head, head_prediction, auxiliary_latent)


def square_graph():
    parent = ParentSpec(np.full(4, 3), np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]], float),
                        np.eye(3)*10, np.arange(4), np.arange(4)[None], [False]*3)
    return build_periodic_graph(parent, 1.01)


def test_observable_distinguishes_adjacent_and_opposite_vacancies():
    graph = square_graph()
    x = np.array([[0,0,1,1],[0,1,0,1],[1,1,1,1],[0,0,0,0]], np.uint8)
    got, spec = vacancy_pair_values(x, graph)
    assert got == pytest.approx([.25,0,0,1])
    assert spec['directed_periodic_image_edges'] == 8
    # A quarter turn permutes square sites and preserves this scalar.
    rotated, _ = vacancy_pair_values(x[:, [1,2,3,0]], graph)
    assert np.array_equal(got, rotated)


def test_train_only_scaling_and_test_masks_are_not_observed():
    graph = square_graph()
    x = np.array([[0,0,1,1],[0,1,0,1],[1,1,0,0],[np.nan]*4])
    counts=np.full(4,2); bins=np.array(['low']*3+['']); roles=np.array(['train','train','val','test'])
    raw, genuine, target, metadata = prepare_targets(x, counts, bins, roles, graph, mode='latent', seed=2)
    assert raw[:3] == pytest.approx([.25,0,.25])
    assert np.isnan(raw[-1]) and genuine[-1] == target[-1] == 0
    assert genuine[:3] == pytest.approx([1,-1,1])
    assert metadata['cells']['2:low']['mean'] == .125
    x[2]=[0,1,0,1]
    _, changed, _, after = prepare_targets(x, counts, bins, roles, graph, mode='latent', seed=2)
    assert metadata['cells'] == after['cells'] and changed[2] == -1


def test_shuffled_control_is_within_cell_and_validation_is_genuine():
    graph=square_graph(); x=np.array([[0,0,1,1],[0,1,0,1],[1,1,0,0],[0,1,0,1],[1,1,0,0]])
    counts=np.full(5,2); bins=np.array(['low','low','high','high','low']); roles=np.array(['train']*4+['val'])
    _, true, shuffled, metadata=prepare_targets(x, counts, bins, roles, graph, mode='shuffled', seed=45)
    for b in ['low','high']:
        t=np.flatnonzero((bins==b)&(roles=='train'))
        assert np.array_equal(np.sort(true[t]),np.sort(shuffled[t]))
    assert shuffled[-1]==true[-1]
    assert metadata['shuffle_map_sha256']


def test_auxiliary_head_initialization_preserves_global_rng_and_condition_control_has_no_z_gradient():
    torch.manual_seed(4); expected=torch.rand(5)
    torch.manual_seed(4); head=make_head(2,4,'condition',19,'cpu'); actual=torch.rand(5)
    assert torch.equal(expected,actual)
    z=torch.randn(3,2,requires_grad=True); c=torch.randn(3,4)
    head_prediction(head,z,c,'condition').sum().backward()
    assert z.grad is None
    head=make_head(2,4,'latent',19,'cpu')
    head_prediction(head,z,c,'latent').sum().backward()
    assert z.grad is not None and torch.count_nonzero(z.grad)>0


def test_missing_training_cell_is_rejected():
    with pytest.raises(ValueError,match='No TRAIN'):
        prepare_targets(np.array([[0,0,1,1],[0,1,0,1]]),np.array([2,2]),
            np.array(['low','high']),np.array(['train','val']),square_graph(),mode='latent',seed=3)


@pytest.mark.parametrize('posterior_input', ['sampled', 'mean'])
def test_condition_only_auxiliary_keeps_generator_updates_identical(tmp_path, posterior_input):
    # This small integration test may run on CI or an allocation; it never
    # disables the site's training guard when invoked on a NERSC login host.
    from cathode_cvae.runtime import require_compute_allocation
    try: require_compute_allocation()
    except RuntimeError as exc: pytest.skip(str(exc))
    from cathode_cvae.learning import train_mask
    parent=tmp_path/'parent.npz'
    np.savez(parent,numbers=np.full(4,3),positions_A=np.array([[0,0,0],[1,0,0],[1,1,0],[0,1,0]],float),
             cell_A=np.eye(3)*10,target_indices=np.arange(4),permutations=np.arange(4)[None],pbc=np.zeros(3,bool))
    data=tmp_path/'data.npz';x=np.array([[1,1,0,0],[1,0,1,0],[1,0,0,1],[0,1,1,0],[0,1,0,1],[0,0,1,1]],np.uint8)
    np.savez(data,occupancy=x,n_target=np.full(6,2),energy_eV_atom=np.array([-2,-1,0,-1.5,-.5,3.]),
             weight=np.ones(6),split=np.array(['train']*3+['val']*2+['test']),
             structure_ids=np.array([f'synthetic_{i}' for i in range(6)]),parent_sha256=np.asarray(train_mask.sha256(parent)))
    checkpoints=[]
    for name, extra in [('baseline',[]),('condition',['--latent-physical-weight','1','--latent-physical-mode','condition',
                                                    '--latent-physical-input',posterior_input])]:
        out=tmp_path/name
        train_mask.main(['--data',str(data),'--parent',str(parent),'--output-dir',str(out),
            '--kind','ar_cvae','--condition-mode','composition','--latent-dim','2','--max-updates','2',
            '--epochs','10','--minimum-epochs','10','--patience','10','--batch-size','2',
            '--validation-samples','1','--device','cpu','--threads','1','--max-seconds','60',*extra])
        checkpoints.append(torch.load(out/'last_update.pt',weights_only=False))
        if name == 'condition':
            metadata=json.loads((out/'configuration.json').read_text())['physical_auxiliary']
            assert metadata['posterior_input'] == posterior_input
            assert metadata['head_inputs'] == 'condition only'
    for key in checkpoints[0]['state_dict']:
        assert torch.equal(checkpoints[0]['state_dict'][key],checkpoints[1]['state_dict'][key]),key
    assert 'physical_auxiliary_state_dict' in checkpoints[1]
    assert checkpoints[0]['val_objective']==checkpoints[1]['val_objective']


def test_mean_supervision_gradient_avoids_posterior_sampling_noise():
    mu=torch.tensor([[.3,-.4]],requires_grad=True)
    logvar=torch.tensor([[.2,-.1]],requires_grad=True)
    sampled=mu+torch.exp(.5*logvar)*torch.ones_like(mu)
    mean_loss=auxiliary_latent(mu,sampled,'mean').square().sum()
    mean_grad=torch.autograd.grad(mean_loss,(mu,logvar),allow_unused=True,retain_graph=True)
    assert torch.count_nonzero(mean_grad[0])>0 and mean_grad[1] is None
    sampled_loss=auxiliary_latent(mu,sampled,'sampled').square().sum()
    sampled_grad=torch.autograd.grad(sampled_loss,(mu,logvar),allow_unused=True)
    assert torch.count_nonzero(sampled_grad[0])>0 and torch.count_nonzero(sampled_grad[1])>0
    with pytest.raises(ValueError,match='sampled or mean'):
        auxiliary_latent(mu,sampled,'unknown')
