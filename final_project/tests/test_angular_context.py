"""Tests use synthetic site geometry; tolerances check numerics, not physics."""

import numpy as np
import pytest
import torch

from cathode_cvae.angular_context import AngularHistogramContext, RedundantRadialContext
from cathode_cvae.graph import build_periodic_graph
from cathode_cvae.learning.models import ARBaseline, ARCVAE, make_model
from cathode_cvae.learning.generate_mask import load_model
from cathode_cvae.learning.schedules import schedule_metadata
from cathode_cvae.parent_spec import ParentSpec


def ring_graph(rotation=None, translation=None):
    angles = np.arange(6) * np.pi / 3
    pos = np.vstack((np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6))),
                     np.zeros((1, 3))))
    cell = np.eye(3) * 10
    if rotation is not None:
        pos = pos @ rotation.T
        cell = cell @ rotation.T
    if translation is not None:
        pos = pos + np.asarray(translation)
    parent = ParentSpec(numbers=np.full(7, 3), positions_A=pos, cell_A=cell,
                        mobile_indices=np.arange(7), permutations=np.arange(7)[None],
                        pbc=[False, False, False])
    return build_periodic_graph(parent, cutoff_A=1.01)


def test_adjacent_and_opposite_vacancies_have_different_angular_features():
    context = AngularHistogramContext(ring_graph()).double()
    x = torch.tensor([[0., 0., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 0., 1., 1., 0.]], dtype=torch.float64)
    values = context.step_features(x, 6)
    assert torch.allclose(values[0, :3], values[1, :3])
    # One vacancy pair among six 60-degree pairs, versus one among three
    # 180-degree pairs. The first three added channels are occupied pairs.
    assert values[0, 6].item() == pytest.approx(1/6, abs=1e-7)
    assert values[1, 8].item() == pytest.approx(1/3, abs=1e-7)
    assert values[0, 8].item() == pytest.approx(0.)
    assert values[1, 6].item() == pytest.approx(0.)


def test_no_future_label_leakage_and_teacher_step_agreement():
    context = AngularHistogramContext(ring_graph()).double()
    x = torch.tensor([[1., 0., 1., 0., 0., 1., 0.],
                      [0., 0., 1., 1., 1., 0., 1.]],
                     dtype=torch.float64, requires_grad=True)
    teacher = context.teacher_features(x)
    for i in range(7):
        future = x.detach().clone(); future[:, i:] = float("nan")
        assert torch.allclose(teacher[:, i], context.teacher_features(future)[:, i], atol=1e-12, rtol=0)
        assert torch.allclose(teacher[:, i], context.step_features(future, i), atol=1e-12, rtol=0)
        assigned = (torch.arange(7)[None] < i).expand_as(future)
        assert torch.allclose(teacher[:, i], context.features(future, assigned)[:, i], atol=1e-12, rtol=0)
        grad = torch.autograd.grad(teacher[:, i].sum(), x, retain_graph=True)[0]
        assert torch.count_nonzero(grad[:, i:]) == 0


def test_rotation_reflection_and_translation_leave_features_invariant():
    q, _ = np.linalg.qr(np.array([[.3,.8,.1],[.6,-.2,.5],[.4,.1,-.7]]))
    if np.linalg.det(q) > 0:
        q[:, 0] *= -1
    assert np.linalg.det(q) < 0  # includes a reflection
    base = AngularHistogramContext(ring_graph()).double()
    transformed = AngularHistogramContext(ring_graph(rotation=q, translation=[3.,-2.,.7])).double()
    x = torch.tensor([[1.,0.,0.,1.,1.,1.,0.]], dtype=torch.float64)
    mask = torch.tensor([[1,1,1,1,1,1,0]], dtype=torch.bool)
    assert torch.allclose(base.features(x, mask), transformed.features(x, mask), atol=1e-7, rtol=0)


def test_two_images_of_one_periodic_site_remain_distinct_neighbors():
    parent = ParentSpec(numbers=[3,3], positions_A=[[0.,0.,0.],[1.5,0.,0.]],
                        cell_A=np.diag([3.,5.,5.]), mobile_indices=[0,1],
                        permutations=[[0,1]], pbc=[True,False,False])
    context = AngularHistogramContext(build_periodic_graph(parent,1.6)).double()
    # Two opposite images share site 0's occupancy; both must enter the pair.
    val = context.step_features(torch.tensor([[0.,0.]],dtype=torch.float64),1)
    assert val[0,-1].item() == pytest.approx(1.)


@pytest.mark.parametrize("with_latent", [False, True])
def test_sampler_and_teacher_logits_agree_with_exact_count(with_latent):
    torch.set_num_threads(1)
    context = AngularHistogramContext(ring_graph())
    model = (ARCVAE(n_sites=7, condition_dim=4, latent_dim=4, hidden_dim=16, graph_context=context)
             if with_latent else ARBaseline(n_sites=7, condition_dim=4, hidden_dim=16, graph_context=context))
    c = torch.tensor([[3/7,1.,0.,0.],[3/7,0.,1.,0.]])
    z = torch.randn(2,4) if with_latent else None
    uniforms = torch.rand(2,7); logits=[]
    hook = model.output.register_forward_hook(lambda _m,_a,out: logits.append(out.detach().reshape(2)))
    samples = model.sample(c,3,z=z,uniforms=uniforms);hook.remove()
    assert torch.equal(samples.sum(1),torch.tensor([3.,3.]))
    assert torch.allclose(model.decode_logits(z,c,samples),torch.stack(logits,dim=1),atol=1e-6,rtol=2e-6)


def test_redundant_control_matches_raw_feature_width_and_has_no_angular_input():
    graph=ring_graph(); angular=AngularHistogramContext(graph); control=RedundantRadialContext(graph,6)
    assert angular.feature_dim == control.feature_dim
    x=torch.tensor([[1.,0.,1.,0.,1.,0.,0.]])
    radial=control.radial.teacher_features(x); observed=control.teacher_features(x)
    assert torch.equal(observed[...,:radial.shape[-1]],radial)
    assert torch.equal(observed[...,radial.shape[-1]:],radial.index_select(-1,torch.arange(6)%radial.shape[-1]))


def test_static_buffer_hash_changes_when_pair_geometry_is_modified():
    context=AngularHistogramContext(ring_graph())
    before=context.to_spec()
    context.pair_weights[0,0,0] += .1
    assert context.to_spec()["angular_buffers_sha256"] != before["angular_buffers_sha256"]


@pytest.mark.parametrize("context_kind", ["hist", "redundant", "redundant_learned", "redundant16", "directional", "smooth"])
def test_version5_full_model_roundtrip_and_corrupt_geometry_rejected(tmp_path, context_kind):
    graph = ring_graph()
    if context_kind == "hist":
        context = AngularHistogramContext(graph)
        geometry_buffer = "graph_context.pair_weights"
    elif context_kind in {"redundant", "redundant_learned", "redundant16"}:
        context = RedundantRadialContext(graph, 16 if context_kind == "redundant16" else 6,
                                         learned_mix=context_kind == "redundant_learned")
        geometry_buffer = "graph_context.radial.kernels"
    elif context_kind == "smooth":
        from cathode_cvae.smooth_angular_context import SmoothAngularContext
        context = SmoothAngularContext(graph)
        geometry_buffer = "graph_context.pair_weights"
    else:
        from cathode_cvae.directional_context import DirectionalGraphContext
        context = DirectionalGraphContext(graph)
        geometry_buffer = "graph_context.vector_geometry"
    spec = context.to_spec()
    model = make_model("ar_baseline", n_sites=7, condition_dim=4, latent_dim=0,
                       graph_context=True, graph_data=graph.to_dict(), context_spec=spec)
    cfg = dict(model_format_version=5, kind="ar_baseline", n_sites=7, condition_dim=4,
               latent_dim=0, hidden_dim=64, encoder_kind="dense", graph_context=True,
               graph=graph.to_dict(), decoder_context_spec=spec, remaining_count_features=False,
               energy_target="deletion", label_energy_protocol="synthetic",
               proposal_geometry_protocol="synthetic", beta_schedule=schedule_metadata("reference_updates",.1,20,79))
    path = tmp_path / "model.pt"
    torch.save(dict(configuration=cfg,state_dict=model.state_dict()),path)
    loaded,_ = load_model(path,"cpu")
    c=torch.tensor([[3/7,1.,0.,0.]]); uniforms=torch.rand(1,7)
    assert torch.equal(model.sample(c,3,uniforms=uniforms),loaded.sample(c,3,uniforms=uniforms))
    damaged={key:value.clone() for key,value in model.state_dict().items()}
    damaged[geometry_buffer].reshape(-1)[0] += .1
    torch.save(dict(configuration=cfg,state_dict=damaged),path)
    with pytest.raises(ValueError,match="buffers disagree"):
        load_model(path,"cpu")
