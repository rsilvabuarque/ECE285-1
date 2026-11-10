"""Causality, geometry, and checkpoint checks for smooth angular descriptors.

Fixtures are synthetic and are not material predictions. Comparison tolerances
reflect the FP32 stored geometry buffers, not physical convergence thresholds.
"""

import math

import numpy as np
import pytest
import torch

from cathode_cvae.graph import build_periodic_graph
from cathode_cvae.learning.models import ARBaseline, ARCVAE
from cathode_cvae.parent_spec import ParentSpec
from cathode_cvae.smooth_angular_context import SmoothAngularContext


def two_shell_graph(rotation=None, permutation=None, translation=None):
    """Two synthetic four-site rings around a final central mobile site."""
    near = np.array([[1., 0., 0.], [0., 1., 0.], [-1., 0., 0.], [0., -1., 0.]])
    pos = np.vstack((near, 2 * near, np.zeros((1, 3))))
    cell = np.eye(3) * 10
    if rotation is not None:
        pos = pos @ rotation.T
        cell = cell @ rotation.T
    if translation is not None:
        pos = pos + np.asarray(translation)
    if permutation is not None:
        pos = pos[np.asarray(permutation)]
    parent = ParentSpec(numbers=np.full(9, 3), positions_A=pos, cell_A=cell,
                        mobile_indices=np.arange(9), permutations=np.arange(9)[None],
                        pbc=[False, False, False])
    return build_periodic_graph(parent, cutoff_A=2.01)


def test_cross_shell_angular_patterns_separate_equal_radial_counts():
    context = SmoothAngularContext(two_shell_graph()).double()
    # At center index 8, each row has one occupied near site and one occupied
    # far site. In row 1 their directions align; in row 2 they are opposite.
    # The same-shell occupation counts (and first-shell occupied-pair count)
    # agree, whereas the cross-shell angle distinguishes them.
    x = torch.tensor([[1., 0., 0., 0., 1., 0., 0., 0., 0.],
                      [0., 0., 1., 0., 1., 0., 0., 0., 0.]], dtype=torch.float64)
    result = context.step_features(x, 8)
    radial_dim = context.radial.feature_dim
    assert torch.allclose(result[0, :radial_dim], result[1, :radial_dim], atol=1e-12, rtol=0)
    assert result[0, radial_dim].item() > 0  # alpha0/zeta1/lambda+ favors alignment
    assert result[1, radial_dim].item() == pytest.approx(0., abs=1e-12)
    assert result[0, radial_dim + 1].item() == pytest.approx(0., abs=1e-12)
    assert result[1, radial_dim + 1].item() > 0  # lambda- favors opposite directions
    # Both radial profiles encode the alignment, with distinct normalizations.
    assert result[0, radial_dim + 4].item() != pytest.approx(result[0, radial_dim].item(), abs=1e-6)
    assert context.to_spec()["dimensionless_radial_alphas"] == [0., 2.]
    assert context.to_spec()["feature_dim"] == radial_dim + 16


def test_teacher_and_step_agree_without_future_values_or_gradients():
    context = SmoothAngularContext(two_shell_graph()).double()
    x = torch.tensor([[1., 0., 1., 0., 0., 1., 0., 1., 0.],
                      [0., 0., 1., 1., 1., 0., 1., 0., 1.]],
                     dtype=torch.float64, requires_grad=True)
    teacher = context.teacher_features(x)
    assert teacher.shape == (2, 9, context.feature_dim)
    assert torch.count_nonzero(teacher[:, 0, context.radial.feature_dim:]) == 0
    for i in range(9):
        changed = x.detach().clone()
        changed[:, i:] = 1 - changed[:, i:]
        nan_future = changed.clone()
        nan_future[:, i:] = float("nan")
        assert torch.allclose(teacher[:, i], context.teacher_features(changed)[:, i], atol=1e-12, rtol=0)
        assert torch.allclose(teacher[:, i], context.teacher_features(nan_future)[:, i], atol=1e-12, rtol=0)
        assert torch.allclose(teacher[:, i], context.step_features(nan_future, i), atol=1e-12, rtol=0)
        assigned = torch.arange(9)[None] < i
        assert torch.allclose(teacher[:, i], context.features(changed, assigned.expand_as(changed))[:, i], atol=1e-12, rtol=0)
        grad = torch.autograd.grad(teacher[:, i].sum(), x, retain_graph=True)[0]
        assert torch.count_nonzero(grad[:, i:]) == 0


@pytest.mark.parametrize("reflection", [False, True])
def test_global_rotation_reflection_and_translation_leave_scalar_features_unchanged(reflection):
    theta, phi = .37, .63
    rz = np.array([[math.cos(theta), -math.sin(theta), 0.],
                   [math.sin(theta), math.cos(theta), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., math.cos(phi), -math.sin(phi)],
                   [0., math.sin(phi), math.cos(phi)]])
    rotation = rz @ rx
    if reflection:
        rotation = np.diag([-1., 1., 1.]) @ rotation
    first = SmoothAngularContext(two_shell_graph()).double()
    second = SmoothAngularContext(two_shell_graph(rotation=rotation, translation=[3.1, -2.3, .75])).double()
    x = torch.tensor([[1., 0., 0., 1., 1., 1., 0., 1., 0.],
                      [1., 1., 0., 1., 0., 0., 1., 0., 1.]], dtype=torch.float64)
    assigned = torch.tensor([[1, 1, 1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 0, 0, 1, 1, 0, 1]], dtype=torch.bool)
    assert torch.allclose(first.features(x, assigned), second.features(x, assigned), atol=1e-7, rtol=0)
    assert torch.allclose(first.teacher_features(x), second.teacher_features(x), atol=1e-7, rtol=0)


def test_site_relabeling_transforms_features_with_the_assigned_mask():
    permutation = np.array([2, 4, 1, 5, 0, 3, 7, 6, 8])
    first = SmoothAngularContext(two_shell_graph()).double()
    second = SmoothAngularContext(two_shell_graph(permutation=permutation)).double()
    x = torch.tensor([[1., 0., 0., 1., 1., 1., 0., 1., 0.]], dtype=torch.float64)
    assigned = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 1, 0]], dtype=torch.bool)
    assert torch.allclose(first.features(x, assigned)[:, permutation],
                          second.features(x[:, permutation], assigned[:, permutation]), atol=1e-7, rtol=0)


def test_distinct_periodic_images_of_one_site_form_a_valid_pair():
    # The center sees two images of the same other site along +/- x. They
    # share occupancy but are two geometric neighbors with a 180-degree angle.
    parent = ParentSpec(numbers=[3, 3], positions_A=[[0., 0., 0.], [1.5, 0., 0.]],
                        cell_A=np.diag([3., 5., 5.]), mobile_indices=[0, 1],
                        permutations=[[0, 1]], pbc=[True, False, False])
    context = SmoothAngularContext(build_periodic_graph(parent, 1.6)).double()
    assert context.pair_valid[1].sum().item() == 1
    assert context.pair_left[1, 0].item() == context.pair_right[1, 0].item() == 0
    x = torch.tensor([[1., 0.], [0., 0.]], dtype=torch.float64)
    result = context.step_features(x, 1)
    i = context.radial.feature_dim
    # alpha0/zeta1/lambda+ has zero support at 180 degrees, lambda- is one
    # after fixed-pair normalization. First row is OO; second is VV.
    assert result[0, i].item() == 0.
    assert result[0, i + 1].item() == pytest.approx(1.)
    assert result[1, i + 8].item() == 0.
    assert result[1, i + 8 + 1].item() == pytest.approx(1.)


def test_parent_with_fewer_than_two_neighbors_has_zero_angular_features():
    parent = ParentSpec(numbers=[11, 11], positions_A=[[0., 0., 0.], [1., 0., 0.]],
                        cell_A=np.eye(3) * 5, mobile_indices=[0, 1],
                        permutations=[[0, 1]], pbc=[False, False, False])
    context = SmoothAngularContext(build_periodic_graph(parent, 1.1))
    x = torch.tensor([[1., 0.]])
    assert context.pair_left.shape == (2, 0)
    assert torch.count_nonzero(context.teacher_features(x)[..., context.radial.feature_dim:]) == 0
    assert torch.count_nonzero(context.step_features(x, 1)[..., context.radial.feature_dim:]) == 0


@pytest.mark.parametrize("with_latent", [False, True])
def test_smooth_features_work_inside_autoregressive_decoder(with_latent):
    torch.set_num_threads(1)
    torch.manual_seed(904)
    context = SmoothAngularContext(two_shell_graph())
    assert not list(context.parameters())  # immutable features; decoder weights train
    model = (ARCVAE(n_sites=9, condition_dim=4, latent_dim=4, hidden_dim=16, graph_context=context)
             if with_latent else ARBaseline(n_sites=9, condition_dim=4, hidden_dim=16, graph_context=context))
    c = torch.tensor([[4 / 9, 1., 0., 0.], [4 / 9, 0., 1., 0.]])
    z = torch.randn(2, 4) if with_latent else None
    uniforms = torch.rand(2, 9)
    outputs = []
    hook = model.output.register_forward_hook(lambda _m, _args, out: outputs.append(out.detach().reshape(2)))
    draws = model.sample(c, 4, z=z, uniforms=uniforms)
    hook.remove()
    assert torch.equal(draws.sum(dim=1), torch.tensor([4., 4.]))
    logits = model.decode_logits(z, c, draws)
    assert torch.allclose(logits, torch.stack(outputs, dim=1), atol=1e-6, rtol=2e-6)
    loss = model.reconstruction(draws, c, z).mean()
    loss.backward()
    angular_start = 2 + 4 + (4 if with_latent else 0) + context.radial.feature_dim
    gradient = model.gru.weight_ih_l0.grad[:, angular_start:]
    assert torch.isfinite(gradient).all() and gradient.abs().sum().item() > 0


def test_checkpoint_roundtrip_and_static_buffer_hashes(tmp_path):
    original = SmoothAngularContext(two_shell_graph())
    checkpoint = tmp_path / "smooth.pt"
    torch.save({"spec": original.to_spec(), "state_dict": original.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    spec = saved["spec"]
    assert spec["version"] == "causal-angular-smooth-v1"
    assert spec["kernel_mode"] == "angular_smooth"
    assert len(spec["radial_spec"]["kernels_sha256"]) == 64
    restored = SmoothAngularContext(two_shell_graph(), alphas=spec["dimensionless_radial_alphas"],
                                    zetas=spec["angular_zetas"], lambdas=spec["angular_lambdas"])
    assert restored.to_spec() == spec
    restored.load_state_dict(saved["state_dict"], strict=True)
    assert restored.to_spec() == spec
    x = torch.tensor([[1., 0., 0., 1., 1., 1., 0., 1., 0.]])
    assert torch.equal(original.teacher_features(x), restored.teacher_features(x))
    restored.pair_weights[0, 0, 0] += 0.1
    assert restored.to_spec()["angular_buffers_sha256"] != spec["angular_buffers_sha256"]
    restored.radial.kernels[0, 0, 0] += 0.1
    assert restored.to_spec()["radial_spec"]["kernels_sha256"] != spec["radial_spec"]["kernels_sha256"]


@pytest.mark.parametrize("kwargs", [{"alphas": (-1.,)}, {"alphas": (float('inf'),)},
                                      {"zetas": (1.5,)}, {"lambdas": (1.5,)}])
def test_invalid_angular_basis_parameters_are_rejected(kwargs):
    with pytest.raises(ValueError, match="angular basis"):
        SmoothAngularContext(two_shell_graph(), **kwargs)
