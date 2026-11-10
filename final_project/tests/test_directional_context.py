"""Physical and causal checks for the directional decoder context.

Fixtures use small synthetic geometries. Numerical tolerances reflect FP32
stored graph buffers and are not material or physical convergence thresholds.
"""

import math

import numpy as np
import pytest
import torch

from cathode_cvae.directional_context import DirectionalGraphContext
from cathode_cvae.graph import build_periodic_graph
from cathode_cvae.learning.models import ARBaseline, ARCVAE
from cathode_cvae.parent_spec import ParentSpec


def ring_graph(rotation=None, permutation=None, translation=None):
    """Six candidate sites on a unit hexagon plus a final central site."""
    angles = np.arange(6) * np.pi / 3
    pos = np.vstack((np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6))),
                     np.zeros((1, 3))))
    cell = np.eye(3) * 10
    if rotation is not None:
        pos = pos @ rotation.T
        cell = cell @ rotation.T
    if translation is not None:
        pos = pos + np.asarray(translation)
    if permutation is not None:
        pos = pos[np.asarray(permutation)]
    parent = ParentSpec(numbers=np.full(7, 3), positions_A=pos, cell_A=cell,
                        mobile_indices=np.arange(7), permutations=np.arange(7)[None],
                        pbc=[False, False, False])
    return build_periodic_graph(parent, cutoff_A=1.01)


def copy_learned_weights(source, target):
    # Geometric buffers must remain specific to each transformed graph.
    with torch.no_grad():
        target.vector_weights.copy_(source.vector_weights)
        target.quadrupole_weights.copy_(source.quadrupole_weights)


def test_distinguishes_angular_patterns_with_identical_radial_shell_fractions():
    context = DirectionalGraphContext(ring_graph(), channels=2).double()
    with torch.no_grad():
        context.vector_weights.zero_()
        context.quadrupole_weights.zero_()
        # shell 0, state 1: already assigned vacancies.
        context.vector_weights[0, 1] = 1.
        context.quadrupole_weights[0, 1] = 1.
    # At center 6, both examples have four Li and two vacancies. The two
    # vacancies lie next to each other versus opposite on the hexagon.
    x = torch.tensor([[0., 0., 1., 1., 1., 1., 0.],
                      [0., 1., 1., 0., 1., 1., 0.]], dtype=torch.float64)
    result = context.step_features(x, 6)
    assert torch.allclose(result[0, :3], result[1, :3], rtol=0, atol=1e-12)
    # |(u_0+u_60)/6|^2 = 3/36; opposite vectors sum to zero.
    assert result[0, 3].item() == pytest.approx(3 / 36, abs=1e-7)
    assert result[1, 3].item() == pytest.approx(0., abs=1e-12)
    assert not torch.allclose(result[0, 3:], result[1, 3:], rtol=0, atol=1e-7)


def test_teacher_and_step_agree_without_future_values_or_gradients():
    torch.manual_seed(6)
    context = DirectionalGraphContext(ring_graph()).double()
    x = torch.tensor([[1., 0., 1., 0., 0., 1., 0.],
                      [0., 0., 1., 1., 1., 0., 1.]],
                     dtype=torch.float64, requires_grad=True)
    teacher = context.teacher_features(x)
    assert teacher.shape == (2, 7, 9)  # one radial shell + two 2x2 Gram matrices
    for i in range(7):
        changed = x.detach().clone()
        changed[:, i:] = 1 - changed[:, i:]
        nan_future = changed.clone()
        nan_future[:, i:] = float("nan")
        assert torch.allclose(teacher[:, i], context.teacher_features(changed)[:, i], atol=1e-12, rtol=0)
        assert torch.allclose(teacher[:, i], context.teacher_features(nan_future)[:, i], atol=1e-12, rtol=0)
        assert torch.allclose(teacher[:, i], context.step_features(nan_future, i), atol=1e-12, rtol=0)
        assigned = torch.arange(7)[None] < i
        assert torch.allclose(teacher[:, i], context.features(changed, assigned.expand_as(changed))[:, i], atol=1e-12, rtol=0)
        grad = torch.autograd.grad(teacher[:, i].sum(), x, retain_graph=True)[0]
        assert torch.count_nonzero(grad[:, i:]) == 0


@pytest.mark.parametrize("reflection", [False, True])
def test_messages_transform_equivariantly_and_scalar_features_are_invariant(reflection):
    theta, phi = .37, .63
    rz = np.array([[math.cos(theta), -math.sin(theta), 0.],
                   [math.sin(theta), math.cos(theta), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., math.cos(phi), -math.sin(phi)],
                   [0., math.sin(phi), math.cos(phi)]])
    rotation = rz @ rx
    if reflection:
        rotation = np.diag([-1., 1., 1.]) @ rotation
    original = DirectionalGraphContext(ring_graph()).double()
    transformed = DirectionalGraphContext(ring_graph(rotation=rotation)).double()
    copy_learned_weights(original, transformed)
    r = torch.as_tensor(rotation, dtype=torch.float64)
    expected_v = torch.einsum("ab,...b->...a", r, original.vector_geometry)
    expected_q = torch.einsum("ab,...bc,dc->...ad", r, original.quadrupole_geometry, r)
    assert torch.allclose(expected_v, transformed.vector_geometry, rtol=0, atol=1e-7)
    assert torch.allclose(expected_q, transformed.quadrupole_geometry, rtol=0, atol=1e-7)
    x = torch.tensor([[1., 0., 0., 1., 1., 1., 0.],
                      [1., 1., 0., 1., 0., 0., 1.]], dtype=torch.float64)
    assigned = torch.tensor([[1, 1, 1, 1, 1, 1, 0],
                             [0, 1, 1, 0, 0, 1, 1]], dtype=torch.bool)
    assert torch.allclose(original.features(x, assigned), transformed.features(x, assigned), atol=1e-7, rtol=0)
    assert torch.allclose(original.teacher_features(x), transformed.teacher_features(x), atol=1e-7, rtol=0)


def test_site_relabeling_transforms_features_with_the_assigned_mask():
    permutation = np.array([2, 4, 1, 5, 0, 3, 6])
    first = DirectionalGraphContext(ring_graph()).double()
    second = DirectionalGraphContext(ring_graph(permutation=permutation)).double()
    copy_learned_weights(first, second)
    x = torch.tensor([[1., 0., 0., 1., 1., 1., 0.]], dtype=torch.float64)
    assigned = torch.tensor([[1, 0, 1, 1, 0, 1, 0]], dtype=torch.bool)
    assert torch.allclose(first.features(x, assigned)[:, permutation],
                          second.features(x[:, permutation], assigned[:, permutation]), atol=1e-12, rtol=0)


def test_global_translation_preserves_features():
    first = DirectionalGraphContext(ring_graph()).double()
    second = DirectionalGraphContext(ring_graph(translation=[3.1, -2.3, .75])).double()
    copy_learned_weights(first, second)
    x = torch.tensor([[1., 0., 0., 1., 1., 1., 0.]], dtype=torch.float64)
    assigned = torch.tensor([[1, 0, 1, 1, 0, 1, 0]], dtype=torch.bool)
    assert torch.allclose(first.features(x, assigned), second.features(x, assigned), atol=1e-7, rtol=0)


def test_periodic_images_are_combined_after_directional_expansion():
    # Each center sees two images of the *same* other site in opposite
    # directions. The vector messages cancel but their l=2 tensors reinforce.
    parent = ParentSpec(numbers=[3, 3], positions_A=[[0., 0., 0.], [1.5, 0., 0.]],
                        cell_A=np.diag([3., 5., 5.]), mobile_indices=[0, 1],
                        permutations=[[0, 1]], pbc=[True, False, False])
    context = DirectionalGraphContext(build_periodic_graph(parent, 1.6)).double()
    assert torch.allclose(context.vector_geometry[0, 1, 0], torch.zeros(3, dtype=torch.float64), atol=1e-12, rtol=0)
    expected = torch.diag(torch.tensor([2 / 3, -1 / 3, -1 / 3], dtype=torch.float64))
    assert torch.allclose(context.quadrupole_geometry[0, 1, 0], expected, atol=1e-7, rtol=0)
    assert torch.allclose(context.radial.kernels[0, 1, 0], torch.tensor(1., dtype=torch.float64))


@pytest.mark.parametrize("with_latent", [False, True])
def test_trainable_messages_work_inside_autoregressive_decoder(with_latent):
    torch.set_num_threads(1)
    torch.manual_seed(904)
    context = DirectionalGraphContext(ring_graph())
    model = (ARCVAE(n_sites=7, condition_dim=4, latent_dim=4, hidden_dim=16, graph_context=context)
             if with_latent else ARBaseline(n_sites=7, condition_dim=4, hidden_dim=16, graph_context=context))
    c = torch.tensor([[3 / 7, 1., 0., 0.], [3 / 7, 0., 1., 0.]])
    z = torch.randn(2, 4) if with_latent else None
    uniforms = torch.rand(2, 7)
    outputs = []
    hook = model.output.register_forward_hook(lambda _m, _args, out: outputs.append(out.detach().reshape(2)))
    draws = model.sample(c, 3, z=z, uniforms=uniforms)
    hook.remove()
    assert torch.equal(draws.sum(dim=1), torch.tensor([3., 3.]))
    logits = model.decode_logits(z, c, draws)
    assert torch.allclose(logits, torch.stack(outputs, dim=1), atol=1e-6, rtol=2e-6)
    loss = model.reconstruction(draws, c, z).mean()
    loss.backward()
    assert context.vector_weights.grad is not None and torch.isfinite(context.vector_weights.grad).all()
    assert context.quadrupole_weights.grad is not None and torch.isfinite(context.quadrupole_weights.grad).all()
    assert context.vector_weights.grad.abs().sum().item() > 0
    assert context.quadrupole_weights.grad.abs().sum().item() > 0


def test_checkpoint_roundtrip_and_static_buffer_hashes(tmp_path):
    torch.manual_seed(103)
    original = DirectionalGraphContext(ring_graph(), channels=2)
    checkpoint = tmp_path / "directional.pt"
    torch.save({"spec": original.to_spec(), "state_dict": original.state_dict()}, checkpoint)
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    restored = DirectionalGraphContext(ring_graph(), channels=saved["spec"]["channels"])
    assert restored.to_spec() == saved["spec"]
    restored.load_state_dict(saved["state_dict"], strict=True)
    assert restored.to_spec() == saved["spec"]
    x = torch.tensor([[1., 0., 0., 1., 1., 1., 0.]])
    assert torch.equal(original.teacher_features(x), restored.teacher_features(x))
    with torch.no_grad():
        restored.vector_geometry[0, 0, 1, 0] += 0.1
    assert restored.to_spec()["directional_buffers_sha256"] != saved["spec"]["directional_buffers_sha256"]
