"""Independent geometry/count/causality checks, using synthetic small parents."""

import copy

import numpy as np
import pytest
import torch

from cathode_cvae.parent_spec import ParentSpec
from cathode_cvae.graph import (CausalGraphContext, PeriodicGraph, build_periodic_graph,
                               classify_descriptor, descriptor_symmetry_audit,
                               fit_descriptor_schema, vacancy_pair_fraction)
from cathode_cvae.learning.graph_model import GraphOccupationEncoder


def _square(z=3):
    # Synthetic periodic square of mobile sites, not a claimed cathode material.
    return ParentSpec.from_legacy(dict(numbers=np.full(4, z),
        positions_A=np.array([[0, 0, 0], [2, 0, 0], [2, 2, 0], [0, 2, 0]], float),
        cell_A=np.diag([4., 4., 8.]), target_indices=np.arange(4),
        permutations=np.array([[0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2]])))


def test_parent_roundtrip_hash_chemistry_and_large_site_keys():
    parent = _square(11)
    assert parent.symmetry_audit["status"] == "PASS"
    assert set(parent.numbers) == {11}
    assert ParentSpec.from_dict(parent.to_dict()).parent_hash == parent.parent_hash
    corrupt = copy.deepcopy(parent.to_dict())
    corrupt["positions_A"][0][0] += .1
    with pytest.raises(ValueError, match="hash mismatch"):
        ParentSpec.from_dict(corrupt)
    many = ParentSpec(np.full(70, 11), np.column_stack((np.arange(70), np.zeros((70, 2)))),
                      np.diag([75., 5., 5.]), np.arange(70), np.arange(70)[None])
    key = many.canonical_key(np.array([0, 1] * 35))
    assert len(key) == 70 and key == "10" * 35
    row = parent.example_record([1, 0, 1, 0], teacher="synthetic", checkpoint="synthetic-version",
        head="test", precision="float64", geometry_protocol="synthetic unrelaxed fixture", condition_schema_id="fixture-v1")
    assert row["target_count"] == row["present_atoms"] == 2
    assert row["candidate_sites"] == 4
    assert row["parent_hash"] == parent.parent_hash


def test_target_only_symmetry_rejected_when_host_chemistry_breaks_it():
    data = dict(numbers=np.array([3, 3, 27, 8]),
                positions_A=np.array([[0, 0, 0], [2.5, 0, 0], [.4, .6, .7], [1.7, .9, 1.1]]),
                cell_A=np.eye(3) * 5, target_indices=np.arange(2),
                permutations=np.array([[0, 1], [1, 0]]))
    with pytest.raises(ValueError, match="full host chemistry"):
        ParentSpec.from_legacy(data)


def test_periodic_images_and_nonzero_shift_self_neighbors_are_preserved():
    p = ParentSpec.from_legacy(dict(numbers=[11], positions_A=[[0., 0., 0.]],
                                   cell_A=np.eye(3) * 3, target_indices=[0], permutations=[[0]]))
    graph = build_periodic_graph(p, 3.01)
    assert len(graph.source) == 6
    assert set(map(tuple, graph.image_shifts)) == {(1, 0, 0), (-1, 0, 0), (0, 1, 0),
                                                  (0, -1, 0), (0, 0, 1), (0, 0, -1)}
    assert np.allclose(graph.distances_A, 3.)
    assert graph.shell_audit()["reverse_edge_complete"]
    assert graph.shell_audit()["nonzero_image_self_edges"] == 6
    assert vacancy_pair_fraction([0], graph) == 1.
    assert vacancy_pair_fraction([1], graph) == 0.
    reconstructed = PeriodicGraph.from_dict(graph.to_dict())
    assert reconstructed.to_dict() == graph.to_dict()
    same_geometry = ParentSpec.from_dict(p.to_dict())
    same_geometry.source_sha256 = "a different compressed file container"
    assert build_periodic_graph(same_geometry, 3.01).to_dict()["graph_hash"] == graph.to_dict()["graph_hash"]
    corrupted = copy.deepcopy(graph.to_dict()); corrupted["image_shifts"][0][0] += 1
    with pytest.raises(ValueError, match="modified"):
        PeriodicGraph.from_dict(corrupted)


def test_descriptor_independent_square_count_and_symmetry():
    graph = build_periodic_graph(_square(), 2.01)
    # Four unique endpoint pairs, two cell images of each undirected pair.
    assert descriptor_symmetry_audit(graph)["undirected_image_pairs"] == 8
    assert vacancy_pair_fraction([0, 0, 1, 1], graph) == .25
    assert vacancy_pair_fraction([0, 1, 0, 1], graph) == 0.
    masks = np.array([[0, 0, 1, 1], [0, 1, 0, 1]])
    for perm in graph.parent.permutations:
        assert np.array_equal(vacancy_pair_fraction(masks, graph), vacancy_pair_fraction(masks[:, perm], graph))


def test_descriptor_schema_never_reads_test_values_and_keeps_ties():
    a = np.array([0., .1, .1, .2, .4, .5])
    n = np.full(6, 2)
    split = np.array(["train", "train", "train", "train", "val", "test"])
    schema = fit_descriptor_schema(a, n, split)
    b = a.copy(); b[-2:] = [-1000., 1000.]
    assert schema == fit_descriptor_schema(b, n, split)
    assert schema["training_support"]["2"] == {"low": 3, "high": 1}
    assert classify_descriptor(a[:4], n[:4], schema).tolist() == [0, 0, 0, 1]
    with pytest.raises(ValueError, match="unsupported"):
        classify_descriptor([0.], [3], schema)


def test_causal_context_teacher_generation_agree_and_no_future_leakage():
    context = CausalGraphContext(build_periodic_graph(_square(), 2.9)).double()
    x = torch.tensor([[1., 0., 1., 0.], [0., 0., 1., 1.]], dtype=torch.float64, requires_grad=True)
    features = context.teacher_features(x)
    assert features.shape == (2, 4, 6)
    for i in range(4):
        changed = x.detach().clone(); changed[:, i:] = 1 - changed[:, i:]
        assert torch.allclose(features[:, i], context.teacher_features(changed)[:, i], atol=1e-12, rtol=0)
        assert torch.allclose(features[:, i], context.step_features(changed, i), atol=1e-12, rtol=0)
        missing = changed.clone(); missing[:, i:] = float("nan")
        assert torch.allclose(features[:, i], context.teacher_features(missing)[:, i], atol=1e-12, rtol=0)
        gradient = torch.autograd.grad(features[:, i, 0].sum(), x, retain_graph=True)[0]
        assert torch.equal(gradient[:, i:], torch.zeros_like(gradient[:, i:]))
    explicit = context.features(x, torch.zeros_like(x, dtype=torch.bool))
    assert torch.allclose(explicit[..., [0, 1, 3, 4]], torch.zeros_like(explicit[..., [0, 1, 3, 4]]))
    assert torch.allclose(explicit[..., [2, 5]], torch.ones_like(explicit[..., [2, 5]]))


def test_graph_encoder_gradients_host_chemistry_and_site_count():
    torch.set_num_threads(1)
    graph = build_periodic_graph(_square(11), 2.9)
    torch.manual_seed(217)
    encoder = GraphOccupationEncoder(graph, condition_dim=2, latent_dim=4).double()
    x = torch.tensor([[1., 0., 1., 0.], [1., 1., 0., 0.]], dtype=torch.float64, requires_grad=True)
    c = torch.tensor([[.5, 1.], [.5, 1.]], dtype=torch.float64)
    mu, logvar = encoder(x, c)
    assert mu.shape == logvar.shape == (2, 4)
    assert torch.isfinite(mu).all() and torch.isfinite(logvar).all()
    (mu.square().sum() + logvar.square().sum()).backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in encoder.parameters())
    assert not torch.equal(mu[0], mu[1])
    # Identical rotation-related occupations give identical pooled encodings.
    rotated = encoder(x.detach()[:, graph.parent.permutations[1]], c)[0]
    assert torch.allclose(rotated, mu.detach(), atol=1e-10, rtol=0)
    with pytest.raises(ValueError, match="shape"):
        encoder(x[:, :3], c)


def test_graph_encoder_uses_fixed_host_chemical_identity():
    torch.set_num_threads(1)
    args = dict(positions_A=np.array([[0, 0, 0], [2, 0, 0], [1, 1, .5], [1, 1, 1.5]], float),
                cell_A=np.eye(3) * 6, mobile_indices=np.array([0, 1]), permutations=np.array([[0, 1]]))
    p1 = ParentSpec(numbers=np.array([11, 11, 27, 8]), **args)
    p2 = ParentSpec(numbers=np.array([11, 11, 28, 8]), **args)
    torch.manual_seed(771); first = GraphOccupationEncoder(build_periodic_graph(p1, 3.), 2, 4).double()
    torch.manual_seed(771); second = GraphOccupationEncoder(build_periodic_graph(p2, 3.), 2, 4).double()
    x = torch.tensor([[1., 0.]], dtype=torch.float64)
    c = torch.tensor([[.5, 1.]], dtype=torch.float64)
    assert first.host_indices.tolist() == [2, 3]
    assert not torch.allclose(first(x, c)[0], second(x, c)[0], atol=1e-12, rtol=0)

