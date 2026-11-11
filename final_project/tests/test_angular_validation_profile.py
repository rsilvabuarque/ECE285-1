"""Small CPU-only fixtures for the allocation-guarded validation script."""

import importlib.util
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import torch
from torch import nn

from cathode_cvae.graph import build_periodic_graph
from cathode_cvae.parent_spec import ParentSpec

def load_audit_script():
    # Load the sibling script by path: a regular installed ``scripts`` package
    # can shadow this repository's namespace directory on sys.path.
    path = Path(__file__).resolve().parents[1] / "scripts" / "angular_20261110" / "validate_profile.py"
    spec = importlib.util.spec_from_file_location("angular_validation_profile", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


audit = load_audit_script()


def test_audit_import_is_independent_of_installed_scripts_package(monkeypatch):
    shadow = types.ModuleType("scripts")
    shadow.__path__ = []
    monkeypatch.setitem(sys.modules, "scripts", shadow)
    assert load_audit_script().CONTEXTS == audit.CONTEXTS


def ring_graph():
    angles = np.arange(6) * np.pi / 3
    pos = np.vstack((np.column_stack((np.cos(angles), np.sin(angles), np.zeros(6))),
                     np.zeros((1, 3))))
    permutations = np.stack([np.r_[(np.arange(6) + i) % 6, 6] for i in range(6)])
    parent = ParentSpec(numbers=np.full(7, 3), positions_A=pos, cell_A=np.eye(3) * 10,
                        mobile_indices=np.arange(7), permutations=permutations,
                        pbc=[False, False, False])
    return build_periodic_graph(parent, 1.01)


def periodic_square_graph():
    positions = np.array([[0., 0., 0.], [2., 0., 0.], [2., 2., 0.], [0., 2., 0.]])
    permutations = np.stack([(np.arange(4) + i) % 4 for i in range(4)])
    parent = ParentSpec(numbers=np.full(4, 3), positions_A=positions,
                        cell_A=np.diag([4., 4., 8.]), mobile_indices=np.arange(4),
                        permutations=permutations, pbc=[True, True, True])
    return build_periodic_graph(parent, 2.9)


@pytest.mark.parametrize("name", ["angle_hist", "angular_smooth", "directional_l1_l2"])
def test_validators_cover_prefixes_symmetries_rotations_and_negative_cases(name):
    if name == "angle_hist" and importlib.util.find_spec("cathode_cvae.angular_context") is None:
        pytest.skip("Histogram context is tested in the integrating worktree")
    torch.set_num_threads(1)
    torch.manual_seed(14)
    graph = ring_graph()
    factory = lambda g: audit.make_context(name, g, 1e-5)
    context = factory(graph)
    result = audit.validate_context(context, graph, audit.transformed_graphs(graph), factory,
                                    torch.device("cpu"), patterns=2, permutation_batch=2)
    assert result["status"] == "PASS", result
    assert result["checks"]["teacher_vs_step_all_prefixes"]["cases"] == graph.n_sites
    assert result["checks"]["teacher_future_gradient_all_prefixes"]["max_absolute_future_gradient"] == 0.
    assert result["checks"]["retained_parent_permutations"]["permutations"] == 6
    assert result["checks"]["retained_parent_permutations"]["patterns"] == 2

    # A deliberately leaky channel must fail the same future-label audit.
    class Leaky(nn.Module):
        def __init__(self, g):
            super().__init__(); self.base = factory(g); self.feature_dim = self.base.feature_dim

        def features(self, x, assigned):
            return self.base.features(x, assigned)

        def teacher_features(self, x):
            return self.base.teacher_features(x) + .1 * x[..., None]

        def step_features(self, x, i):
            return self.base.step_features(x, i) + .1 * x[:, i, None]

    leaky = audit.validate_context(Leaky(graph), graph, audit.transformed_graphs(graph), Leaky,
                                   torch.device("cpu"), patterns=1, permutation_batch=3)
    assert leaky["status"] == "FAIL"
    assert not leaky["checks"]["teacher_nan_future_all_prefixes"]["pass"]
    assert not leaky["checks"]["teacher_future_gradient_all_prefixes"]["pass"]


@pytest.mark.parametrize("name", ["angle_hist", "angular_smooth", "directional_l1_l2"])
def test_periodic_images_remain_invariant_under_joint_cell_transforms(name):
    if name == "angle_hist" and importlib.util.find_spec("cathode_cvae.angular_context") is None:
        pytest.skip("Histogram context is tested in the integrating worktree")
    graph = periodic_square_graph()
    assert np.any(graph.image_shifts != 0)
    factory = lambda g: audit.make_context(name, g, 1e-5)
    result = audit.validate_context(factory(graph), graph, audit.transformed_graphs(graph), factory,
                                    torch.device("cpu"), patterns=1, permutation_batch=2)
    assert result["status"] == "PASS", result


def test_requested_missing_context_is_explicit_failure(monkeypatch):
    monkeypatch.setitem(audit.CONTEXTS, "missing", ("cathode_cvae._missing_context", "Unavailable"))
    with pytest.raises(RuntimeError, match="Requested context.*unavailable"):
        audit.make_context("missing", ring_graph(), 1e-5)
    with pytest.raises(ValueError, match="Unknown context"):
        audit.make_context("unknown", ring_graph(), 1e-5)


def test_cli_guard_requires_compute_allocation_and_visible_gpu(monkeypatch):
    monkeypatch.setattr(audit, "require_compute_allocation", lambda: None)
    monkeypatch.delenv("SLURM_JOB_ID", raising=False)
    monkeypatch.delenv("SLURMD_NODENAME", raising=False)
    with pytest.raises(RuntimeError, match="Slurm compute-node"):
        audit.require_slurm_gpu()
    monkeypatch.setenv("SLURM_JOB_ID", "1")
    monkeypatch.setenv("SLURMD_NODENAME", "test-node")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="visible CUDA GPU"):
        audit.require_slurm_gpu()


def test_timing_helper_runs_on_cpu_only_for_synthetic_fixture():
    context = audit.make_context("angular_smooth", ring_graph(), 1e-5)
    profile = audit.profile_context(context, 7, torch.device("cpu"), batch_size=4,
                                    warmup=1, repeats=2, seed=45)
    assert profile["batch_size"] == 4
    assert set(profile["step_sites"]) == {"0", "3", "6"}
    assert len(profile["teacher_all_sites"]["seconds"]) == 2
    assert profile["teacher_all_sites"]["peak_allocated_bytes"] is None


@pytest.mark.parametrize("name", ["angle_hist", "angular_smooth", "directional_l1_l2"])
def test_integrated_v5_loader_and_exact_count_if_available(name, tmp_path, monkeypatch):
    if name == "angle_hist" and importlib.util.find_spec("cathode_cvae.angular_context") is None:
        pytest.skip("Histogram context is tested in the integrating worktree")
    from cathode_cvae.learning import models
    if not hasattr(models, "validate_context_spec"):
        pytest.skip("Integrated v5 model factory/loader is tested in the integrating worktree")
    # CPU test emulates cuDNN's requirement that a gradient-enabled GRU
    # forward must have been executed in train mode. This catches the CUDA
    # failure without claiming to replace the allocation-based GPU test.
    original_gru_forward = nn.GRU.forward
    gradient_forward_modes = []
    def cudnn_mode_guard(module, *args, **kwargs):
        if torch.is_grad_enabled():
            gradient_forward_modes.append(module.training)
            if not module.training:
                raise RuntimeError("cudnn RNN backward can only be called in training mode")
        return original_gru_forward(module, *args, **kwargs)
    monkeypatch.setattr(nn.GRU, "forward", cudnn_mode_guard)
    graph = ring_graph()
    context = audit.make_context(name, graph, 1e-5)
    result = audit.checkpoint_check(context, graph, torch.device("cpu"), tmp_path,
                                    atol=2e-6, rtol=2e-6, seed=451)
    assert result["ar_baseline"]["pass"], result
    assert result["ar_cvae"]["pass"], result
    assert gradient_forward_modes and all(gradient_forward_modes)
    assert context.training is False  # probe restores inference mode
    assert len(result["ar_baseline"]["tampered_static_buffers_rejected"]) == 2
    assert all(result["ar_baseline"]["tampered_static_buffers_rejected"].values())
    assert result["ar_cvae"]["reload_encoder_mean"]["pass"]
    assert result["ar_cvae"]["reload_encoder_logvar"]["pass"]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires a CUDA GPU in a Slurm allocation")
@pytest.mark.parametrize("name", ["angle_hist", "angular_smooth", "directional_l1_l2"])
@pytest.mark.parametrize("cudnn_enabled", [True, False], ids=["cudnn", "native_cuda"])
def test_checkpoint_gradient_probe_with_real_cudnn(name, cudnn_enabled, tmp_path, monkeypatch):
    from cathode_cvae.learning import models
    if name == "angle_hist" and importlib.util.find_spec("cathode_cvae.angular_context") is None:
        pytest.skip("Histogram context is tested in the integrating worktree")
    if not hasattr(models, "validate_context_spec"):
        pytest.skip("Integrated v5 model factory/loader is tested in the integrating worktree")
    if not torch.backends.cudnn.is_available():
        pytest.skip("cuDNN is unavailable")
    audit.require_slurm_gpu()
    # Match the profiler CLI's documented full-FP32 policy. The native CUDA
    # arm is a control for sequence-vs-step cuDNN rounding, not a substitute
    # for testing the cuDNN path used in real runs.
    monkeypatch.setattr(torch.backends.cuda.matmul, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "allow_tf32", False)
    monkeypatch.setattr(torch.backends.cudnn, "enabled", cudnn_enabled)
    graph = ring_graph()
    context = audit.make_context(name, graph, 1e-5).to("cuda:0")
    result = audit.checkpoint_check(context, graph, torch.device("cuda:0"), tmp_path,
                                    atol=2e-6, rtol=2e-6, seed=451)
    assert result["ar_baseline"]["pass"], result
    assert result["ar_cvae"]["pass"], result
