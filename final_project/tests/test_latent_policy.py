"""Scientific controls for latent size, paired draws, and one-trajectory selection."""
import json

import numpy as np
import pytest
import torch

from cathode_cvae.learning.generate_mask import load_model
from cathode_cvae.learning.models import make_model
from cathode_cvae.learning.train_mask import latent_statistics, main, selection_checkpoint_name, sha256


@pytest.mark.parametrize("dimension", [4, 8, 16, 32])
def test_explicit_latent_dimension_reload_and_common_random_stream(tmp_path, dimension):
    torch.set_num_threads(1)
    model = make_model("ar_cvae", n_sites=6, condition_dim=4, latent_dim=dimension)
    cfg = dict(model_format_version=2, kind="ar_cvae", n_sites=6,
               condition_dim=4, latent_dim=dimension, hidden_dim=64,
               encoder_kind="dense", graph_context=False, graph=None)
    path = tmp_path / "fit.pt"
    torch.save(dict(state_dict=model.state_dict(), configuration=cfg), path)
    loaded, _ = load_model(path, "cpu")
    c = torch.tensor([[.5, 1, 0, 0], [.5, 0, 1, 0]], dtype=torch.float32)
    z = torch.randn(2, dimension)
    uniforms = torch.rand(2, 6)
    a = model.sample(c, 3, z=z, uniforms=uniforms)
    b = loaded.sample(c, 3, z=z, uniforms=uniforms)
    assert torch.equal(a, b)
    assert torch.equal(a.sum(1), torch.tensor([3., 3.]))
    corrupt = {**cfg, "latent_dim": dimension + 1}
    torch.save(dict(state_dict=model.state_dict(), configuration=corrupt), path)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_model(path, "cpu")
    corrupt = {k: v for k, v in cfg.items() if k != "latent_dim"}
    torch.save(dict(state_dict=model.state_dict(), configuration=corrupt), path)
    with pytest.raises(ValueError, match="missing explicit"):
        load_model(path, "cpu")


def test_no_latent_is_an_explicit_ar_baseline():
    model = make_model("ar_baseline", n_sites=6, latent_dim=32)
    assert model.latent_dim == 0 and not model.has_latent
    mu, logvar = model.encode(torch.zeros(2, 6), torch.zeros(2, 2))
    stats = latent_statistics(mu, logvar)
    assert stats["total_KL_nats"] == 0
    assert stats["KL_nats_by_dimension"] == []
    assert all(v == 0 for v in stats["active_components_by_cutoff"].values())


def test_component_statistics_use_variance_and_report_cutoff_sensitivity():
    mu = torch.tensor([[1., 0.], [-1., 0.]], dtype=torch.float64)
    stats = latent_statistics(mu, torch.zeros_like(mu))
    assert stats["posterior_mean_variance_by_dimension"] == [1., 0.]
    assert stats["KL_nats_by_dimension"] == [.5, 0.]
    assert stats["total_KL_nats"] == .5
    assert stats["mean_KL_nats_per_dimension"] == .25
    assert stats["active_components_by_cutoff"] == {"0.001": 1, "0.01": 1, "0.1": 1}
    assert selection_checkpoint_name(.1) == "best_validation_beta_0p1.pt"
    with pytest.raises(ValueError, match="finite"):
        selection_checkpoint_name(float("nan"))


def test_one_short_trajectory_saves_both_selection_rules_and_never_tests(tmp_path):
    # Synthetic identity-symmetry fixture: this verifies accounting, not LCO quality.
    parent = tmp_path / "parent.npz"
    np.savez(parent, permutations=np.arange(4)[None])
    data = tmp_path / "training.npz"
    occupancy = np.asarray([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
                            [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]], np.uint8)
    np.savez(data, occupancy=occupancy, n_target=np.full(6, 2),
             energy_eV_atom=np.arange(6, dtype=float), weight=np.ones(6),
             split=np.asarray(["train"]*3 + ["val"]*2 + ["test"]),
             structure_ids=np.asarray([f"synthetic_{i}" for i in range(6)]),
             parent_sha256=np.asarray(sha256(parent)))
    output = tmp_path / "fit"
    main(["--data", str(data), "--parent", str(parent), "--output-dir", str(output),
          "--kind", "ar_cvae", "--condition-mode", "multibin", "--latent-dim", "4",
          "--max-updates", "2", "--epochs", "10", "--minimum-epochs", "10",
          "--patience", "10", "--batch-size", "2", "--validation-samples", "1",
          "--device", "cpu", "--threads", "1", "--max-seconds", "60"])
    summary = json.loads((output / "summary.json").read_text())
    assert summary["optimizer_updates"] == 2
    assert not summary["heldout_test_evaluated"]
    assert set(summary["selection_checkpoints"]) == {"1.0", "0.1"}
    for beta in [1., .1]:
        model, cfg = load_model(output / selection_checkpoint_name(beta), "cpu")
        assert model.latent_dim == 4 and cfg["selected_validation_beta"] == beta
    history = [json.loads(line) for line in (output / "latent_history.jsonl").read_text().splitlines()]
    assert history[-1]["pooled"]["rows"] == 2
    assert history[-1]["pooled"]["latent_dim"] == 4


def test_descriptor_requests_append_two_components_and_enforce_joint_support(tmp_path):
    from cathode_cvae.conditions import fit_schema
    from cathode_cvae.descriptor import fit_descriptor_schema
    from cathode_cvae.learning.generate_mask import resolve_request
    energy = fit_schema([2]*3, [0., 1., 2.], ["train"]*3, n_sites=4)
    descriptor = fit_descriptor_schema([.1, .2, .3], [2]*3,
                                      ["low", "middle", "high"], ["train"]*3)
    cfg = dict(model_format_version=2, kind="ar_cvae", n_sites=4, condition_mode="multibin",
               condition_dim=6, latent_dim=4, hidden_dim=64, encoder_kind="dense",
               graph_context=False, graph=None, condition_schema=energy,
               descriptor_schema=descriptor, descriptor_condition_dim=2)
    path = tmp_path / "descriptor.pt"
    model = make_model("ar_cvae", n_sites=4, condition_dim=6, latent_dim=4)
    torch.save(dict(state_dict=model.state_dict(), configuration=cfg), path)
    load_model(path, "cpu")
    vector, *_ = resolve_request(cfg, 2, "low", descriptor_range="low")
    assert vector.tolist() == [.5, 1., 0., 0., 1., 0.]
    with pytest.raises(ValueError, match="unsupported joint"):
        resolve_request(cfg, 2, "low", descriptor_range="high")
    with pytest.raises(ValueError, match="explicit descriptor"):
        resolve_request(cfg, 2, "low")


@pytest.mark.parametrize("encoder_kind,context", [("graph", False), ("dense", True), ("graph", True)])
def test_graph_model_reload_count_and_causal_decoder(tmp_path, encoder_kind, context):
    from cathode_cvae.parent_spec import ParentSpec
    from cathode_cvae.graph import build_periodic_graph
    spec = ParentSpec(numbers=[11, 11, 11, 11, 8],
        positions_A=[[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0], [.5, .5, 1]],
        cell_A=np.eye(3)*3, mobile_indices=np.arange(4), permutations=np.arange(4)[None],
        pbc=[True]*3)
    graph = build_periodic_graph(spec, cutoff_A=1.6).to_dict()
    model = make_model("ar_cvae", n_sites=4, latent_dim=4,
                       encoder_kind=encoder_kind, graph_context=context, graph_data=graph)
    cfg = dict(model_format_version=2, kind="ar_cvae", n_sites=4,
        condition_dim=2, latent_dim=4, hidden_dim=64, encoder_kind=encoder_kind,
        graph_context=context, graph=graph)
    path = tmp_path / "graph.pt"
    torch.save(dict(state_dict=model.state_dict(), configuration=cfg), path)
    loaded, _ = load_model(path, "cpu")
    x = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.]])
    c = torch.tensor([[.5, 1.], [.5, 1.]])
    mu, logvar = loaded.encode(x, c)
    loss = loaded.reconstruction(x, c, mu)
    (loss.mean() + .1 * logvar.square().mean()).backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in loaded.parameters())
    uniforms = torch.rand(2, 4)
    assert torch.equal(model.sample(c, 2, z=mu.detach(), uniforms=uniforms),
                       loaded.sample(c, 2, z=mu.detach(), uniforms=uniforms))
    assert torch.equal(loaded.sample(c, 2).sum(1), torch.tensor([2., 2.]))
    altered = x.clone(); altered[:, 2:] = 1 - altered[:, 2:]
    before = loaded.decode_logits(mu.detach(), c, x)
    after = loaded.decode_logits(mu.detach(), c, altered)
    assert torch.equal(before[:, :3], after[:, :3])
