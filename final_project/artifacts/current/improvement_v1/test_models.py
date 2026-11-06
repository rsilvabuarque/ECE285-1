"""Small independent likelihood/count/gradient checks; run in an allocation."""
import argparse
import itertools
import json
from pathlib import Path

import torch

from models import exact_count_nll, gaussian_kl, make_model, sample_exact_count


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--receipt")
    args = ap.parse_args()
    device = torch.device(args.device)
    torch.set_num_threads(1)
    torch.manual_seed(782)
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = False
    checks = []
    logits = torch.tensor([[0.2, -0.4, 1.3, 0.1]], dtype=torch.float64,
                          requires_grad=True, device=device)
    x = torch.tensor([[1., 0., 1., 0.]], dtype=torch.float64, device=device)
    all_binary = torch.tensor(list(itertools.product([0., 1.], repeat=4)),
                              dtype=torch.float64, device=device)
    support = all_binary[all_binary.sum(1) == 2]
    expected = torch.logsumexp(support @ logits[0], 0) - (logits * x).sum()
    actual = exact_count_nll(logits, x)[0]
    assert torch.allclose(actual, expected, atol=1e-11, rtol=0)
    actual.backward()
    gradient = ((torch.softmax(support @ logits.detach()[0], 0)[:, None] * support).sum(0)
                - x[0])
    assert torch.allclose(logits.grad[0], gradient, atol=1e-11, rtol=0)
    checks.append("Exact-K log partition and gradient match exhaustive 4-site enumeration")
    for n in range(5):
        support = all_binary[all_binary.sum(1) == n]
        loss = exact_count_nll(logits.detach().expand(len(support), -1), support)
        assert abs(torch.exp(-loss).sum().item() - 1) < 1e-11
        samples = sample_exact_count(logits.detach().expand(1000, -1), n)
        assert torch.all(samples.sum(1) == n)
    checks.append("Dense conditional distributions normalize for counts 0..4; draws obey counts")
    for kind in ["ar_cvae", "ar_baseline"]:
        model = make_model(kind, n_sites=4).to(device=device, dtype=torch.float64)
        for n in range(5):
            support = all_binary[all_binary.sum(1) == n]
            condition = torch.tensor([n / 4, 1.], dtype=torch.float64,
                                     device=device).expand(len(support), -1)
            latent = torch.zeros(len(support), model.latent_dim, dtype=torch.float64,
                                 device=device)
            loss = model.reconstruction(support, condition, latent, exact_count=True)
            assert abs(torch.exp(-loss).sum().item() - 1) < 1e-10, (kind, n)
            loss.sum().backward()
            assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
        condition = torch.tensor([[n / 4, 1.] for n in range(5)], dtype=torch.float64,
                                 device=device)
        samples = model.sample(condition, torch.arange(5, device=device))
        assert torch.equal(samples.sum(1), torch.arange(5, dtype=torch.float64, device=device))
        empty_mu, empty_lv = model.encode(samples, condition)
        kl = gaussian_kl(empty_mu, empty_lv)
        assert kl.shape == (5,) and torch.isfinite(kl).all()
        if kind == "ar_baseline":
            assert (kl == 0).all() and model.latent_dim == 0
        checks.append(f"{kind}: exhaustive likelihood normalization, finite gradients, count support")
    root = Path(__file__).resolve().parent.parent
    checkpoint = root / "cvae/runs/uma_cvae_beta01_main_20261106/best.pt"
    original = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = make_model("dense_count")
    model.load_state_dict(original["state_dict"], strict=True)
    model = model.to(device)
    condition = torch.tensor([[0.25, 1.], [0.5, 1.], [0.75, 1.]], device=device)
    target = torch.tensor([12, 24, 36], device=device)
    samples = model.sample(condition, target)
    assert torch.equal(samples.sum(1), target.to(samples.dtype))
    mu, lv = model.encode(samples, condition)
    latent = mu + torch.exp(0.5 * lv) * torch.randn_like(mu)
    loss = (model.reconstruction(samples, condition, latent) + gaussian_kl(mu, lv)).mean()
    loss.backward()
    assert all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters())
    checks.append("Original dense state_dict loads strictly; 48-site sample and gradients pass")
    result = {"status": "PASS", "device": str(device), "torch": torch.__version__,
              "checks": checks, "scope": "Numerical/model API checks, not scientific sampler performance"}
    print(json.dumps(result, indent=2), flush=True)
    if args.receipt:
        Path(args.receipt).write_text(json.dumps(result, indent=2) + "\n")


if __name__ == "__main__":
    main()
