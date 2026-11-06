"""Bounded occupancy-model fits with frozen lineage splits and train-only conditions.

This learns deletion-mask distributions, not a physical relaxation or energy
gradient. Occupancy is 1 for a present Li. Validation likelihoods with different
count constraints are not a common discovery-quality metric.
"""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from models import gaussian_kl, make_model


HERE = Path(__file__).resolve().parent


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def canonicalize(x, inverse_permutations, batch_size=256):
    """Exact minimum vacancy code; int64 safely represents all 48-bit masks."""
    powers = 1 << torch.arange(47, -1, -1, device=x.device, dtype=torch.int64)
    result = torch.empty_like(x)
    codes = torch.empty(len(x), dtype=torch.int64, device=x.device)
    for start in range(0, len(x), batch_size):
        block = x[start:start + batch_size]
        vacancies = (1 - block).to(torch.int64)
        candidates = vacancies[:, inverse_permutations]
        values = (candidates * powers).sum(-1)
        minimum, witness = values.min(1)
        chosen = candidates[torch.arange(len(block), device=x.device), witness]
        result[start:start + len(block)] = 1 - chosen.to(x.dtype)
        codes[start:start + len(block)] = minimum
    return result, codes.cpu().numpy()


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=HERE / "training_round0.npz")
    parser.add_argument("--parent", type=Path,
                        default=HERE.parent / "data/parents/O3_N48_pilot.npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["dense_bce", "dense_count", "ar_cvae", "ar_baseline"],
                        default="ar_cvae")
    parser.add_argument("--condition-mode", choices=["energy", "composition"], default="energy")
    parser.add_argument("--orientation", choices=["canonical", "augment"], default="canonical")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--counts", type=int, nargs="+", default=[12, 24, 36])
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--beta-final", type=float, default=0.1)
    parser.add_argument("--beta-warmup", type=int, default=20)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--validation-samples", type=int, default=2)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args()


def main():
    a = arguments()
    assert a.epochs > 0 and a.batch_size > 0 and a.validation_samples > 0
    assert a.beta_final >= 0 and a.beta_warmup > 0 and a.max_seconds > 0
    assert a.patience > 0 and a.minimum_epochs > 0
    assert all(0 <= n <= 48 for n in a.counts)
    if a.device == "cuda":
        assert os.environ.get("SLURMD_NODENAME", "").startswith("nid"), "Use a compute allocation"
        assert torch.cuda.is_available(), "CUDA requested but unavailable"
    a.output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    random.seed(a.seed)
    np.random.seed(a.seed)
    torch.manual_seed(a.seed)
    if a.device == "cuda":
        torch.cuda.manual_seed_all(a.seed)
    with np.load(a.data, allow_pickle=False) as archive:
        required = ["occupancy", "n_Li", "energy_eV_atom", "split", "weight", "structure_ids"]
        d = {key: archive[key] for key in required}
    lengths = {len(value) for value in d.values()}
    assert len(lengths) == 1
    keep = np.isin(d["n_Li"], a.counts)
    d = {key: value[keep] for key, value in d.items()}
    assert d["occupancy"].shape == (len(d["n_Li"]), 48)
    assert np.isin(d["occupancy"], [0, 1]).all()
    assert np.array_equal(d["occupancy"].sum(1), d["n_Li"])
    assert np.isfinite(d["energy_eV_atom"]).all()
    assert np.isfinite(d["weight"]).all() and (d["weight"] > 0).all()
    assert len(set(d["structure_ids"].tolist())) == len(d["structure_ids"])
    splits = d["split"].astype(str)
    train = np.flatnonzero(splits == "train")
    val = np.flatnonzero(splits == "val")
    test = np.flatnonzero(splits == "test")
    assert len(train) > 0 and len(val) > 0
    with np.load(a.parent, allow_pickle=False) as archive:
        perms = archive["permutations"]
    assert perms.ndim == 2 and perms.shape[1] == 48
    assert np.array_equal(np.sort(perms, axis=1), np.tile(np.arange(48), (len(perms), 1)))
    inv = torch.as_tensor(np.argsort(perms, axis=1), device=a.device)
    x = torch.as_tensor(d["occupancy"], dtype=torch.float32, device=a.device)
    x, codes = canonicalize(x, inv)
    seen = {}
    for code, split in zip(codes.tolist(), splits.tolist()):
        if code in seen:
            assert seen[code] == split, "A canonical vacancy orbit crosses protected splits"
        seen[code] = split
    assert len(seen) == len(codes), "Input must contain one row per canonical vacancy orbit"
    conditions = np.zeros((len(x), 2), dtype=np.float32)
    conditions[:, 0] = d["n_Li"] / 48
    statistics = {}
    weights = d["weight"].astype(np.float64).copy()
    for n in sorted(set(d["n_Li"].tolist())):
        ids = train[d["n_Li"][train] == n]
        assert len(ids) > 0, "Each requested composition needs training reference energies"
        values = d["energy_eV_atom"][ids]
        q = float(np.sort(values)[math.ceil(0.2 * len(values)) - 1])
        same_count = d["n_Li"] == n
        low = d["energy_eV_atom"] <= q
        conditions[same_count, 1] = low[same_count]
        low_weight_fraction = float(weights[ids[low[ids]]].sum() / weights[ids].sum())
        statistics[int(n)] = dict(train_rows=len(ids), q20_eV_atom=q,
                                  low_fraction_train=float(low[ids].mean()),
                                  low_weight_fraction_train=low_weight_fraction)
        if a.class_balance:
            assert 0 < low_weight_fraction < 1
            weights[ids[low[ids]]] /= 2 * low_weight_fraction
            weights[ids[~low[ids]]] /= 2 * (1 - low_weight_fraction)
    if a.condition_mode == "composition":
        conditions = conditions[:, :1]
    weights /= weights[train].mean()
    c = torch.as_tensor(conditions, dtype=torch.float32, device=a.device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=a.device)
    train_t = torch.as_tensor(train, device=a.device)
    val_t = torch.as_tensor(val, device=a.device)
    model = make_model(a.kind, cond_dim=conditions.shape[1]).to(a.device)
    if a.warm_start:
        assert a.kind.startswith("dense") and a.condition_mode == "energy"
        checkpoint = torch.load(a.warm_start, map_location=a.device, weights_only=False)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    noise_generator = torch.Generator(device=a.device).manual_seed(a.seed + 8191)
    validation_noise = torch.randn((a.validation_samples, len(val), model.latent_dim),
                                   generator=noise_generator, device=a.device)
    exact_count = a.kind != "dense_bce"
    metadata = dict(
        kind=a.kind, condition_mode=a.condition_mode, condition_dim=conditions.shape[1],
        n_sites=48, latent_dim=model.latent_dim, seed=a.seed, counts=a.counts,
        train_rows=len(train), val_rows=len(val), test_rows=len(test),
        excluded_split_rows=int(len(x) - len(train) - len(val) - len(test)),
        data=str(a.data.resolve()), data_sha256=sha256(a.data),
        parent=str(a.parent.resolve()), parent_sha256=sha256(a.parent),
        trainer_sha256=sha256(__file__), model_source_sha256=sha256(HERE / "models.py"),
        source_id_sha256=hashlib.sha256("\n".join(d["structure_ids"].tolist()).encode()).hexdigest(),
        condition_stats=statistics,
        energy_condition="Unweighted train-only bottom20% threshold within Li count; exact ties retained",
        orientation=a.orientation, canonicalization="Exact minimum 48-bit vacancy code under parent permutations",
        symmetry_augmentation="Random retained parent permutation on train batches only" if a.orientation == "augment" else "None; one canonical orientation per orbit",
        class_balance=a.class_balance, class_balance_scope="Train only within each composition; keeps its total base weight",
        input_weight_scope="Inherited row weights; normalized to mean 1 over train rows",
        count_likelihood=exact_count, beta_final=a.beta_final, beta_warmup_epochs=a.beta_warmup,
        maximum_epochs=a.epochs, minimum_epochs=a.minimum_epochs, early_stop_patience=a.patience,
        maximum_seconds=a.max_seconds, batch_size=a.batch_size, learning_rate=a.learning_rate,
        validation_samples=a.validation_samples, validation_noise="Fixed seeded standard-normal samples; no train RNG consumption",
        checkpoint_selection="Minimum validation weighted reconstruction plus KL at beta=1, within this model likelihood only",
        warm_start=None if a.warm_start is None else dict(path=str(a.warm_start.resolve()), sha256=sha256(a.warm_start)),
        torch=torch.__version__, numpy=np.__version__, dtype="float32", tf32=False,
        device=a.device, gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
        scientific_scope="Deletion vacancy masks, not relaxed geometry; no force/stress inputs or energy gradients; no global minimum guarantee",
    )
    (a.output_dir / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    history = []
    best = float("inf")
    best_epoch = 0
    stop_reason = "maximum_epochs"
    for epoch in range(1, a.epochs + 1):
        model.train()
        order = train_t[torch.randperm(len(train_t), device=a.device)]
        totals = np.zeros(4, dtype=np.float64)
        beta = a.beta_final * min(1, epoch / a.beta_warmup)
        for ids in order.split(a.batch_size):
            xx = x[ids]
            if a.orientation == "augment":
                ops = inv[torch.randint(len(inv), (len(ids),), device=a.device)]
                xx = xx.gather(1, ops)
            mu, logvar = model.encode(xx, c[ids])
            z = mu + torch.exp(0.5 * logvar) * torch.randn_like(mu) if model.latent_dim else None
            reconstruction = model.reconstruction(xx, c[ids], z, exact_count=exact_count)
            kl = gaussian_kl(mu, logvar)
            objective = reconstruction + beta * kl
            loss = (objective * w[ids]).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite training loss; inputs and failed attempt remain preserved")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
            totals += np.array([float((objective * w[ids]).sum()),
                                float((reconstruction * w[ids]).sum()),
                                float((kl * w[ids]).sum()), float(w[ids].sum())])
        model.eval()
        val_reconstruction = 0.0
        val_kl = 0.0
        val_weight = 0.0
        with torch.no_grad():
            for start in range(0, len(val_t), a.batch_size):
                ids = val_t[start:start + a.batch_size]
                mu, logvar = model.encode(x[ids], c[ids])
                kl = gaussian_kl(mu, logvar)
                rec = torch.zeros(len(ids), device=a.device)
                for eps in validation_noise[:, start:start + len(ids)]:
                    z = mu + torch.exp(0.5 * logvar) * eps if model.latent_dim else None
                    rec += model.reconstruction(x[ids], c[ids], z, exact_count=exact_count) / a.validation_samples
                val_reconstruction += float((rec * w[ids]).sum())
                val_kl += float((kl * w[ids]).sum())
                val_weight += float(w[ids].sum())
        val_reconstruction /= val_weight
        val_kl /= val_weight
        objective = val_reconstruction + val_kl
        if not math.isfinite(objective):
            raise FloatingPointError("Nonfinite validation objective")
        row = dict(epoch=epoch, beta=beta, train_objective=totals[0] / totals[3],
                   train_reconstruction=totals[1] / totals[3], train_KL=totals[2] / totals[3],
                   val_objective=objective, val_reconstruction=val_reconstruction,
                   val_KL=val_kl, elapsed_seconds=time.perf_counter() - started)
        history.append(row)
        if objective < best - 1e-4:
            best = objective
            best_epoch = epoch
            torch.save(dict(state_dict=model.state_dict(), configuration=metadata,
                            epoch=epoch, val_objective=best), a.output_dir / "best.pt")
        with (a.output_dir / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
        if time.perf_counter() - started >= a.max_seconds:
            stop_reason = "time_cap_after_complete_epoch"
            break
        if epoch >= a.minimum_epochs and epoch - best_epoch >= a.patience:
            stop_reason = "validation_early_stop"
            break
    summary = dict(status="FIT_COMPLETE", stop_reason=stop_reason,
                   completed_epochs=len(history), best_epoch=best_epoch,
                   best_val_objective=best, training_seconds=time.perf_counter() - started,
                   checkpoint_sha256=sha256(a.output_dir / "best.pt"),
                   train_rows=len(train), val_rows=len(val), test_rows=len(test),
                   heldout_test_evaluated=False,
                   generation_quality="Requires new frozen MLIP-oracle proposal evaluation; validation fit is not discovery utility")
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TRAINING_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
