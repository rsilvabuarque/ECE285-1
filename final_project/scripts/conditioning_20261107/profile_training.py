"""Bounded CUDA optimizer timing, not a training-quality experiment.

All updates are temporary engineering probes on one fixed training microbatch.
No weights are saved. Cold load/startup is separate from synchronized step time.
"""
import time

IMPORT_STARTED = time.perf_counter()
import argparse
import gc
import hashlib
import json
import math
import os
from pathlib import Path

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from cathode_cvae.conditions import encode_observations
from cathode_cvae.learning.generate_mask import load_model
from cathode_cvae.learning.models import gaussian_kl
from cathode_cvae.runtime import require_compute_allocation

PROJECT = Path(__file__).resolve().parents[2]


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            h.update(block)
    return h.hexdigest()


def parse_checkpoint(item):
    if "=" not in item:
        raise argparse.ArgumentTypeError("Use LABEL=CHECKPOINT_PATH")
    label, path = item.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("Checkpoint label and path must be nonempty")
    return label, Path(path)


def profile_one(label, path, arrays, data_sha, train_rows, val_rows, args):
    checkpoint_sha = digest(path)
    torch.cuda.synchronize()
    begin = time.perf_counter()
    model, cfg = load_model(path, "cuda")
    torch.cuda.synchronize()
    load_seconds = time.perf_counter()-begin
    if cfg.get("data_sha256") != data_sha:
        raise ValueError("Timing data differ from the checkpoint training archive")
    if cfg.get("descriptor_schema"):
        raise ValueError("This bounded profiler targets the declared energy-only graph/dense pair")
    if cfg["n_sites"] != arrays["occupancy"].shape[1]:
        raise ValueError("Checkpoint and training parent dimensions differ")
    train = train_rows[np.isin(arrays["counts"][train_rows], cfg.get("counts", np.unique(arrays["counts"])))]
    val = val_rows[np.isin(arrays["counts"][val_rows], cfg.get("counts", np.unique(arrays["counts"])))]
    rng = np.random.default_rng(args.seed)
    train_ids = rng.choice(train, args.batch_size, replace=False)
    val_ids = rng.choice(val, min(args.batch_size, len(val)), replace=False)

    def tensors(indices):
        # The archive's explicit canonical vacancy strings supply the trainer's
        # orientation without including canonicalization in optimizer timings.
        keys = arrays["canonical_masks"][indices]
        occupancy = 1-np.asarray([[int(b) for b in k] for k in keys], np.uint8)
        if not np.array_equal(occupancy.sum(1), arrays["counts"][indices]):
            raise ValueError("Canonical strings are not vacancy-one masks with declared counts")
        condition = encode_observations(cfg["condition_schema"], arrays["counts"][indices],
                                        arrays["energy_eV_atom"][indices], mode=cfg["condition_mode"])
        if condition.shape[1] != cfg["condition_dim"]:
            raise ValueError("Profiler condition dimension differs from the saved model")
        return (torch.as_tensor(occupancy, dtype=torch.float32, device="cuda"),
                torch.as_tensor(condition, dtype=torch.float32, device="cuda"))

    begin = time.perf_counter()
    x, c = tensors(train_ids)
    xv, cv = tensors(val_ids)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.get("learning_rate", .001))
    torch.cuda.synchronize()
    batch_prepare_seconds = time.perf_counter()-begin
    beta = float(cfg.get("beta_final", .1))
    exact_count = cfg.get("count_likelihood", cfg["kind"] != "dense_bce")
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    model.train()

    def step():
        optimizer.zero_grad(set_to_none=True)
        mu, logvar = model.encode(x, c)
        z = mu + torch.exp(.5*logvar)*torch.randn_like(mu) if model.latent_dim else None
        reconstruction = model.reconstruction(x, c, z, exact_count=exact_count)
        loss = (reconstruction+beta*gaussian_kl(mu, logvar)).mean()
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite profiling loss")
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10)
        optimizer.step()
        return loss.detach()

    torch.cuda.synchronize(); begin = time.perf_counter()
    for _ in range(args.warmup):
        loss = step()
    torch.cuda.synchronize()
    warmup_seconds = time.perf_counter()-begin
    torch.cuda.reset_peak_memory_stats()
    baseline_allocated = torch.cuda.memory_allocated()
    times = []
    for _ in range(args.steps):
        torch.cuda.synchronize(); begin = time.perf_counter()
        loss = step()
        torch.cuda.synchronize(); times.append(time.perf_counter()-begin)
        if not bool(torch.isfinite(loss)):
            raise FloatingPointError("Nonfinite profiling loss")
    peak_allocated = torch.cuda.max_memory_allocated()
    peak_reserved = torch.cuda.max_memory_reserved()
    model.eval()
    mc = int(cfg.get("validation_samples", 2))
    torch.cuda.synchronize(); begin = time.perf_counter()
    with torch.no_grad():
        mu, logvar = model.encode(xv, cv)
        kl = gaussian_kl(mu, logvar)
        for _ in range(mc):
            z = mu + torch.exp(.5*logvar)*torch.randn_like(mu) if model.latent_dim else None
            _ = model.reconstruction(xv, cv, z, exact_count=exact_count)+beta*kl
    torch.cuda.synchronize()
    validation_batch_seconds = time.perf_counter()-begin
    median, p95 = float(np.median(times)), float(np.quantile(times, .95))
    validation_batches = math.ceil(len(val)/len(val_ids))
    epochs_4000 = math.ceil(4000/math.ceil(len(train)/args.batch_size))
    result = dict(label=label, status="PROFILE_COMPLETE", checkpoint=str(path.resolve()),
        checkpoint_sha256=checkpoint_sha, parameter_count=sum(p.numel() for p in model.parameters()),
        encoder_kind=cfg.get("encoder_kind", "dense"), graph_context=cfg.get("graph_context", False),
        latent_dim=model.latent_dim, train_batch_rows=len(train_ids), validation_batch_rows=len(val_ids),
        train_source_indices_sha256=hashlib.sha256(train_ids.tobytes()).hexdigest(),
        validation_source_indices_sha256=hashlib.sha256(val_ids.tobytes()).hexdigest(),
        checkpoint_load_and_device_transfer_seconds=load_seconds,
        batch_and_optimizer_preparation_seconds=batch_prepare_seconds,
        warmup_optimizer_steps=args.warmup, warmup_seconds=warmup_seconds,
        timed_optimizer_steps=args.steps, optimizer_step_seconds=times,
        optimizer_step_median_seconds=median, optimizer_step_p95_seconds=p95,
        optimizer_only_4000_step_estimate_seconds=dict(median=4000*median, p95=4000*p95),
        validation_batch_MC_samples=mc, validation_one_batch_seconds=validation_batch_seconds,
        validation_batches_per_epoch=validation_batches,
        validation_50_epoch_estimate_seconds=50*validation_batches*validation_batch_seconds,
        expected_epochs_for_4000_updates=epochs_4000,
        validation_4000_updates_estimate_seconds=epochs_4000*validation_batches*validation_batch_seconds,
        median_optimizer_plus_validation_estimate_seconds=4000*median+epochs_4000*validation_batches*validation_batch_seconds,
        allocated_bytes_before_timed_steps=baseline_allocated,
        timed_peak_allocated_bytes=peak_allocated, timed_peak_reserved_bytes=peak_reserved,
        last_microbatch_loss=float(loss), final_beta=beta,
        loss_scope="Unweighted fixed canonical TRAIN microbatch; reconstruction plus beta times summed latent KL, with gradient clipping and fresh Adam",
        exclusions="Estimates exclude imports, Slurm startup, checkpoint/data preparation, canonicalization, batch gathering/epoch shuffling, per-step diagnostic reductions, checkpoint writes, diagnostic serialization and runtime variation; this is not an end-to-end fit-time bound",
        validation_scope="One validation batch with the saved number of MC reconstructions; extrapolated using full-batch cost, not a measured full validation epoch",
        checkpoint_unchanged=digest(path) == checkpoint_sha, weights_saved=False,
        optimizer_updates_are_engineering_only=True)
    if not result["checkpoint_unchanged"]:
        raise ValueError("Protected input checkpoint changed during timing")
    del optimizer, model, x, c, xv, cv, mu, logvar, kl, loss
    gc.collect(); torch.cuda.empty_cache()
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", action="append", type=parse_checkpoint)
    p.add_argument("--data", type=Path, default=PROJECT/"artifacts/current/improvement_v1/training_round0.npz")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--seed", type=int, default=20261107)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args(argv)
    require_compute_allocation()
    if args.output.exists():raise FileExistsError(args.output)
    if args.steps < 1 or args.warmup < 1 or args.batch_size < 1:raise ValueError("Positive step and batch counts required")
    if not torch.cuda.is_available():raise RuntimeError("CUDA timing requires an explicitly GPU-bound compute step")
    torch.set_num_threads(args.threads)
    torch.backends.cuda.matmul.allow_tf32=False; torch.backends.cudnn.allow_tf32=False
    torch.use_deterministic_algorithms(True)
    import_seconds = time.perf_counter()-IMPORT_STARTED
    begin = time.perf_counter(); torch.cuda.init(); torch.cuda.synchronize()
    cuda_context_seconds = time.perf_counter()-begin
    defaults = [("graph", PROJECT/"runs/conditioning_20261107/engineering_graph_smoke/best.pt"),
                ("dense", PROJECT/"runs/conditioning_20261107/fits/balance_1_d8_20261106/best_validation_beta_0p1.pt")]
    checkpoints = args.checkpoint or defaults
    if len({label for label, _ in checkpoints}) != len(checkpoints):raise ValueError("Checkpoint labels must be unique")
    begin = time.perf_counter()
    with np.load(args.data, allow_pickle=False) as z:
        arrays = {key:z[key] for key in ["occupancy", "energy_eV_atom", "canonical_masks", "split"]}
        arrays["counts"] = z["n_target"] if "n_target" in z else z["n_Li"]
    train = np.flatnonzero(arrays["split"] == "train")
    val = np.flatnonzero(np.isin(arrays["split"], ["val", "validation"]))
    data_sha = digest(args.data)
    data_seconds = time.perf_counter()-begin
    results = []
    for label, path in checkpoints:
        begin = time.perf_counter()
        try:result = profile_one(label, path, arrays, data_sha, train, val, args)
        except Exception as error:
            result = dict(label=label, status="FAILED", error_type=type(error).__name__, error=str(error),
                          elapsed_seconds=time.perf_counter()-begin, weights_saved=False)
            gc.collect(); torch.cuda.empty_cache()
        results.append(result); print(json.dumps(result), flush=True)
    receipt = dict(status="PROFILE_COMPLETE" if all(r["status"] == "PROFILE_COMPLETE" for r in results) else "PROFILE_COMPLETE_WITH_FAILURES",
        engineering_only=True, scientific_training_trajectories=0, weights_saved=False,
        python_import_and_argument_seconds=import_seconds, data_load_hash_seconds=data_seconds,
        initial_CUDA_context_seconds=cuda_context_seconds,
        data_sha256=data_sha, source_sha256=digest(__file__), torch=torch.__version__, numpy=np.__version__,
        gpu=torch.cuda.get_device_name(), tf32=False, dtype="float32", deterministic_algorithms=True,
        seed=args.seed, total_elapsed_seconds=time.perf_counter()-IMPORT_STARTED, models=results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:stream.write(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PROFILE_COMPLETE":raise SystemExit(1)


if __name__ == "__main__":main()
