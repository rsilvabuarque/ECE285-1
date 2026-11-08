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

from .models import gaussian_kl, make_model
from .runtime import device_name, trusted_checkpoint
from ..runtime import require_compute_allocation
from ..conditions import (fit_schema, load_schema, save_schema, schema_identity,
                          classify, encode_observations, condition_dimension)


HERE = Path(__file__).resolve().parent
FROZEN_PARENT_SHA256 = "ae8f8bc8d7e767165b0172b396e3c723b15e09f4b79429ac01b7c78fd4741e95"
FROZEN_DATA_SHA256 = {
    "52d4724a066a5fe26d337295fa7dfe19328ad7bc82420841477d9737cabab630",
    "d2c8fffa90628660400420fbb4d81566335578bb18fab49f20b0018d4e31a465",
    "5b7342efbb523dbee9c92c796f0ebd9aa79fdff5f0658056ed3099231e02f7b0",
}


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify_parent_binding(data_path, parent_path, embedded=None, asserted=None):
    """Legacy archives use frozen receipts; new data must name their producer parent."""
    expected = str(np.asarray(embedded).item()) if embedded is not None else None
    if asserted is not None:
        if expected is not None and expected != asserted:
            raise ValueError("Producer parent metadata and --parent-sha256 disagree")
        expected = asserted
    if expected is None and sha256(data_path) in FROZEN_DATA_SHA256:
        expected = FROZEN_PARENT_SHA256
    if expected is None:
        raise ValueError("New training arrays require parent_sha256 metadata or an explicit --parent-sha256 producer assertion; data prepare-learning writes that metadata")
    if expected != sha256(parent_path):
        raise ValueError("Training labels belong to a different parent, even if the site dimensions agree")


def canonicalize(x, inverse_permutations, batch_size=256):
    """Exact minimum vacancy code for fixed parents with at most 62 sites."""
    n_sites = x.shape[1]
    if n_sites > 62:
        raise ValueError("This canonicalization implementation supports at most 62 target sites")
    powers = 1 << torch.arange(n_sites - 1, -1, -1, device=x.device, dtype=torch.int64)
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


def selection_checkpoint_name(beta):
    """Stable filename for a validation rule; beta is dimensionless."""
    if not math.isfinite(beta) or beta < 0:
        raise ValueError("Selection beta must be finite and nonnegative")
    return "best_validation_beta_" + format(beta, ".12g").replace(".", "p") + ".pt"


def latent_statistics(mu, logvar, weights=None, cutoffs=(0.001, 0.01, 0.1)):
    """Weighted VAL latent summaries, not a diagnosis of useful generation.

    KL is in nats and summed over latent coordinates. Active-component counts
    use Var_config(mu_j)>cutoff; the three dimensionless cutoffs are provisional
    descriptive thresholds, reported together rather than presented as a law.
    """
    if mu.shape != logvar.shape or mu.ndim != 2 or len(mu) == 0:
        raise ValueError("Latent summaries need matching nonempty (rows, dimensions) tensors")
    if weights is None:
        weights = mu.new_ones(len(mu))
    if weights.shape != (len(mu),) or not bool((weights > 0).all()):
        raise ValueError("Latent summary weights must be positive per-row values")
    ww = weights / weights.sum()
    mean = (ww[:, None] * mu).sum(0)
    variance = (ww[:, None] * (mu - mean).square()).sum(0)
    kl = 0.5 * (mu.square() + logvar.exp() - 1 - logvar)
    mean_kl = (ww[:, None] * kl).sum(0)
    return dict(rows=len(mu), latent_dim=mu.shape[1],
        total_KL_nats=float(mean_kl.sum()),
        mean_KL_nats_per_dimension=float(mean_kl.mean()) if mu.shape[1] else 0.,
        KL_nats_by_dimension=mean_kl.detach().cpu().tolist(),
        posterior_mean_by_dimension=mean.detach().cpu().tolist(),
        posterior_mean_variance_by_dimension=variance.detach().cpu().tolist(),
        posterior_variance_mean_by_dimension=(ww[:, None] * logvar.exp()).sum(0).detach().cpu().tolist(),
        active_definition="Weighted variance across validation configurations of posterior mean strictly exceeds cutoff",
        active_components_by_cutoff={str(t): int((variance > t).sum()) for t in cutoffs},
        active_cutoff_scope="Dimensionless provisional descriptive sensitivity; no physical convergence or useful-latent guarantee")


def arguments(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--parent-sha256", help="Producer assertion for new archives lacking embedded parent identity")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kind", choices=["dense_bce", "dense_count", "ar_cvae", "ar_baseline"],
                        default="ar_cvae")
    parser.add_argument("--condition-mode", choices=["energy", "composition", "multibin"], default="energy")
    parser.add_argument("--latent-dim", type=int, default=8,
                        help="CVAE latent dimension; --kind ar_baseline always has no latent variables")
    parser.add_argument("--encoder", choices=["dense", "graph"], default="dense")
    parser.add_argument("--graph-context", action="store_true",
                        help="Append causal physical-neighbor features to each AR decoder step")
    parser.add_argument("--graph-cutoff-A", type=float, default=5.0,
                        help="Periodic parent-neighbor construction cutoff in angstrom")
    parser.add_argument("--condition-schema-in", type=Path,
                        help="Frozen train-fitted energy partition; do not refit its edges")
    parser.add_argument("--descriptor-schema-in", type=Path,
                        help="Frozen train-fitted descriptor partition with explicit joint energy support")
    parser.add_argument("--descriptor-values", type=Path,
                        help="NPZ structure_ids and descriptor_values; append descriptor low/high one-hot conditions")
    parser.add_argument("--schema-quantiles", type=float, nargs=2, default=[0.2, 0.8])
    parser.add_argument("--teacher", default="uma", help="Declared label teacher; recorded with pinned head/checkpoint identity")
    parser.add_argument("--teacher-head", help="Explicit label head; default omat for UMA, omat_pbe for MACE")
    parser.add_argument("--geometry-protocol", default="pristine-derived deletion; unrelaxed")
    parser.add_argument("--orientation", choices=["canonical", "augment"], default="canonical")
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--counts", type=int, nargs="+", default=None)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=0.001)
    parser.add_argument("--beta-final", type=float, default=0.1)
    parser.add_argument("--beta-warmup", type=int, default=20)
    parser.add_argument("--validation-beta", type=float, default=1.,
                        help="Checkpoint selection uses reconstruction plus this times KL; historical fits used 1")
    parser.add_argument("--selection-betas", type=float, nargs="+", default=[1., .1],
                        help="Save independent best checkpoints for these validation rules on one training trajectory")
    parser.add_argument("--max-updates", type=int, default=None,
                        help="Optional exact optimizer-update budget for controlled ablations")
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--minimum-epochs", type=int, default=20)
    parser.add_argument("--max-seconds", type=float, default=900)
    parser.add_argument("--validation-samples", type=int, default=2)
    parser.add_argument("--class-balance", action="store_true")
    parser.add_argument("--warm-start", type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu", "auto"], default="auto")
    parser.add_argument("--threads", type=int, default=4)
    return parser.parse_args(argv)


def main(argv=None):
    a = arguments(argv)
    require_compute_allocation()
    a.device = device_name(a.device)
    assert a.epochs > 0 and a.batch_size > 0 and a.validation_samples > 0
    assert a.beta_final >= 0 and a.beta_warmup > 0 and a.max_seconds > 0
    assert a.patience > 0 and a.minimum_epochs > 0
    assert a.validation_beta >= 0
    if a.kind != "ar_baseline" and a.latent_dim < 1:
        raise ValueError("A CVAE needs latent_dim>=1; use --kind ar_baseline for the no-latent AR control")
    if a.kind == "ar_baseline" and a.encoder != "dense":
        raise ValueError("The no-latent AR control has no graph encoder")
    if not math.isfinite(a.graph_cutoff_A) or a.graph_cutoff_A <= 0:
        raise ValueError("Graph cutoff must be finite and positive angstrom")
    selection_betas = sorted(set([a.validation_beta] + a.selection_betas), reverse=True)
    for selection_beta in selection_betas:
        selection_checkpoint_name(selection_beta)
    assert a.max_updates is None or a.max_updates > 0
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
        verify_parent_binding(a.data, a.parent,
                              archive["parent_sha256"] if "parent_sha256" in archive else None,
                              a.parent_sha256)
        required = ["occupancy", "energy_eV_atom", "split", "weight", "structure_ids"]
        d = {key: archive[key] for key in required}
        d["n_Li"] = archive["n_target"] if "n_target" in archive else archive["n_Li"]
    lengths = {len(value) for value in d.values()}
    assert len(lengths) == 1
    n_sites = d["occupancy"].shape[1]
    assert 1 <= n_sites <= 62, "Current bit-code canonicalization supports 1..62 target sites"
    if a.counts is None:
        a.counts = sorted(set(d["n_Li"].astype(int).tolist()))
    assert all(0 <= n <= n_sites for n in a.counts)
    keep = np.isin(d["n_Li"], a.counts)
    d = {key: value[keep] for key, value in d.items()}
    assert d["occupancy"].shape == (len(d["n_Li"]), n_sites)
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
        parent_arrays = {key: archive[key] for key in archive.files}
    assert perms.ndim == 2 and perms.shape[1] == n_sites
    assert np.array_equal(np.sort(perms, axis=1), np.tile(np.arange(n_sites), (len(perms), 1)))
    inv = torch.as_tensor(np.argsort(perms, axis=1), device=a.device)
    x = torch.as_tensor(d["occupancy"], dtype=torch.float32, device=a.device)
    x, codes = canonicalize(x, inv)
    seen = {}
    for code, split in zip(codes.tolist(), splits.tolist()):
        if code in seen:
            assert seen[code] == split, "A canonical vacancy orbit crosses protected splits"
        seen[code] = split
    assert len(seen) == len(codes), "Input must contain one row per canonical vacancy orbit"
    split_hash = hashlib.sha256("\n".join(f"{sid}\t{split}" for sid, split in
                                         zip(d["structure_ids"].tolist(), splits.tolist())).encode()).hexdigest()
    from ..mlip.registry import MODELS
    teacher_spec = MODELS.get(a.teacher, {})
    teacher_head = a.teacher_head or teacher_spec.get("head", "user-declared")
    provenance = dict(teacher=a.teacher, teacher_head=teacher_head,
        teacher_checkpoint_name=teacher_spec.get("name", "user-declared"),
        teacher_checkpoint_hashes=teacher_spec.get("files", {}),
        geometry_protocol=a.geometry_protocol, parent_sha256=sha256(a.parent),
        data_sha256=sha256(a.data), split_sha256=split_hash,
        training_ids_sha256=hashlib.sha256("\n".join(d["structure_ids"][train].tolist()).encode()).hexdigest())
    if a.condition_schema_in:
        schema = load_schema(a.condition_schema_in)
        if schema["n_sites"] != n_sites:
            raise ValueError("Frozen condition schema and parent dimensions differ")
        for key in ["parent_sha256", "teacher", "teacher_head", "geometry_protocol"]:
            old = schema.get("provenance", {}).get(key)
            if old is not None and old != provenance[key]:
                raise ValueError(f"Frozen condition schema has incompatible {key}")
    else:
        schema = fit_schema(d["n_Li"], d["energy_eV_atom"], splits, n_sites=n_sites,
                            quantiles=tuple(a.schema_quantiles), provenance=provenance)
    save_schema(schema, a.output_dir / "condition_schema.json")
    conditions = encode_observations(schema, d["n_Li"], d["energy_eV_atom"], mode=a.condition_mode)
    observed_bins = classify(schema, d["n_Li"], d["energy_eV_atom"])
    descriptor_schema = None
    descriptor_values = None
    if a.descriptor_schema_in is not None and a.descriptor_values is None:
        raise ValueError("A descriptor schema requires aligned --descriptor-values")
    if a.descriptor_values is not None:
        if a.condition_mode != "multibin":
            raise ValueError("The descriptor pilot extends multibin energy conditions explicitly")
        from .. import descriptor as descriptor_module
        with np.load(a.descriptor_values, allow_pickle=False) as archive:
            ids = archive["structure_ids"].astype(str)
            values = archive["descriptor_values"].astype(float)
            descriptor_provenance = {key: str(np.asarray(archive[key]).item())
                for key in ["parent_sha256", "parent_hash", "graph_hash"] if key in archive}
        if descriptor_provenance.get("parent_sha256", sha256(a.parent)) != sha256(a.parent):
            raise ValueError("Descriptor labels belong to a different parent source")
        if ids.ndim != 1 or values.shape != ids.shape or len(set(ids.tolist())) != len(ids):
            raise ValueError("Descriptor values need unique structure_ids and one scalar per row")
        if not np.isfinite(values).all():
            raise ValueError("Descriptor observations must be finite")
        lookup = dict(zip(ids.tolist(), values.tolist()))
        try:
            descriptor_values = np.asarray([lookup[str(s)] for s in d["structure_ids"]], dtype=float)
        except KeyError as failure:
            raise ValueError("Descriptor labels do not cover every retained training/validation/test identity") from failure
        descriptor_schema = (descriptor_module.load_schema(a.descriptor_schema_in)
            if a.descriptor_schema_in is not None else
            descriptor_module.fit_descriptor_schema(descriptor_values, d["n_Li"], observed_bins, splits,
                parent_hash=descriptor_provenance.get("parent_hash"), graph_hash=descriptor_provenance.get("graph_hash")))
        for key in ["parent_hash", "graph_hash"]:
            expected = descriptor_schema.get("provenance", {}).get(key)
            if expected is not None and descriptor_provenance.get(key) != expected:
                raise ValueError("Descriptor observations and fitted schema have different " + key)
        descriptor_module.save_schema(descriptor_schema, a.output_dir / "descriptor_schema.json")
        extra = descriptor_module.encode_observations(descriptor_schema, d["n_Li"], observed_bins, descriptor_values)
        conditions = np.concatenate((conditions, extra), axis=-1)
    unsupported_observations = {split: sum(
        label not in schema["compositions"][str(int(n))]["supported_bins"]
        for n, label, actual in zip(d["n_Li"], observed_bins, splits) if actual == split)
        for split in sorted(set(splits.tolist()))}
    statistics = {}
    weights = d["weight"].astype(np.float64).copy()
    for n in sorted(set(d["n_Li"].tolist())):
        ids = train[d["n_Li"][train] == n]
        assert len(ids) > 0, "Each requested composition needs training reference energies"
        values = d["energy_eV_atom"][ids]
        q20 = float(np.sort(values)[math.ceil(0.2 * len(values)) - 1])
        low = observed_bins == "low"
        low_weight_fraction = float(weights[ids[low[ids]]].sum() / weights[ids].sum())
        classes = observed_bins if a.condition_mode == "multibin" else np.where(low, "low", "rest")
        class_fractions = {label: float(weights[ids[classes[ids] == label]].sum() / weights[ids].sum())
                           for label in sorted(set(classes[ids].tolist()))}
        statistics[int(n)] = dict(train_rows=len(ids), q20_eV_atom=q20,
            low_threshold_eV_atom=schema["compositions"][str(int(n))]["edges_eV_atom"][0],
            low_fraction_train=float(low[ids].mean()), low_weight_fraction_train=low_weight_fraction,
            class_weight_fractions=class_fractions,
            bin_counts_train={label: int(np.sum(observed_bins[ids] == label)) for label in ["low", "middle", "high"]})
        if a.class_balance:
            for label, fraction in class_fractions.items():
                weights[ids[classes[ids] == label]] /= len(class_fractions) * fraction
    weights /= weights[train].mean()
    c = torch.as_tensor(conditions, dtype=torch.float32, device=a.device)
    w = torch.as_tensor(weights, dtype=torch.float32, device=a.device)
    train_t = torch.as_tensor(train, device=a.device)
    val_t = torch.as_tensor(val, device=a.device)
    graph_data = None
    if a.encoder == "graph" or a.graph_context:
        from ..parent_spec import ParentSpec
        from ..graph import build_periodic_graph
        parent_spec = ParentSpec.from_legacy(parent_arrays, source_sha256=sha256(a.parent))
        graph_data = build_periodic_graph(parent_spec, cutoff_A=a.graph_cutoff_A).to_dict()
    model = make_model(a.kind, n_sites=n_sites, cond_dim=conditions.shape[1],
                       latent_dim=a.latent_dim, encoder_kind=a.encoder,
                       graph_data=graph_data, graph_context=a.graph_context).to(a.device)
    if a.warm_start:
        checkpoint = trusted_checkpoint(a.warm_start, a.device)
        prior_cfg = checkpoint.get("configuration", {})
        if prior_cfg.get("condition_dim", model.condition_dim) != model.condition_dim:
            raise ValueError("Warm-start condition dimensions differ; binary weights cannot silently become multibin weights")
        if prior_cfg.get("latent_dim", 8) != model.latent_dim:
            raise ValueError("Warm-start latent dimensions differ")
        if prior_cfg.get("n_sites", 48) != n_sites:
            raise ValueError("Warm-start parent site dimensions differ")
        if prior_cfg.get("encoder_kind", "dense") != a.encoder or prior_cfg.get("graph_context", False) != a.graph_context:
            raise ValueError("Warm-start graph architecture differs")
        if prior_cfg.get("parent_sha256") not in {None, sha256(a.parent)}:
            raise ValueError("Warm-start parent identity differs")
        model.load_state_dict(checkpoint["state_dict"], strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=a.learning_rate)
    noise_generator = torch.Generator(device=a.device).manual_seed(a.seed + 8191)
    validation_noise = torch.randn((a.validation_samples, len(val), model.latent_dim),
                                   generator=noise_generator, device=a.device)
    exact_count = a.kind != "dense_bce"
    metadata = dict(
        model_format_version=2, kind=a.kind, condition_mode=a.condition_mode, condition_dim=conditions.shape[1],
        n_sites=n_sites, latent_dim=model.latent_dim, seed=a.seed, counts=a.counts,
        hidden_dim=64, parameter_count=sum(p.numel() for p in model.parameters()),
        encoder_kind=a.encoder, graph_context=a.graph_context, graph=graph_data,
        graph_cutoff_A=a.graph_cutoff_A if graph_data is not None else None,
        train_rows=len(train), val_rows=len(val), test_rows=len(test),
        excluded_split_rows=int(len(x) - len(train) - len(val) - len(test)),
        data=str(a.data.resolve()), data_sha256=sha256(a.data),
        parent=str(a.parent.resolve()), parent_sha256=sha256(a.parent),
        trainer_sha256=sha256(__file__), model_source_sha256=sha256(HERE / "models.py"),
        source_id_sha256=hashlib.sha256("\n".join(d["structure_ids"].tolist()).encode()).hexdigest(),
        condition_stats=statistics, condition_schema=schema,
        descriptor_schema=descriptor_schema,
        descriptor_condition_dim=0 if descriptor_schema is None else 2,
        descriptor_values_source=None if a.descriptor_values is None else dict(path=str(a.descriptor_values.resolve()), sha256=sha256(a.descriptor_values)),
        descriptor_schema_source=None if a.descriptor_schema_in is None else dict(path=str(a.descriptor_schema_in.resolve()), sha256=sha256(a.descriptor_schema_in)),
        condition_schema_id=schema_identity(schema), split_sha256=split_hash,
        observed_rows_outside_training_bin_support=unsupported_observations,
        condition_schema_source=None if a.condition_schema_in is None else dict(path=str(a.condition_schema_in.resolve()), sha256=sha256(a.condition_schema_in)),
        teacher=a.teacher, teacher_head=teacher_head,
        teacher_checkpoint_name=teacher_spec.get("name", "user-declared"),
        teacher_checkpoint_hashes=teacher_spec.get("files", {}), geometry_protocol=a.geometry_protocol,
        energy_condition="Versioned train-only composition-specific energy partition; exact ties retained; encoder mode recorded separately",
        orientation=a.orientation, canonicalization="Exact minimum vacancy bit code under supplied parent permutations",
        symmetry_augmentation="Random retained parent permutation on train batches only" if a.orientation == "augment" else "None; one canonical orientation per orbit",
        class_balance=a.class_balance, class_balance_scope="Train only within each composition and actual nonempty energy classes; descriptor classes are not separately reweighted; keeps total base weight",
        input_weight_scope="Inherited row weights; normalized to mean 1 over train rows",
        count_likelihood=exact_count, beta_final=a.beta_final, beta_warmup_epochs=a.beta_warmup,
        maximum_epochs=a.epochs, minimum_epochs=a.minimum_epochs, early_stop_patience=a.patience,
        maximum_seconds=a.max_seconds, batch_size=a.batch_size, learning_rate=a.learning_rate,
        validation_samples=a.validation_samples, validation_noise="Fixed seeded standard-normal samples; no train RNG consumption",
        checkpoint_selection="Minimum validation weighted reconstruction plus validation_beta*KL, within this model likelihood only",
        validation_beta=a.validation_beta, maximum_updates=a.max_updates,
        saved_checkpoint_selection_betas=selection_betas,
        KL_convention="Per-row sum across latent dimensions, then row-weighted mean; never divided by latent dimension in training",
        latent_activity_cutoffs=[0.001, 0.01, 0.1],
        latent_activity_scope="Descriptive VAL posterior-mean variance sensitivity; no test rows, not physical convergence",
        warm_start=None if a.warm_start is None else dict(path=str(a.warm_start.resolve()), sha256=sha256(a.warm_start)),
        torch=torch.__version__, numpy=np.__version__, dtype="float32", tf32=False,
        device=a.device, gpu=torch.cuda.get_device_name() if a.device == "cuda" else None,
        scientific_scope="Deletion vacancy masks, not relaxed geometry; no force/stress inputs or energy gradients; no global minimum guarantee",
    )
    (a.output_dir / "configuration.json").write_text(json.dumps(metadata, indent=2) + "\n")
    history = []
    best = float("inf")
    best_epoch = 0
    selections = {b: dict(best=float("inf"), epoch=0) for b in selection_betas}
    stop_reason = "maximum_epochs"
    updates = 0
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
            updates += 1
            totals += np.array([float((objective * w[ids]).sum().detach()),
                                float((reconstruction * w[ids]).sum().detach()),
                                float((kl * w[ids]).sum().detach()), float(w[ids].sum())])
            if a.max_updates is not None and updates >= a.max_updates:
                break
        model.eval()
        val_reconstruction = 0.0
        val_kl = 0.0
        val_weight = 0.0
        latent_mu, latent_logvar, latent_weights = [], [], []
        with torch.no_grad():
            for start in range(0, len(val_t), a.batch_size):
                ids = val_t[start:start + a.batch_size]
                mu, logvar = model.encode(x[ids], c[ids])
                latent_mu.append(mu)
                latent_logvar.append(logvar)
                latent_weights.append(w[ids])
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
        objective = val_reconstruction + a.validation_beta * val_kl
        if not math.isfinite(objective):
            raise FloatingPointError("Nonfinite validation objective")
        row = dict(epoch=epoch, optimizer_updates=updates, beta=beta, train_objective=totals[0] / totals[3],
                   train_reconstruction=totals[1] / totals[3], train_KL=totals[2] / totals[3],
                   val_objective=objective, val_reconstruction=val_reconstruction,
                   val_KL=val_kl, elapsed_seconds=time.perf_counter() - started)
        for selection_beta in selection_betas:
            row["val_objective_beta_" + format(selection_beta, ".12g").replace(".", "p")] = val_reconstruction + selection_beta * val_kl
        history.append(row)
        with torch.no_grad():
            all_mu, all_logvar, all_weights = torch.cat(latent_mu), torch.cat(latent_logvar), torch.cat(latent_weights)
            latent_record = dict(epoch=epoch, optimizer_updates=updates,
                                 pooled=latent_statistics(all_mu, all_logvar, all_weights),
                                 by_composition={})
            for n in a.counts:
                take = torch.as_tensor(d["n_Li"][val] == n, device=a.device)
                if bool(take.any()):
                    latent_record["by_composition"][str(n)] = latent_statistics(all_mu[take], all_logvar[take], all_weights[take])
        with (a.output_dir / "latent_history.jsonl").open("a") as handle:
            handle.write(json.dumps(latent_record) + "\n")
        for selection_beta in selection_betas:
            score = val_reconstruction + selection_beta * val_kl
            if score < selections[selection_beta]["best"] - 1e-4:
                selections[selection_beta] = dict(best=score, epoch=epoch)
                selected_metadata = {**metadata, "selected_validation_beta": selection_beta}
                checkpoint = dict(state_dict=model.state_dict(), configuration=selected_metadata,
                                  epoch=epoch, optimizer_updates=updates, val_objective=score,
                                  selection_beta=selection_beta, latent_statistics=latent_record)
                torch.save(checkpoint, a.output_dir / selection_checkpoint_name(selection_beta))
                (a.output_dir / (selection_checkpoint_name(selection_beta) + ".latent.json")).write_text(json.dumps(latent_record, indent=2) + "\n")
        if objective < best - 1e-4:
            best = objective
            best_epoch = epoch
            torch.save(dict(state_dict=model.state_dict(), configuration={**metadata, "selected_validation_beta": a.validation_beta},
                            epoch=epoch, optimizer_updates=updates, val_objective=best,
                            selection_beta=a.validation_beta, latent_statistics=latent_record), a.output_dir / "best.pt")
        with (a.output_dir / "history.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            writer.writeheader()
            writer.writerows(history)
        print(json.dumps(row), flush=True)
        if a.max_updates is not None and updates >= a.max_updates:
            stop_reason = "optimizer_update_cap"
            break
        if time.perf_counter() - started >= a.max_seconds:
            stop_reason = "time_cap_after_complete_epoch"
            break
        if epoch >= a.minimum_epochs and epoch - best_epoch >= a.patience:
            stop_reason = "validation_early_stop"
            break
    summary = dict(status="FIT_COMPLETE", stop_reason=stop_reason,
                   completed_epochs=len(history), optimizer_updates=updates, best_epoch=best_epoch,
                   best_val_objective=best, training_seconds=time.perf_counter() - started,
                   checkpoint_sha256=sha256(a.output_dir / "best.pt"),
                   parameter_count=metadata["parameter_count"], latent_dim=model.latent_dim,
                   selection_checkpoints={str(b): dict(epoch=selections[b]["epoch"], val_objective=selections[b]["best"],
                       filename=selection_checkpoint_name(b), sha256=sha256(a.output_dir / selection_checkpoint_name(b))) for b in selection_betas},
                   train_rows=len(train), val_rows=len(val), test_rows=len(test),
                   heldout_test_evaluated=False,
                   generation_quality="Requires new frozen MLIP-oracle proposal evaluation; validation fit is not discovery utility")
    (a.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("TRAINING_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
