"""Validate and profile angular decoder contexts on a supplied fixed parent.

Run the CLI only inside a Slurm GPU allocation. It reads parent geometry and
symmetries, creates synthetic binary occupations, and performs no potential,
DFT, MD, model-fitting, or saved-label calls. The timing measures context
inference, not an optimizer step or end-to-end autoregressive sampling.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import math
import os
from pathlib import Path
import socket
import sys
import tempfile
import time
import traceback

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
import numpy as np
import torch

from cathode_cvae.graph import build_periodic_graph
from cathode_cvae.learning.generate_mask import load_model
from cathode_cvae.learning.graph_model import GraphOccupationEncoder
from cathode_cvae.learning.models import ARBaseline, ARCVAE
from cathode_cvae.parent_spec import ParentSpec
from cathode_cvae.runtime import require_compute_allocation


CONTEXTS = {
    "angle_hist": ("cathode_cvae.angular_context", "AngularHistogramContext"),
    "angular_smooth": ("cathode_cvae.smooth_angular_context", "SmoothAngularContext"),
    "directional_l1_l2": ("cathode_cvae.directional_context", "DirectionalGraphContext"),
}


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    return digest.hexdigest()


def require_slurm_gpu():
    """Disallow profiling on login nodes or outside a visible GPU allocation."""
    require_compute_allocation()
    if not os.environ.get("SLURM_JOB_ID") or not os.environ.get("SLURMD_NODENAME"):
        raise RuntimeError("Run inside a Slurm compute-node job/step with SLURM_JOB_ID and SLURMD_NODENAME")
    if not torch.cuda.is_available():
        raise RuntimeError("A visible CUDA GPU is required for this allocation-only profiler")


def load_parent(path):
    """Load legacy NPZ or hashed ParentSpec JSON; audit full host for NPZ."""
    path = Path(path)
    if path.suffix.lower() == ".json":
        return ParentSpec.from_dict(json.loads(path.read_text()))
    if path.suffix.lower() != ".npz":
        raise ValueError("Parent must be a legacy .npz or a ParentSpec .json")
    keys = {"numbers", "positions_A", "cell_A", "target_indices", "li_indices",
            "permutations", "pbc", "site_groups"}
    with np.load(path, allow_pickle=False) as archive:
        source = {key: archive[key] for key in archive.files if key in keys}
    return ParentSpec.from_legacy(source, file_hash(path))


def make_context(name, graph, shell_tolerance_A):
    if name not in CONTEXTS:
        raise ValueError(f"Unknown context {name!r}; choose from {sorted(CONTEXTS)}")
    module_name, class_name = CONTEXTS[name]
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as exc:
        if exc.name == module_name:
            raise RuntimeError(f"Requested context {name!r} is unavailable: missing {module_name}") from exc
        raise
    try:
        cls = getattr(module, class_name)
    except AttributeError as exc:
        raise RuntimeError(f"Requested context {name!r} is unavailable: missing {class_name}") from exc
    return cls(graph, shell_tolerance_A=shell_tolerance_A)


def transformed_parent(parent, rotation, translation):
    """Apply one O(3) transform jointly to all coordinates and lattice rows."""
    return ParentSpec(numbers=parent.numbers.copy(),
                      positions_A=parent.positions_A @ rotation.T + translation,
                      cell_A=parent.cell_A @ rotation.T,
                      mobile_indices=parent.mobile_indices.copy(),
                      permutations=parent.permutations.copy(), pbc=parent.pbc.copy(),
                      site_groups=None if parent.site_groups is None else parent.site_groups.copy())


def transformed_graphs(graph):
    theta, phi = .37, .63
    rz = np.array([[math.cos(theta), -math.sin(theta), 0.],
                   [math.sin(theta), math.cos(theta), 0.], [0., 0., 1.]])
    rx = np.array([[1., 0., 0.], [0., math.cos(phi), -math.sin(phi)],
                   [0., math.sin(phi), math.cos(phi)]])
    rotation = rz @ rx
    reflection = np.diag([-1., 1., 1.]) @ rotation
    translation = np.array([3.1, -2.3, .75])
    return {
        "rotation_translation": build_periodic_graph(
            transformed_parent(graph.parent, rotation, translation), graph.cutoff_A),
        "reflection_translation": build_periodic_graph(
            transformed_parent(graph.parent, reflection, translation), graph.cutoff_A),
    }


def copy_parameters(source, target):
    """Share learned context weights, keeping transformed geometry buffers."""
    source_parameters = dict(source.named_parameters())
    target_parameters = dict(target.named_parameters())
    if source_parameters.keys() != target_parameters.keys():
        raise ValueError("Transformed context has different trainable parameter names")
    with torch.no_grad():
        for name, target_parameter in target_parameters.items():
            target_parameter.copy_(source_parameters[name])


def comparison(expected, actual, atol, rtol):
    """Dimensionless FP tolerance check; report both raw and scaled error."""
    if expected.shape != actual.shape:
        return {"pass": False, "reason": "shape mismatch",
                "expected_shape": list(expected.shape), "actual_shape": list(actual.shape)}
    if not bool(torch.isfinite(expected).all() and torch.isfinite(actual).all()):
        return {"pass": False, "reason": "nonfinite values"}
    if expected.numel() == 0:
        return {"pass": True, "max_abs_error": 0., "max_tolerance_ratio": 0.}
    delta = (expected - actual).abs()
    allowed = atol + rtol * expected.abs()
    ratio = torch.where(allowed > 0, delta / allowed,
                        torch.where(delta == 0, torch.zeros_like(delta), torch.full_like(delta, float("inf"))))
    max_abs = float(delta.max().item())
    max_ratio = float(ratio.max().item())
    return {"pass": bool(torch.all(delta <= allowed).item()), "max_abs_error": max_abs,
            "max_tolerance_ratio": max_ratio if math.isfinite(max_ratio) else None}


def combine_comparisons(items):
    items = list(items)
    return {"pass": bool(items) and all(item["pass"] for item in items),
            "cases": len(items),
            "max_abs_error": max((item.get("max_abs_error", 0.) for item in items), default=0.),
            "max_tolerance_ratio": max((item.get("max_tolerance_ratio") or 0. for item in items), default=0.),
            "failures": [item for item in items if not item["pass"]][:5]}


def validate_context(context, graph, transformed, context_factory, device,
                     atol=2e-6, rtol=2e-6, seed=20261110,
                     patterns=2, permutation_batch=32):
    """All-prefix causality and descriptor-only parent symmetry checks."""
    context = context.to(device).eval()
    n = graph.n_sites
    if patterns < 1 or permutation_batch < 1:
        raise ValueError("patterns and permutation_batch must be positive")
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randint(0, 2, (patterns, n), generator=rng, device=device).float()
    assigned = torch.randint(0, 2, (patterns, n), generator=rng, device=device).bool()
    with torch.no_grad():
        teacher = context.teacher_features(x)
        reference = context.features(x, assigned)
    if teacher.shape != (patterns, n, context.feature_dim) or reference.shape != teacher.shape:
        raise ValueError("Context output shape disagrees with its declared feature_dim")

    teacher_step, suffix_changed, teacher_nan, step_nan = [], [], [], []
    with torch.no_grad():
        for i in range(n):
            expected = teacher[:, i]
            teacher_step.append(comparison(expected, context.step_features(x, i), atol, rtol))
            changed = x.clone(); changed[:, i:] = 1 - changed[:, i:]
            suffix_changed.append(comparison(expected, context.teacher_features(changed)[:, i], atol, rtol))
            missing = x.clone(); missing[:, i:] = float("nan")
            teacher_nan.append(comparison(expected, context.teacher_features(missing)[:, i], atol, rtol))
            step_nan.append(comparison(expected, context.step_features(missing, i), atol, rtol))

    # Weight every output differently so a signed sum is unlikely to hide a
    # leaky channel. The input is finite; future gradients should be exactly 0.
    differentiable = x[:1].detach().clone().requires_grad_(True)
    output = context.teacher_features(differentiable)
    probe = torch.linspace(.37, 1.43, context.feature_dim, device=device, dtype=x.dtype)
    gradient_errors = []
    for i in range(n):
        grad = torch.autograd.grad((output[:, i] * probe).sum(), differentiable,
                                   retain_graph=i < n - 1)[0]
        future = grad[:, i:]
        gradient_errors.append(float(future.abs().max().item()) if bool(torch.isfinite(future).all()) else float("nan"))
    finite_grad = all(math.isfinite(x) for x in gradient_errors)
    max_gradient = max(gradient_errors) if finite_grad else None

    # The permutation applies to the entire parent occupation and assignment
    # mask. The comparison is for the local context, not fixed-order sampling.
    permutations = torch.as_tensor(graph.parent.permutations, dtype=torch.long, device=device)
    symmetry = []
    with torch.no_grad():
        for start in range(0, len(permutations), permutation_batch):
            p = permutations[start:start + permutation_batch]
            b, count = patterns, len(p)
            px = x[:, p].reshape(b * count, n)
            pa = assigned[:, p].reshape(b * count, n)
            expected = reference[:, p].reshape(b * count, n, context.feature_dim)
            symmetry.append(comparison(expected, context.features(px, pa), atol, rtol))

    transforms = {}
    with torch.no_grad():
        for name, other_graph in transformed.items():
            other = context_factory(other_graph).to(device).eval()
            copy_parameters(context, other)
            transforms[name] = comparison(reference, other.features(x, assigned), atol, rtol)

    checks = {
        "teacher_vs_step_all_prefixes": combine_comparisons(teacher_step),
        "teacher_changed_future_all_prefixes": combine_comparisons(suffix_changed),
        "teacher_nan_future_all_prefixes": combine_comparisons(teacher_nan),
        "step_nan_future_all_prefixes": combine_comparisons(step_nan),
        "teacher_future_gradient_all_prefixes": {
            "pass": finite_grad and max_gradient == 0., "cases": n,
            "max_absolute_future_gradient": max_gradient,
            "method": "one deterministic weighted feature probe per prefix; not a full Jacobian",
        },
        "retained_parent_permutations": {**combine_comparisons(symmetry),
                                          "permutations": len(permutations),
                                          "patterns": patterns,
                                          "scope": "local features with occupation and assignment permuted together"},
        **transforms,
    }
    return {"status": "PASS" if all(check["pass"] for check in checks.values()) else "FAIL",
            "feature_dim": context.feature_dim, "checks": checks}


def checkpoint_check(context, graph, device, directory, atol, rtol, seed):
    """Exercise integrated v5 loader and exact counts for AR and graph CVAE."""
    n, condition_dim, hidden_dim, latent_dim = graph.n_sites, 4, 64, 4
    counts = torch.as_tensor([0, min(1, n), n // 2, n], dtype=torch.long, device=device)
    c = torch.zeros((4, condition_dim), device=device)
    c[:, 0] = counts / n; c[:, 1] = 1.
    rng = torch.Generator(device=device).manual_seed(seed)
    uniforms = torch.rand((4, n), device=device, generator=rng)
    z = torch.randn((4, latent_dim), device=device, generator=rng)
    result = {}
    for kind in ("ar_baseline", "ar_cvae"):
        is_cvae = kind == "ar_cvae"
        encoder = GraphOccupationEncoder(graph, condition_dim, latent_dim) if is_cvae else None
        model = (ARCVAE(n_sites=n, condition_dim=condition_dim, latent_dim=latent_dim,
                        hidden_dim=hidden_dim, graph_encoder=encoder, graph_context=context)
                 if is_cvae else ARBaseline(n_sites=n, condition_dim=condition_dim,
                                             hidden_dim=hidden_dim, graph_context=context)).to(device).eval()
        fixed_z = z if is_cvae else None
        outputs = []
        hook = model.output.register_forward_hook(lambda _m, _args, out: outputs.append(out.detach().reshape(4)))
        with torch.no_grad():
            expected = model.sample(c, counts, z=fixed_z, uniforms=uniforms)
        hook.remove()
        with torch.no_grad():
            teacher = model.decode_logits(fixed_z, c, expected)
        step_logits = torch.stack(outputs, dim=1)
        logits_agree = comparison(teacher, step_logits, atol, rtol)
        exact = bool(torch.equal(expected.sum(dim=1).long(), counts))

        differentiable = expected[:1].detach().clone().requires_grad_(True)
        c_one = c[:1]; z_one = z[:1] if is_cvae else None
        all_logits = model.decode_logits(z_one, c_one, differentiable)
        selected = sorted(set((0, n // 2, n - 1)))
        grad_max = []
        for index, i in enumerate(selected):
            grad = torch.autograd.grad(all_logits[:, i].sum(), differentiable,
                                       retain_graph=index < len(selected)-1)[0]
            future = grad[:, i:]
            grad_max.append(float(future.abs().max().item()) if bool(torch.isfinite(future).all()) else float("nan"))
        grad_ok = all(math.isfinite(value) and value == 0. for value in grad_max)
        cfg = {
            "model_format_version": 5, "kind": kind, "n_sites": n,
            "condition_dim": condition_dim, "latent_dim": latent_dim if is_cvae else 0,
            "hidden_dim": hidden_dim, "encoder_kind": "graph" if is_cvae else "dense",
            "graph_context": True, "graph": graph.to_dict(), "decoder_context_spec": context.to_spec(),
            "remaining_count_features": False, "energy_target": "synthetic_no_label",
            "label_energy_protocol": "No energy labels; untrained synthetic check",
            "proposal_geometry_protocol": "Synthetic binary occupancy on supplied parent",
            "beta_schedule": {"scope": "untrained synthetic check"},
        }
        path = Path(directory) / f"{context.to_spec()['version']}_{kind}.pt"
        torch.save({"configuration": cfg, "state_dict": model.state_dict()}, path)
        restored, restored_cfg = load_model(path, str(device))
        with torch.no_grad():
            actual = restored.sample(c, counts, z=fixed_z, uniforms=uniforms)
            actual_logits = restored.decode_logits(fixed_z, c, expected)
            if is_cvae:
                mean, logvar = model.encode(expected, c)
                restored_mean, restored_logvar = restored.encode(expected, c)
                encoder_mean = comparison(mean, restored_mean, atol, rtol)
                encoder_logvar = comparison(logvar, restored_logvar, atol, rtol)
            else:
                encoder_mean = encoder_logvar = None
        reload_logits = comparison(teacher.detach(), actual_logits, atol, rtol)
        equal_draws = bool(torch.equal(expected, actual))
        hash_match = restored.graph_context.to_spec() == restored_cfg["decoder_context_spec"]
        rejected_corrupt = {}
        if not is_cvae:
            version = context.to_spec()["version"]
            angular_key = ("graph_context.vector_geometry" if version == "causal-directional-v1"
                           else "graph_context.pair_weights")
            for key in ("graph_context.radial.kernels", angular_key):
                state = dict(model.state_dict())
                state[key] = state[key].clone()
                state[key].reshape(-1)[0] += .1
                corrupt = Path(directory) / f"corrupt_{version}_{key.rsplit('.', 1)[-1]}.pt"
                torch.save({"configuration": cfg, "state_dict": state}, corrupt)
                try:
                    load_model(corrupt, str(device))
                except ValueError as exc:
                    if "buffers disagree" not in str(exc):
                        raise
                    rejected_corrupt[key] = True
                else:
                    rejected_corrupt[key] = False
        result[kind] = {
            "pass": (exact and logits_agree["pass"] and grad_ok and equal_draws and reload_logits["pass"]
                     and hash_match and all(rejected_corrupt.values())
                     and (not is_cvae or (encoder_mean["pass"] and encoder_logvar["pass"]))),
            "exact_counts": exact, "counts": counts.tolist(),
            "teacher_vs_step_logits": logits_agree,
            "decoder_future_gradients_exact_zero": grad_ok,
            "decoder_gradient_positions": selected,
            "decoder_future_gradient_max": max(grad_max) if grad_ok else None,
            "integrated_v5_loader_completed": True, "reload_identical_draws": equal_draws,
            "reload_logits": reload_logits, "reconstructed_context_spec_matches": hash_match,
            "reload_encoder_mean": encoder_mean, "reload_encoder_logvar": encoder_logvar,
            "tampered_static_buffers_rejected": rejected_corrupt,
            "parameters": sum(p.numel() for p in model.parameters()),
        }
        del model, restored
    return result


def synchronized_time(call, device, warmup, repeats):
    """Feature-only inference timing. CPU support is for tiny unit fixtures."""
    if warmup < 0 or repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    is_cuda = torch.device(device).type == "cuda"
    sync = (lambda: torch.cuda.synchronize(device)) if is_cuda else (lambda: None)
    with torch.no_grad():
        for _ in range(warmup):
            call()
        sync()
        if is_cuda:
            torch.cuda.reset_peak_memory_stats(device)
            baseline = torch.cuda.memory_allocated(device)
        durations = []
        for _ in range(repeats):
            sync(); start = time.perf_counter(); call(); sync()
            durations.append(time.perf_counter() - start)
    record = {"seconds": durations, "median_seconds": float(np.median(durations)),
              "p95_seconds": float(np.quantile(durations, .95)), "warmup": warmup, "repeats": repeats}
    if is_cuda:
        peak = torch.cuda.max_memory_allocated(device)
        record.update(baseline_allocated_bytes=baseline, peak_allocated_bytes=peak,
                      incremental_peak_allocated_bytes=max(0, peak-baseline),
                      peak_reserved_bytes=torch.cuda.max_memory_reserved(device))
    else:
        record.update(baseline_allocated_bytes=None, peak_allocated_bytes=None,
                      incremental_peak_allocated_bytes=None, peak_reserved_bytes=None)
    return record


def profile_context(context, n, device, batch_size, warmup, repeats, seed):
    context = context.to(device).eval()
    rng = torch.Generator(device=device).manual_seed(seed)
    x = torch.randint(0, 2, (batch_size, n), generator=rng, device=device).float()
    profile = {"batch_size": batch_size,
               "scope": "FP32 no-grad context inference only; not full model, training, or MLIP",
               "teacher_all_sites": synchronized_time(lambda: context.teacher_features(x), device, warmup, repeats),
               "step_sites": {}}
    for i in sorted(set((0, n // 2, n - 1))):
        profile["step_sites"][str(i)] = synchronized_time(
            lambda i=i: context.step_features(x, i), device, warmup, repeats)
    def all_steps():
        for i in range(n):
            context.step_features(x, i)
    profile["all_sequential_context_steps"] = synchronized_time(all_steps, device, warmup, repeats)
    return profile


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--contexts", nargs="+", choices=tuple(CONTEXTS), default=list(CONTEXTS))
    parser.add_argument("--cutoff-A", type=float, default=5.)
    parser.add_argument("--shell-tolerance-A", type=float, default=1e-5)
    parser.add_argument("--atol", type=float, default=2e-6)
    parser.add_argument("--rtol", type=float, default=2e-6)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--patterns", type=int, default=2)
    parser.add_argument("--permutation-batch", type=int, default=32)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261110)
    parser.add_argument("--scratch", type=Path, help="Existing directory for temporary untrained checkpoints")
    args = parser.parse_args(argv)
    if (not math.isfinite(args.atol) or args.atol <= 0 or not math.isfinite(args.rtol) or args.rtol < 0
            or args.batch_size < 1 or args.patterns < 1 or args.permutation_batch < 1
            or args.warmup < 0 or args.repeats < 1):
        parser.error("finite positive atol/batch/patterns/permutation-batch/repeats and nonnegative rtol/warmup required")
    if len(args.contexts) != len(set(args.contexts)):
        parser.error("Each requested context may occur only once")
    if args.output.exists():
        parser.error("Output already exists; preserving existing evidence")
    if args.scratch is not None and not args.scratch.is_dir():
        parser.error("scratch must be an existing directory")
    require_slurm_gpu()
    torch.set_num_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda:0")
    parent = load_parent(args.parent)
    graph = build_periodic_graph(parent, args.cutoff_A)
    transforms = transformed_graphs(graph)
    properties = torch.cuda.get_device_properties(device)
    sources = {
        "validation_script": file_hash(__file__),
        "graph_module": file_hash(inspect.getsourcefile(build_periodic_graph)),
        "parent_module": file_hash(inspect.getsourcefile(ParentSpec)),
        "model_module": file_hash(inspect.getsourcefile(ARCVAE)),
        "checkpoint_loader": file_hash(inspect.getsourcefile(load_model)),
    }
    receipt = {
        "status": "RUNNING", "parent_path": str(args.parent), "parent_source_sha256": file_hash(args.parent),
        "parent_hash": parent.parent_hash, "graph_hash": graph.to_dict()["graph_hash"],
        "n_sites": graph.n_sites, "retained_permutations": len(parent.permutations),
        "parent_symmetry_audit": parent.symmetry_audit, "source_sha256": sources,
        "settings": {"feature_cutoff_A": args.cutoff_A, "shell_tolerance_A": args.shell_tolerance_A,
                     "atol": args.atol, "rtol": args.rtol, "gradient_atol": 0.,
                     "tolerance_scope": "dimensionless FP32 local-feature numerical checks, not physical equivalence",
                     "tolerance_provenance": "2e-6 default builds on existing 1e-6/2e-6 FP32 teacher/step tests; observed maxima are retained",
                     "batch_size": args.batch_size, "patterns": args.patterns,
                     "permutation_batch": args.permutation_batch, "warmup": args.warmup,
                     "repeats": args.repeats, "seed": args.seed, "dtype": "float32", "tf32": False},
        "runtime": {"host": socket.gethostname(), "python_executable": sys.executable,
                    "torch": torch.__version__, "torch_cuda": torch.version.cuda,
                    "numpy": np.__version__, "ase": importlib.metadata.version("ase"),
                    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                    "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
                    "slurm_step_id": os.environ.get("SLURM_STEP_ID"),
                    "slurmd_nodename": os.environ.get("SLURMD_NODENAME"),
                    "gpu_name": properties.name, "gpu_total_memory_bytes": properties.total_memory},
        "no_label_reads": True, "no_model_training": True, "no_MLIP_or_DFT_calls": True,
        "scope": "Local descriptor invariance only; fixed-order sampler equivariance is not asserted",
        "contexts": {},
    }
    with tempfile.TemporaryDirectory(prefix="angular-context-check-", dir=args.scratch) as temporary:
        for index, name in enumerate(args.contexts):
            try:
                torch.manual_seed(args.seed + index)
                context = make_context(name, graph, args.shell_tolerance_A).to(device)
                sources[name] = file_hash(inspect.getsourcefile(context.__class__))
                factory = lambda other, name=name: make_context(name, other, args.shell_tolerance_A)
                validation = validate_context(context, graph, transforms, factory, device,
                    atol=args.atol, rtol=args.rtol, seed=args.seed,
                    patterns=args.patterns, permutation_batch=args.permutation_batch)
                checkpoint = checkpoint_check(context, graph, device, temporary, args.atol, args.rtol, args.seed)
                performance = profile_context(context, graph.n_sites, device, args.batch_size,
                                               args.warmup, args.repeats, args.seed)
                ok = validation["status"] == "PASS" and all(value["pass"] for value in checkpoint.values())
                receipt["contexts"][name] = {"status": "PASS" if ok else "FAIL",
                    "spec": context.to_spec(), "validation": validation,
                    "checkpoint_and_count_checks": checkpoint, "inference_profile": performance}
            except Exception as exc:
                receipt["contexts"][name] = {"status": "FAIL", "error": str(exc),
                    "error_type": type(exc).__name__, "traceback": traceback.format_exc()}
    receipt["status"] = "PASS" if all(value["status"] == "PASS" for value in receipt["contexts"].values()) else "FAIL"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(receipt, stream, indent=2, allow_nan=False); stream.write("\n")
    print(json.dumps({"status": receipt["status"], "output": str(args.output),
                      "contexts": {key: value["status"] for key, value in receipt["contexts"].items()}}, indent=2))
    return 0 if receipt["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
