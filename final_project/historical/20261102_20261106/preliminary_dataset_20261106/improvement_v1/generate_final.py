"""Freeze equal-attempt proposals and checkpoint receipts before oracle scoring."""
import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path

import numpy as np
import torch
from models import DenseCVAE, make_model

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def seed_for(pool, run, n):
    return int.from_bytes(hashlib.sha256(f"{pool}|{run}|{n}".encode()).digest()[:8], "little") % (2**63 - 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pool", choices=["development", "final"], required=True)
    ap.add_argument("--count", type=int, default=256)
    ap.add_argument("--known-keys", type=Path, required=True)
    ap.add_argument("--run-list", type=Path, default=HERE / "data/fit_tasks.json")
    ap.add_argument("--output-prefix", type=Path, required=True)
    ap.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    a = ap.parse_args()
    assert a.count > 0
    if a.device == "cuda":
        assert os.environ.get("SLURMD_NODENAME", "").startswith("nid") and torch.cuda.is_available()
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    prefix = a.output_prefix if a.output_prefix.is_absolute() else HERE / a.output_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    paths = {name: Path(str(prefix) + suffix) for name, suffix in
             [("attempts", "_attempts.csv"), ("all", ".npz"), ("fresh", "_fresh.npz"), ("receipt", "_receipt.json")]}
    assert not any(path.exists() for path in paths.values()), "Preserve previous proposal pools"
    known_payload = json.loads(a.known_keys.read_text())
    known = set(known_payload["canonical_masks"] if isinstance(known_payload, dict) else known_payload)
    assert all(len(key) == 48 and set(key) <= {"0", "1"} for key in known)
    with np.load(ROOT / "data/parents/O3_N48_pilot.npz", allow_pickle=False) as z:
        parent = {key: z[key] for key in ["numbers", "positions_A", "cell_A", "li_indices", "permutations"]}
    inv = torch.tensor(np.argsort(parent["permutations"], axis=1), device=a.device)
    powers = 1 << torch.arange(47, -1, -1, dtype=torch.int64, device=a.device)
    attempts = []
    union = {}
    frozen = []

    def record(occupancy, method, seed, n, run, sampling):
        o = torch.as_tensor(occupancy, dtype=torch.float32, device=a.device)
        assert o.shape == (a.count, 48) and bool(((o == 0) | (o == 1)).all())
        assert bool((o.sum(1) == n).all())
        codes = []
        for start in range(0, len(o), 128):
            v = (1 - o[start:start + 128]).to(torch.int64)
            code = (v[:, inv] * powers).sum(-1).min(1).values
            codes.extend(code.cpu().tolist())
        for i, code in enumerate(codes):
            mask = f"{code:048b}"
            sid = f"mask48_{code:012x}"
            union[mask] = sid
            attempts.append(dict(method=method, seed=seed, n_Li=n, attempt=i,
                                 scoring_structure_id=sid, canonical_mask=mask,
                                 known_before=mask in known, training_run=run, sampling=sampling))

    tasks = json.loads(a.run_list.read_text())
    if isinstance(tasks, dict):
        tasks = tasks.get("runs", tasks.get("tasks"))
    assert isinstance(tasks, list)
    for task in tasks:
        name = task if isinstance(task, str) else task["name"]
        directory = HERE / "runs" / name
        config = json.loads((directory / "configuration.json").read_text())
        assert json.loads((directory / "summary.json").read_text())["status"] == "FIT_COMPLETE"
        checkpoint_path = directory / "best.pt"
        checkpoint = torch.load(checkpoint_path, map_location=a.device, weights_only=False)
        model = make_model(config["kind"], cond_dim=config["condition_dim"]).to(a.device)
        model.load_state_dict(checkpoint["state_dict"], strict=True)
        model.eval()
        method = name.rsplit("_", 1)[0] if isinstance(task, str) else task.get("variant", name.rsplit("_", 1)[0])
        seed = int(config["seed"])
        frozen.append(dict(run=name, checkpoint=str(checkpoint_path), checkpoint_sha256=sha(checkpoint_path),
                           configuration_sha256=sha(directory / "configuration.json")))
        for n in [12, 24, 36]:
            rng = torch.Generator(device=a.device).manual_seed(seed_for(a.pool, name, n))
            c = torch.tensor([[n / 48, 1.0]], device=a.device).repeat(a.count, 1)[:, :config["condition_dim"]]
            with torch.no_grad():
                o = model.sample(c, n, generator=rng)
            record(o, method, seed, n, name, "exact_count_prior")
    # Frozen pre-improvement controls receive independent proposals in each pool.
    for beta, method in [(False, "legacy_uma_cvae_topk"), (True, "legacy_uma_beta01_topk")]:
        for seed in [20261106, 20261107, 20261108]:
            name = f"uma_cvae_{'beta01_' if beta else ''}main_{seed}"
            path = ROOT / "cvae/runs" / name / "best.pt"
            checkpoint = torch.load(path, map_location=a.device, weights_only=False)
            model = DenseCVAE().to(a.device)
            model.load_state_dict(checkpoint["state_dict"], strict=True)
            model.eval()
            frozen.append(dict(run=name, checkpoint=str(path), checkpoint_sha256=sha(path)))
            for n in [12, 24, 36]:
                rng = np.random.default_rng(seed_for(a.pool, name, n))
                z = torch.tensor(rng.standard_normal((a.count, 8)), dtype=torch.float32, device=a.device)
                c = torch.tensor([[n / 48, 1.0]], device=a.device).repeat(a.count, 1)
                with torch.no_grad():
                    logits = model.decode_logits(z, c).cpu().numpy()
                order = np.lexsort((rng.random(logits.shape), -logits), axis=1)
                o = np.zeros((a.count, 48), np.uint8)
                np.put_along_axis(o, order[:, :n], 1, axis=1)
                record(o, method, seed, n, name, "top_count_MAP_random_ties")
    original_path = HERE / "data/original_low_training_seeds.json"
    original = json.loads(original_path.read_text())["uma"]
    with np.load(HERE / "training_round0.npz", allow_pickle=False) as z:
        train = z["split"].astype(str) == "train"
        adapted = {}
        adapted_thresholds = {}
        for n in [12, 24, 36]:
            ids = np.flatnonzero(train & (z["n_Li"] == n))
            e = z["energy_eV_atom"][ids]
            q = float(np.sort(e)[math.ceil(0.2 * len(e)) - 1])
            adapted[n] = z["occupancy"][ids[e <= q]].astype(np.uint8)
            adapted_thresholds[n] = q
    with np.load(HERE / "training_round1.npz", allow_pickle=False) as z:
        train_round1 = z["split"].astype(str) == "train"
        adapted_round1 = {}
        for n in [12, 24, 36]:
            ids = np.flatnonzero(train_round1 & (z["n_Li"] == n))
            values = z["energy_eV_atom"][ids]
            q = np.sort(values)[math.ceil(0.2 * len(values)) - 1]
            adapted_round1[n] = z["occupancy"][ids[values <= q]].astype(np.uint8)
    for seed in [20261106, 20261107, 20261108]:
        for n in [12, 24, 36]:
            rng = np.random.default_rng(seed_for(a.pool, f"uniform_{seed}", n))
            o = np.zeros((a.count, 48), np.uint8)
            for row in o:
                row[rng.choice(48, n, replace=False)] = 1
            record(o, "uniform", seed, n, "", "uniform_raw_fixed_count")
            for method in ["mutation_original", "mutation_adapt", "mutation_round1"]:
                rng = np.random.default_rng(seed_for(a.pool, f"{method}_{seed}", n))
                source = (1 - np.array([[int(bit) for bit in mask] for mask in original[str(n)]], np.uint8)
                          if method == "mutation_original" else adapted_round1[n] if method == "mutation_round1" else adapted[n])
                assert len(source) and (source.sum(1) == n).all()
                o = source[rng.integers(len(source), size=a.count)].copy()
                for row in o:
                    swaps = int(rng.integers(1, 4))
                    ones = rng.choice(np.flatnonzero(row), swaps, replace=False)
                    zeros = rng.choice(np.flatnonzero(1 - row), swaps, replace=False)
                    row[ones] = 0
                    row[zeros] = 1
                record(o, method, seed, n, "", "one_to_three_swaps")
    with paths["attempts"].open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(attempts[0]))
        writer.writeheader()
        writer.writerows(attempts)
    for name, keys in [("all", sorted(union)), ("fresh", sorted(key for key in union if key not in known))]:
        masks = np.array([[int(bit) for bit in key] for key in keys], dtype=np.uint8).reshape(-1, 48)
        np.savez_compressed(paths[name], structure_ids=np.array([union[key] for key in keys], dtype="U19"),
                            masks=masks, **{k: parent[k] for k in ["numbers", "positions_A", "cell_A", "li_indices"]})
    assert all(sha(item["checkpoint"]) == item["checkpoint_sha256"] for item in frozen)
    receipt = dict(status="FROZEN_BEFORE_ORACLE_SCORING", pool=a.pool, count_per_group=a.count,
                   attempt_groups=len(attempts) // a.count, attempts=len(attempts),
                   unique_orbits=len(union), fresh_orbits=sum(key not in known for key in union),
                   known_attempts=sum(row["known_before"] for row in attempts),
                   known_keys=str(a.known_keys), known_keys_sha256=sha(a.known_keys),
                   run_list_sha256=sha(a.run_list), generator_source_sha256=sha(__file__),
                   model_source_sha256=sha(HERE / "models.py"), frozen_models=frozen,
                   original_mutation_seed_sha256=sha(original_path),
                   adaptation_training_sha256=sha(HERE / "training_round0.npz"),
                   adaptation_q20_eV_atom=adapted_thresholds,
                   round1_mutation_training_sha256=sha(HERE / "training_round1.npz"),
                   outputs={key: {"path": str(path), "sha256": sha(path)} for key, path in paths.items() if key != "receipt"},
                   notes=["Independent SHA256-derived pool/run/count proposal RNGs; training seeds retained in labels.",
                          "Every duplicate/known proposal retained in denominator; all/fresh NPZ vacancy masks canonical.",
                          "No energies assigned by generator; low-tail membership requires independent oracle scores."])
    paths["receipt"].write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({key: receipt[key] for key in ["status", "pool", "attempts", "unique_orbits", "fresh_orbits"]}), flush=True)


if __name__ == "__main__":
    main()
