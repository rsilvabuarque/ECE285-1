"""Inspect frozen data, prepare a parent, and export vacancy proposals."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            h.update(block)
    return h.hexdigest()


def target_indices(parent):
    """New parents use target_indices; li_indices is the historical alias."""
    return np.asarray(parent["target_indices"] if "target_indices" in parent else parent["li_indices"], dtype=int)


def validate_permutation_group(permutations):
    """Verify a finite action, generating closure without a quadratic array.

    A filtered set of approximate coordinate matches need not be a group. Orbit
    equivalence and canonicalization require identity and closure, not just a
    collection of individually valid site permutations.
    """
    from collections import deque
    p = np.asarray(permutations, dtype=int)
    if p.ndim != 2 or not len(p) or not np.all(np.sort(p, axis=1) == np.arange(p.shape[1])):
        raise ValueError("Invalid permutation array")
    allowed = {tuple(row) for row in p}
    identity = tuple(range(p.shape[1]))
    if identity not in allowed:
        raise ValueError("Site operations do not include identity")
    generated, generators = {identity}, []
    for row in p:
        if tuple(row) in generated:
            continue
        generators.append(row)
        queue = deque(generated)
        while queue:
            current = np.asarray(queue.popleft())
            for generator in generators:
                product = tuple(current[generator])
                if product not in allowed:
                    raise ValueError("Retained site operations are not closed; review the pristine structure and symmetry tolerance before defining orbits")
                if product not in generated:
                    generated.add(product)
                    queue.append(product)
    return True


def canonical_mask(occupancy, permutations):
    """Lexicographically minimum vacancy bit string, with its gather witness.

    Parent permutations map source site to destination site. Inverse gathers
    therefore transform an occupancy into the symmetry-related orientation.
    This works beyond 63 sites without fixed-width integer overflow.
    """
    o = np.asarray(occupancy)
    if o.ndim != 1 or not np.isin(o, [0, 1]).all():
        raise ValueError("occupancy must be one binary vector")
    perms = np.asarray(permutations, dtype=int)
    if perms.ndim != 2 or perms.shape[1] != len(o):
        raise ValueError("permutations and occupancy have different site counts")
    if not np.all(np.sort(perms, axis=1) == np.arange(len(o))):
        raise ValueError("invalid site permutations")
    inv = np.argsort(perms, axis=1)
    keys = ["".join(map(str, (1 - o[p]).astype(int))) for p in inv]
    witness = min(range(len(keys)), key=keys.__getitem__)
    return keys[witness], inv[witness]


def deletion_atoms(parent, occupancy):
    from ase import Atoms
    idx = target_indices(parent)
    o = np.asarray(occupancy)
    if o.shape != (len(idx),) or not np.isin(o, [0, 1]).all():
        raise ValueError("one binary occupancy per removable parent site is required")
    keep = np.ones(len(parent["numbers"]), dtype=bool)
    keep[idx[o == 0]] = False
    return Atoms(numbers=parent["numbers"][keep], positions=parent["positions_A"][keep],
                 cell=parent["cell_A"], pbc=True), np.flatnonzero(keep)


def prepare_parent(structure, target, output, symprec):
    """Retain only operations that bijectively match every chemical species.

    symprec is a numerical matching tolerance in Angstrom, supplied explicitly
    by the caller. It is not a physical criterion for energetic equivalence.
    """
    from ase.io import read
    from ase.geometry import find_mic
    from scipy.optimize import linear_sum_assignment
    import spglib

    if symprec <= 0:
        raise ValueError("--symprec must be positive, in Angstrom")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    atoms = read(structure)
    if not np.all(atoms.pbc) or abs(atoms.cell.volume) < 1e-10:
        raise ValueError("A three-dimensional periodic parent with nonzero volume is required")
    numbers = atoms.numbers
    idx = np.flatnonzero(np.array(atoms.get_chemical_symbols()) == target)
    if not len(idx):
        raise ValueError(f"No {target} sites in the pristine parent")
    frac = atoms.get_scaled_positions(wrap=True)
    symmetry = spglib.get_symmetry((atoms.cell.array, frac, numbers), symprec=symprec)
    if symmetry is None:
        raise ValueError("spglib found no symmetry operations")
    full, rotations, translations, errors = [], [], [], []
    for rot, trans in zip(symmetry["rotations"], symmetry["translations"]):
        mapped = (frac @ rot.T + trans) @ atoms.cell.array
        perm = np.empty(len(atoms), dtype=int)
        maximum = 0.0
        for species in np.unique(numbers):
            rows = np.flatnonzero(numbers == species)
            delta = mapped[rows, None, :] - atoms.positions[None, rows, :]
            _, distances = find_mic(delta.reshape(-1, 3), atoms.cell, pbc=True)
            distances = distances.reshape(len(rows), len(rows))
            a, b = linear_sum_assignment(distances)
            maximum = max(maximum, float(distances[a, b].max()))
            perm[rows[a]] = rows[b]
        if maximum <= symprec:
            full.append(perm)
            rotations.append(rot)
            translations.append(trans)
            errors.append(maximum)
    if not full:
        raise ValueError("No chemically valid operations met the supplied tolerance")
    local = {int(v): i for i, v in enumerate(idx)}
    target_perms = np.array([[local[int(p[i])] for i in idx] for p in full])
    # Different space-group operations can induce the same target permutation.
    unique, operation_indices = np.unique(target_perms, axis=0, return_index=True)
    validate_permutation_group(np.unique(np.asarray(full), axis=0))
    validate_permutation_group(unique)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("xb") as f:
        np.savez_compressed(f, numbers=numbers, positions_A=atoms.positions, cell_A=atoms.cell.array,
                            target_indices=idx, target_symbol=np.array(target), permutations=unique,
                            permutation_operation_indices=operation_indices,
                            full_permutations=np.array(full), rotations=np.array(rotations),
                            translations=np.array(translations), symprec_A=np.array(symprec))
    receipt = dict(source_sha256=sha256(structure), output_sha256=sha256(output), target=target,
                   target_sites=len(idx), atoms=len(atoms), retained_operations=len(full),
                   distinct_target_permutations=len(unique), maximum_match_residual_A=max(errors),
                   symprec_A=symprec, spglib=spglib.__version__,
                   permutation_group_verified=True,
                   scope="Operations of this periodic pristine parent; does not identify phases or electronic/magnetic equivalence")
    output.with_suffix(output.suffix + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def verify_snapshot(snapshot):
    snapshot = Path(snapshot).resolve()
    manifest = json.loads((snapshot / "SNAPSHOT_MANIFEST.json").read_text())
    failures = []
    for name, record in manifest["files"].items():
        p = (snapshot / name).resolve()
        if not p.is_relative_to(snapshot):
            raise ValueError("Unsafe manifest path")
        if not p.is_file() or p.stat().st_size != record["bytes"] or sha256(p) != record["sha256"]:
            failures.append(name)
    result = dict(status="FAIL" if failures else "PASS", files_checked=len(manifest["files"]), failures=failures)
    print(json.dumps(result, indent=2))
    if failures:
        raise RuntimeError("Snapshot checksums differ")
    return result


def prepare_learning(labels, parent_path, split_path, output, split_column="split", weighting="equal-orbit", parent_sha256=None):
    """Convert existing scalar labels to the trainer schema without resplitting.

    Rows marked diagnostic/pilot_holdout stay outside this learning export.
    Equal-composition weights use training counts only; no energy or force is
    recalculated. The source table's canonical keys are checked independently.
    """
    import csv
    from collections import Counter
    if Path(output).exists():
        raise FileExistsError(output)
    with np.load(labels, allow_pickle=False) as z:
        d = {k: z[k] for k in z.files}
    with np.load(parent_path, allow_pickle=False) as z:
        parent = {k: z[k] for k in z.files}
    occupancy = d["occupancy"]
    idx = target_indices(parent)
    n_sites = len(idx)
    if occupancy.ndim != 2 or occupancy.shape[1] != n_sites or not np.isin(occupancy, [0, 1]).all():
        raise ValueError("Label occupancy and removable parent sites differ")
    if not 1 <= n_sites <= 62:
        raise ValueError("Current learning export supports 1..62 removable sites")
    ids = d["structure_ids"].astype(str)
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate source IDs")
    records = {r["structure_id"]: r for r in csv.DictReader(Path(split_path).open())}
    if set(ids) - set(records):
        raise ValueError("Labels missing from protected split table")
    parents = {records[s]["parent"] for s in ids}
    if len(parents) != 1:
        raise ValueError("Export one parent at a time; mixed-size network training is not implemented")
    expected = d.get("parent_sha256", parent_sha256)
    if expected is None:
        # The frozen bundle supplies the producer's parent next to label arrays.
        producer_parent = Path(labels).parent / "parents" / (next(iter(parents)) + ".npz")
        if producer_parent.is_file():
            expected = sha256(producer_parent)
    if expected is None:
        raise ValueError("New label archives need parent_sha256 metadata or an explicit --parent-sha256 producer assertion")
    if str(np.asarray(expected).item()) != sha256(parent_path):
        raise ValueError("Labels and requested parent have different hashes")
    n_target = occupancy.sum(1).astype(int)
    host = len(parent["numbers"]) - n_sites
    if "n_Li" in d and not np.array_equal(d["n_Li"], n_target):
        raise ValueError("Label counts differ from occupancy")
    if not np.array_equal(d["n_atoms"], host + n_target):
        raise ValueError("Label atom counts differ from the supplied host and occupancy")
    e = d["energy_eV"] / d["n_atoms"]
    if not np.isfinite(e).all():
        raise ValueError("Nonfinite scalar energy")
    inv = np.argsort(parent["permutations"], axis=1)
    validate_permutation_group(parent["permutations"])
    powers = np.left_shift(np.uint64(1), np.arange(n_sites - 1, -1, -1, dtype=np.uint64))
    keys = []
    for start in range(0, len(occupancy), 128):
        v = (1 - occupancy[start:start + 128]).astype(np.uint64)
        codes = (v[:, inv] * powers).sum(-1).min(1)
        keys.extend(format(int(k), f"0{n_sites}b") for k in codes)
    if len(set(keys)) != len(keys):
        raise ValueError("Source labels repeat a parent symmetry orbit")
    if any(records[s]["canonical_mask"] != k for s, k in zip(ids, keys)):
        raise ValueError("Parent symmetry/site order differs from the protected split keys")
    splits = np.asarray([records[s][split_column] for s in ids])
    keep = np.isin(splits, ["train", "val", "test"])
    train_counts = Counter(n_target[keep & (splits == "train")].tolist())
    if not train_counts or set(n_target[keep]) - set(train_counts):
        raise ValueError("Every retained composition needs training references")
    weights = np.ones(len(ids))
    if weighting == "equal-composition":
        total = sum(train_counts.values())
        for n, count in train_counts.items():
            weights[n_target == n] = total / (len(train_counts) * count)
    Path(output).parent.mkdir(parents=True, exist_ok=True)
    with Path(output).open("xb") as f:
        np.savez_compressed(f, occupancy=occupancy[keep], n_Li=n_target[keep], n_target=n_target[keep],
            energy_eV_atom=e[keep], structure_ids=ids[keep], split=splits[keep], weight=weights[keep],
            canonical_masks=np.asarray(keys)[keep], parent_sha256=np.asarray(sha256(parent_path)))
    receipt = dict(label_sha256=sha256(labels), parent_sha256=sha256(parent_path), split_sha256=sha256(split_path),
                   output_sha256=sha256(output), split_column=split_column, weighting=weighting,
                   split_counts=dict(Counter(splits[keep].tolist())), excluded_split_counts=dict(Counter(splits[~keep].tolist())),
                   teacher_namespace=Path(labels).stem, training_counts=dict(train_counts),
                   scope="Existing energies only; protected splits preserved; no new physical labels")
    Path(str(output) + ".json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    return receipt


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)
    p = sub.add_parser("verify", help="Check every file against the frozen snapshot manifest")
    p.add_argument("--snapshot", type=Path, required=True)
    p = sub.add_parser("inspect", help="Print array names/shapes/types without loading pickled objects")
    p.add_argument("path", type=Path)
    p = sub.add_parser("prepare-parent", help="Make a species-aware periodic parent and site symmetry")
    p.add_argument("--structure", type=Path, required=True)
    p.add_argument("--target", default="Li", help="Element to remove; e.g. Li or Na")
    p.add_argument("--symprec", type=float, required=True, help="Explicit symmetry matching tolerance in Angstrom")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("prepare-learning", help="Join existing label arrays to their frozen orbit splits")
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--splits", type=Path, required=True)
    p.add_argument("--split-column", choices=["split", "composition_split"], default="split")
    p.add_argument("--weighting", choices=["equal-orbit", "equal-composition"], default="equal-orbit")
    p.add_argument("--parent-sha256", help="Producer assertion for a new archive lacking embedded parent identity")
    p.add_argument("--output", type=Path, required=True)
    p = sub.add_parser("export", help="Export deletion occupancy NPZ rows as POSCAR and CIF")
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--input", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    a = ap.parse_args(argv)
    if a.command == "verify":
        return verify_snapshot(a.snapshot)
    if a.command == "inspect":
        with np.load(a.path, allow_pickle=False) as z:
            print(json.dumps({k: dict(shape=z[k].shape, dtype=str(z[k].dtype)) for k in z.files}, indent=2))
    elif a.command == "prepare-parent":
        from .runtime import require_compute_allocation
        require_compute_allocation()
        return prepare_parent(a.structure, a.target, a.output, a.symprec)
    elif a.command == "prepare-learning":
        from .runtime import require_compute_allocation
        require_compute_allocation()
        return prepare_learning(a.labels, a.parent, a.splits, a.output, a.split_column, a.weighting, a.parent_sha256)
    elif a.command == "export":
        from ase.io import write
        a.output_dir.mkdir(parents=True, exist_ok=False)
        with np.load(a.parent, allow_pickle=False) as z:
            parent = {k: z[k] for k in z.files}
        with np.load(a.input, allow_pickle=False) as z:
            occupancy = z["occupancy"] if "occupancy" in z else 1 - z["masks"]
        rows = []
        for i, o in enumerate(occupancy):
            atoms, slots = deletion_atoms(parent, o)
            name = f"structure_{i:05d}"
            write(a.output_dir / (name + ".vasp"), atoms, format="vasp", direct=True, sort=False)
            write(a.output_dir / (name + ".cif"), atoms)
            rows.append(dict(name=name, parent_slot=slots.tolist(), occupancy=o.astype(int).tolist()))
        (a.output_dir / "manifest.json").write_text(json.dumps(dict(parent_sha256=sha256(a.parent), input_sha256=sha256(a.input), structures=rows), indent=2) + "\n")


if __name__ == "__main__":
    main()
