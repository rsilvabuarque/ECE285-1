"""Inspect the physical parent graph and a training-fitted vacancy descriptor.

Run on an allocation for a full archive. This only reads existing scalar labels;
it performs no potential inference and never changes protected role assignments.
"""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import pearsonr, spearmanr

from cathode_cvae.conditions import classify as energy_classify, load_schema as load_energy_schema
from cathode_cvae.data import sha256
from cathode_cvae.descriptor import (classify as descriptor_classify, fit_descriptor_schema,
                                    save_schema)
from cathode_cvae.graph import build_periodic_graph, descriptor_symmetry_audit, vacancy_pair_fraction
from cathode_cvae.parent_spec import ParentSpec
from cathode_cvae.runtime import require_compute_allocation


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--parent", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--energy-schema", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--cutoff-A", type=float, default=5.)
    args = p.parse_args(argv)
    require_compute_allocation()
    args.output_dir.mkdir(parents=True, exist_ok=False)
    with np.load(args.parent, allow_pickle=False) as z:
        source = {k: z[k] for k in z.files if k in {"numbers", "positions_A", "cell_A", "target_indices", "li_indices", "permutations", "pbc", "site_groups"}}
    spec = ParentSpec.from_legacy(source, sha256(args.parent))
    graph = build_periodic_graph(spec, args.cutoff_A)
    spec.write(args.output_dir / "parent_spec.json")
    (args.output_dir / "periodic_graph.json").write_text(json.dumps(graph.to_dict(), indent=2) + "\n")
    shell = graph.shell_audit()
    invariant = descriptor_symmetry_audit(graph)
    with np.load(args.data, allow_pickle=False) as z:
        data = {k: z[k] for k in z.files}
    x = data["occupancy"]
    n = x.sum(axis=1).astype(int)
    split = data["split"].astype(str)
    # Existing learning arrays store total energy plus atom count, or eV/atom.
    if "energy_eV_atom" in data:
        energy = data["energy_eV_atom"].astype(float)
    elif "e_per_atom" in data:
        energy = data["e_per_atom"].astype(float)
    elif "energy_eV" in data and "n_atoms" in data:
        energy = data["energy_eV"] / data["n_atoms"]
    else:
        raise ValueError(f"archive has no recognized energy fields: {sorted(data)}")
    descriptor = vacancy_pair_fraction(x, graph)
    energy_schema = load_energy_schema(args.energy_schema)
    bins = energy_classify(energy_schema, n, energy)
    schema = fit_descriptor_schema(descriptor, n, bins, split, parent_hash=spec.parent_hash,
                                   graph_hash=graph.to_dict()["graph_hash"])
    save_schema(schema, args.output_dir / "descriptor_schema.json")
    ids = data.get("structure_ids", data.get("canonical_keys", data.get("keys")))
    if ids is None:
        raise ValueError("stable structure IDs are needed for aligned descriptor values")
    np.savez_compressed(args.output_dir / "descriptor_values.npz", structure_ids=ids,
                        descriptor_values=descriptor, n_target=n, split=split, energy_bins=bins,
                        parent_hash=np.asarray(spec.parent_hash), graph_hash=np.asarray(graph.to_dict()["graph_hash"]),
                        parent_sha256=np.asarray(sha256(args.parent)))
    records = []
    for count in sorted(np.unique(n)):
        for role in ("train", "val", "test"):
            for label in ("all", "low", "middle", "high"):
                chosen = (n == count) & (split == role)
                if label != "all":
                    chosen &= bins == label
                d, e = descriptor[chosen], energy[chosen]
                varying = len(d) > 2 and np.ptp(d) > 0 and np.ptp(e) > 0
                records.append(dict(n_target=int(count), split=role, energy_bin=label, rows=len(d),
                    descriptor_min=float(d.min()) if len(d) else None,
                    descriptor_max=float(d.max()) if len(d) else None,
                    descriptor_unique_values=len(np.unique(d)),
                    descriptor_std=float(d.std()) if len(d) else None,
                    pearson_r=float(pearsonr(d, e).statistic) if varying else None,
                    spearman_rho=float(spearmanr(d, e).statistic) if varying else None))
    with (args.output_dir / "descriptor_energy.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0])); writer.writeheader(); writer.writerows(records)
    counts = sorted(np.unique(n))
    fig, axes = plt.subplots(1, len(counts), figsize=(4.1 * len(counts), 3.7), squeeze=False)
    for ax, count in zip(axes[0], counts):
        chosen = (n == count) & (split == "train")
        reference = float(np.median(energy[chosen]))
        ax.hexbin(descriptor[chosen], (energy[chosen] - reference) * 1000, gridsize=25, mincnt=1, bins="log")
        ax.axvline(schema["compositions"][str(int(count))]["boundary"], color="red", lw=1, ls="--", label="training descriptor split")
        ax.set(title=f"{count}/{spec.n_sites} ions; training only", xlabel="Nearest-shell vacancy-pair fraction")
        ax.legend(fontsize=7)
    axes[0, 0].set_ylabel("Teacher energy − training median (meV/atom)")
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(args.output_dir / f"descriptor_vs_energy.{ext}", dpi=180)
    plt.close(fig)
    # For this parent diagnostic only: exact Cartesian z planes are a descriptive
    # grouping, not a generic crystallographic sublattice assignment.
    z = spec.positions_A[spec.mobile_indices, 2]
    plane_values, plane_ids = np.unique(np.round(z, decimals=5), return_inverse=True)
    a, b, _, distance = graph.mobile_edges()
    shell["cartesian_z_plane_diagnostic"] = dict(planes=len(plane_values), plane_values_A=plane_values.tolist(),
        rounding_decimals=5, cross_plane_directed_edges=int(np.sum(plane_ids[a] != plane_ids[b])),
        warning="Diagnostic for this orientation only; no automatic layer grouping is put in ParentSpec")
    result = dict(status="PASS", data_rows=len(x), protected_split_counts={r: int(np.sum(split == r)) for r in np.unique(split)},
                  parent_source_sha256=sha256(args.parent), data_source_sha256=sha256(args.data),
                  energy_schema_identity=energy_schema["identity"], descriptor_schema_identity=schema["identity"],
                  parent_hash=spec.parent_hash, graph_hash=graph.to_dict()["graph_hash"],
                  parent_symmetry=spec.symmetry_audit, graph_audit=shell, descriptor_invariance=invariant,
                  condition_pilot_support="Full joint counts are saved; decide pilot from training support and within-bin variation, not test correlations",
                  teacher_inference_calls=0)
    (args.output_dir / "receipt.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()

