"""Prepare a new idealized N12 O3 lattice and exhaustive vacancy-orbit pilot."""
import argparse
import json
from math import comb
from pathlib import Path
import numpy as np
from ase import Atoms
from ase.io import write
from ase.geometry import find_mic
from scipy.optimize import linear_sum_assignment
import spglib
from cathode_cvae.data import prepare_parent, sha256
from cathode_cvae.entropy import enumerate_orbits
from cathode_cvae.parent_spec import ParentSpec


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source-parent', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--symprec-A', type=float, default=1e-5,
                   help='Provisional coordinate matching audit in Å, not energy tolerance')
    a = p.parse_args()
    if a.output_dir.exists(): raise FileExistsError(a.output_dir)
    with np.load(a.source_parent, allow_pickle=False) as z:
        cell, positions, numbers = z['cell_A'], z['positions_A'], z['numbers']
    frac = np.linalg.solve(cell.T, positions.T).T % 1.
    dataset = spglib.get_symmetry_dataset((cell, frac, numbers), symprec=a.symprec_A)
    if dataset.number != 166: raise ValueError('Source is not audited R-3m O3 parent')
    raw_std = spglib.standardize_cell((cell, frac, numbers), to_primitive=False,
                                     no_idealize=True, symprec=a.symprec_A)
    ideal = spglib.standardize_cell((cell, frac, numbers), to_primitive=False,
                                   no_idealize=False, symprec=a.symprec_A)
    c0, f0, z0 = raw_std; c1, f1, z1 = ideal
    if (z1 == 3).sum() != 3 or len(z1) != 12:
        raise ValueError('Expected conventional Li3Co3O6 cell')
    # Match fractional sites in the ideal conventional metric. This measures
    # internal site adjustment separately from cell standardization/deformation.
    residuals = []
    for species in np.unique(z1):
        i, j = np.flatnonzero(z0 == species), np.flatnonzero(z1 == species)
        delta = (f0[i, None] - f1[None, j]) @ c1
        _, distances = find_mic(delta.reshape(-1, 3), c1, pbc=True)
        distances = distances.reshape(len(i), len(j)); rows, cols = linear_sum_assignment(distances)
        residuals.extend(distances[rows, cols].tolist())
    parent = Atoms(numbers=z1, scaled_positions=f1, cell=c1, pbc=True).repeat((2, 2, 1))
    a.output_dir.mkdir(parents=True)
    structure = a.output_dir / 'ideal_O3_N12.extxyz'
    write(structure, parent, format='extxyz')
    parent_path = a.output_dir / 'parent.npz'
    parent_receipt = prepare_parent(structure, 'Li', parent_path, a.symprec_A)
    with np.load(parent_path, allow_pickle=False) as z: arrays = {k: z[k] for k in z.files}
    spec = ParentSpec.from_legacy(arrays, source_sha256=sha256(parent_path), symprec_A=a.symprec_A)
    spec.write(a.output_dir / 'parent_spec.json')
    catalog = enumerate_orbits(arrays['permutations'])
    ids = np.array([f'mask12_{int(c):03x}'
                    for c in catalog['canonical_vacancy_codes']])
    np.savez_compressed(a.output_dir / 'orbit_representatives.npz',
        **{k: v for k, v in catalog.items() if k != 'counts'}, structure_ids=ids,
        parent_sha256=np.array(sha256(parent_path)))
    # Numerical invariance audit: independent symmetry-related orientations,
    # not new physical orbits and not pooled as new independent observations.
    rng = np.random.default_rng(2026111203); rows=[]; origins=[]; op_index=[]
    permutations = arrays['permutations']
    for n in (3, 6, 9):
        possible = np.flatnonzero((catalog['n_target'] == n) & (catalog['multiplicity'] > 1))
        for idx in rng.choice(possible, size=min(4, len(possible)), replace=False):
            o = catalog['occupancy'][idx]; seen={o.tobytes()}
            for operation in rng.permutation(len(permutations)):
                changed = o[np.argsort(permutations[operation])]
                if changed.tobytes() in seen: continue
                seen.add(changed.tobytes()); rows.append(changed);origins.append(ids[idx]);op_index.append(operation)
                if len(seen) == 4: break
    orientation = np.asarray(rows, np.uint8)
    np.savez_compressed(a.output_dir / 'symmetry_orientation_checks.npz',
        occupancy=orientation, masks=1-orientation,
        structure_ids=np.array([f'orientation12_{i:04d}' for i in range(len(rows))]),
        original_structure_ids=np.asarray(origins), operation_indices=np.asarray(op_index),
        n_target=orientation.sum(1), parent_sha256=np.array(sha256(parent_path)))
    receipt = dict(status='PREPARED_EXACT_LATTICE_CATALOG', source_parent=str(a.source_parent),
        source_parent_sha256=sha256(a.source_parent), source_spacegroup=166,
        source_symprec_A=a.symprec_A, spglib_version=spglib.__version__,
        source_to_standard_transformation_matrix=dataset.transformation_matrix.tolist(),
        source_to_standard_origin_shift=dataset.origin_shift.tolist(),
        conventional_cell_without_idealization_A=np.asarray(c0).tolist(),
        conventional_cell_idealized_A=np.asarray(c1).tolist(),
        conventional_cell_difference_frobenius_A=float(np.linalg.norm(c1-c0)),
        standardized_site_adjustment_max_A=float(max(residuals)),
        standardized_site_adjustment_rms_A=float(np.sqrt(np.mean(np.square(residuals)))),
        adjustment_scope='Matched fractional sites in ideal conventional metric; cell change reported separately, not a full original-supercell displacement',
        repeat_matrix=[[2,0,0],[0,2,0],[0,0,1]], parent=parent_receipt,
        parent_audit=spec.symmetry_audit, catalogue_counts=catalog['counts'],
        labelled_masks=2**12, orbit_representatives=len(ids), symmetry_orientation_checks=len(rows),
        source_labels_used=False,
        scope='New idealized O3 fixed-lattice model; no inherited Hu/DFT labels; no phase, spin, relaxation or infinite-size entropy claim')
    for n in range(13):
        assert sum(catalog['multiplicity'][catalog['n_target']==n]) == comb(12,n)
    receipt['outputs']={p.name:sha256(p) for p in a.output_dir.iterdir() if p.is_file()}
    (a.output_dir/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({'status':receipt['status'],'orbits':len(ids),'parent':str(parent_path),
                      'scoring_input':str(a.output_dir/'orbit_representatives.npz')},indent=2))


if __name__ == '__main__': main()
