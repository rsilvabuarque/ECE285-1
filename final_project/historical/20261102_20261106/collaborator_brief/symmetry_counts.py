"""Count vacancy patterns modulo species-preserving periodic parent symmetry.

This is bounded crystallographic bookkeeping on three archived parent frames,
not relaxation, MD, or model inference. Burnside's lemma counts fixed masks by
permutation cycle lengths. The Hu parent requires a declared ideal-parent
symmetry identification tolerance; it is not an energetic acceptance criterion.
"""
import hashlib
import json
import math
from collections import Counter
from itertools import combinations
from pathlib import Path

import numpy as np
import spglib
from ase import Atoms
from ase.data import atomic_numbers
from ase.io import read

BASE = Path(__file__).resolve().parent
ROOT = Path('/global/cfs/cdirs/m5056/rsb/2026/cathode_relithiation/cathode_relithiation_project')
HU = ROOT / 'for_chat_gpt/LiCoO2_All_data/Data_Set/1_O3/1_Opt_Structures/Li12Co12O24'
HIST = ROOT / 'results/publication/default_systems/electrode/LCO_mp-22526/structure_generation_tiny/md_runs/engine_uma/T_393K/lith_100.00pct/base_00000000.extxyz'
FUTURE = ROOT / 'results/publication/default_systems/deepmd_collaborator_electrode/full_study_4x6x2/structure_selection/pristine_4x6x2_POSCAR'


def analyze(name, atoms, tolerance, probes):
    lattice = atoms.cell.array
    frac = atoms.get_scaled_positions()
    species = atoms.numbers
    sites = frac[species == 3]
    n = len(sites)
    cell = (lattice, frac, species)
    sensitivity = []
    for eps in probes:
        d = spglib.get_symmetry_dataset(cell, symprec=eps)
        sensitivity.append(dict(symprec_A=eps, spacegroup=d.international,
                                n_operations=len(d.rotations)))
    d = spglib.get_symmetry_dataset(cell, symprec=tolerance)
    permutations = []
    max_target = max_species = 0.0
    for rotation, translation in zip(d.rotations, d.translations):
        def match(points):
            delta = (points @ rotation.T + translation)[:, None, :] - points[None, :, :]
            delta -= np.round(delta)
            distance = np.linalg.norm(delta @ lattice, axis=-1)
            p = distance.argmin(axis=1)
            assert len(set(p)) == len(points)
            return tuple(map(int, p)), float(distance[np.arange(len(p)), p].max())
        p, residual = match(sites)
        max_target = max(max_target, residual)
        permutations.append(p)
        for element in set(species):
            _, residual = match(frac[species == element])
            max_species = max(max_species, residual)
    permutations = sorted(set(permutations))
    pa = np.array(permutations, dtype='<i2')
    group = {p.tobytes() for p in pa}
    assert np.arange(n, dtype='<i2').tobytes() in group
    for p in pa:
        assert all(q.tobytes() in group for q in p[pa])

    max_k = n // 2
    total = [0] * (max_k + 1)
    cycles = Counter()
    for p in permutations:
        seen, lengths = set(), []
        for start in range(n):
            if start in seen:
                continue
            current, length = start, 0
            while current not in seen:
                seen.add(current)
                length += 1
                current = p[current]
            lengths.append(length)
        cycles[','.join(map(str, sorted(lengths)))] += 1
        coefficient = [1] + [0] * max_k
        for length in lengths:
            for k in range(max_k, length - 1, -1):
                coefficient[k] += coefficient[k - length]
        total = [a + b for a, b in zip(total, coefficient)]
    assert all(value % len(permutations) == 0 for value in total)
    counts = [total[min(k, n-k)] // len(permutations) for k in range(n+1)]
    # Independent small-orbit enumerations check the cycle-polynomial result.
    pairs = {min(tuple(sorted((p[i], p[j]))) for p in permutations)
             for i, j in combinations(range(n), 2)}
    assert len(pairs) == counts[2]
    assert len({min(p[i] for p in permutations) for i in range(n)}) == counts[1]
    assert counts[0] == counts[n] == 1
    checks = {'identity_closure': 'PASS', 'k0_k1_k2': 'PASS'}
    if n == 12:
        states = np.arange(1 << n, dtype=np.uint32)
        canonical = states.copy()
        for p in permutations:
            transformed = np.zeros_like(states)
            for i, j in enumerate(p):
                transformed |= ((states >> i) & 1) << j
            canonical = np.minimum(canonical, transformed)
        pop = np.array([int(i).bit_count() for i in states])
        enumerated = [len(set(canonical[pop == k])) for k in range(n+1)]
        assert enumerated == counts
        checks['all4096_Hu_masks'] = 'PASS'
    digest = hashlib.sha256()
    for value, dtype in [(species, '<i4'), (lattice, '<f8'), (atoms.positions, '<f8')]:
        digest.update(np.asarray(value, dtype=dtype).tobytes())
    return dict(name=name, n_sites=n, n_atoms=len(atoms), symprec_A=tolerance,
                spacegroup=d.international, sensitivity=sensitivity,
                n_full_operations=len(d.rotations), n_site_permutations=len(permutations),
                max_Li_mapping_residual_A=max_target, max_species_mapping_residual_A=max_species,
                geometry_sha256=digest.hexdigest(), independent_checks=checks,
                cycle_histogram=dict(cycles), orbit_counts_by_k=counts,
                raw_counts_by_k=[math.comb(n,k) for k in range(n+1)])


def main():
    types = np.loadtxt(HU / 'type.raw', dtype=int)
    names = (HU / 'type_map.raw').read_text().split()
    hu = Atoms(numbers=[atomic_numbers[names[t]] for t in types],
               positions=np.load(HU / 'set.000/coord.npy')[0].reshape(-1,3).astype(float),
               cell=np.load(HU / 'set.000/box.npy')[0].reshape(3,3).astype(float), pbc=True)
    cases = [analyze('Hu_O3_12', hu, .001, [1e-5,1e-4,.001,.01,.05,.1]),
             analyze('Historical_O3_24', read(HIST,index=0,format='extxyz'), 1e-5,
                     [1e-6,1e-5,1e-4,.001,.01]),
             analyze('Future_O3_144_excluded_from_training', read(FUTURE,format='vasp'), 1e-5,
                     [1e-6,1e-5,.001,.01])]
    inputs = [HIST,FUTURE,HU/'type.raw',HU/'type_map.raw',HU/'set.000/box.npy',HU/'set.000/coord.npy']
    receipt = dict(spglib_version=spglib.__version__, frame_indices={'Hu':0,'Historical':0},
                   method='Burnside: mean_g coefficient[t^k] product_cycles(1+t^length).',
                   tolerance_provenance='Parent symmetry identification; Hu ideal R-3m inferred from stored nearly symmetric geometry. Not a physical acceptance tolerance or symmetry of every thermal snapshot.',
                   input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in inputs},
                   cases=cases)
    (BASE/'assets/symmetry_counts.json').write_text(json.dumps(receipt,indent=2)+'\n')
    rows = []
    for case in cases[:2]:
        m = case['n_sites']
        for numerator in [3,5,11]:
            # Requested common fractions 3/4, 5/6, and 11/12.
            denominator = {3:4,5:6,11:12}[numerator]
            k = m - m*numerator//denominator
            rows.append(f"{m} & ${numerator}/{denominator}$ & {k} & {case['orbit_counts_by_k'][k]:,} " + chr(92)*2)
    header = (r'\begin{tabular}{rrrr}' + '\n' + r'\toprule' + '\n'
              + r'Parent Li sites $M$ & Li retained $x$ & Vacancies $k$ & Distinct patterns\\'
              + '\n' + r'\midrule' + '\n')
    footer = '\n' + r'\bottomrule' + '\n' + r'\end{tabular}' + '\n'
    (BASE/'assets/symmetry_table.tex').write_text(header+'\n'.join(rows)+footer)
    future = cases[2]['orbit_counts_by_k']
    macros = {'FutureSeven':f'{future[7]:,}', 'FutureThirtyFive':f'{future[35]:.3e}',
              'FutureSeventyTwo':f'{future[72]:.3e}'}
    (BASE/'assets/symmetry_facts.tex').write_text(''.join('\\newcommand{\\'+k+'}{'+v+'}\n' for k,v in macros.items()))
    for case in cases:
        print(case['name'], case['n_site_permutations'], {k:case['orbit_counts_by_k'][k] for k in ([1,2,3,6] if case['n_sites']<100 else [7,35,72])})


if __name__ == '__main__':
    main()
