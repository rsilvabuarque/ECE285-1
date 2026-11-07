"""Finite-support thermodynamic-weight arithmetic, not physical LCO validation."""

import itertools
import math
from pathlib import Path

import numpy as np
import pytest

from cathode_cvae.acquisition import (
    KB_EV_K, allocate_integer, bin_probabilities, canonical_keys,
    exploration_policy, load_reference, make_allocation_plan,
    orbit_multiplicities, reference_pool_sensitivity, static_weights,
    validate_exhaustive_support,
)


def ring_group(n):
    return np.stack([np.roll(np.arange(n), k) for k in range(n)])


def test_orbit_measure_and_direct_raw_boltzmann_sum_agree():
    # Four-site periodic ring at fixed count two: adjacent orbit size4, opposite2.
    occupancy = np.array([[1, 1, 0, 0], [1, 0, 1, 0]], np.uint8)
    g = orbit_multiplicities(occupancy, ring_group(4))
    assert g.tolist() == [4, 2]
    e = np.array([0., .02])
    actual = static_weights(e, g, 433.15, measure_kind="exhaustive_orbits")
    raw_e = np.repeat(e, g)
    raw_p = np.exp(-(raw_e - raw_e.min()) / (KB_EV_K * 433.15)); raw_p /= raw_p.sum()
    assert actual["probability"][0] == pytest.approx(raw_p[:4].sum())
    assert actual["microstate_or_particle_ess"] == pytest.approx(1 / np.dot(raw_p, raw_p))
    expected_row_ess = 1 / np.dot(actual["probability"], actual["probability"])
    assert actual["representative_ess"] == pytest.approx(expected_row_ess)
    shifted = static_weights(e + 12345.678, g, 433.15, measure_kind="exhaustive_orbits")
    np.testing.assert_allclose(shifted["probability"], actual["probability"], rtol=1e-9, atol=1e-11)
    equal = static_weights([0., 0.], g, 433.15, measure_kind="exhaustive_orbits")
    assert equal["microstate_or_particle_ess"] == pytest.approx(6)
    assert equal["representative_ess"] < 2


def test_raw_sample_frequencies_are_not_orbit_degeneracies():
    # Three IID raw draws landed on one representative, one on another.
    result = static_weights([0., 0.], [3, 1], 400., measure_kind="uniform_raw_sample")
    np.testing.assert_allclose(result["probability"], [.75, .25])
    assert result["microstate_or_particle_ess"] == pytest.approx(4.)
    assert result["representative_ess"] == pytest.approx(1 / (.75**2 + .25**2))
    with pytest.raises(ValueError, match="finite total energy"):
        static_weights([0., np.nan], [3, 1], 400., measure_kind="uniform_raw_sample")
    with pytest.raises(ValueError, match="kelvin"):
        static_weights([0.], [1], 0., measure_kind="uniform_raw_sample")


def test_integer_policy_totals_and_temperature_mixture():
    p = {"low": .9999, "middle": .0001, "high": 0.}
    floor = exploration_policy(p, .1)
    assert min(floor.values()) >= .1 / 3
    assert sum(allocate_integer(floor, 17).values()) == 17
    assert allocate_integer({"low": 1/3, "middle": 1/3, "high": 1/3}, 2) == dict(low=1, middle=1, high=0)
    plan = make_allocation_plan([0., .1, .2], ["low", "middle", "high"], [1, 1, 1],
        [393.15, 493.15], ["low", "middle", "high"], 101,
        measure_kind="uniform_raw_sample", mixture_weights=[1, 3], schema_id="schema", composition=24)
    for group in [*plan["per_temperature"], *plan["temperature_mixtures"]]:
        assert sum(r["attempts"] for r in group["requests"]) == 101
        assert sum(r["probability"] for r in group["requests"]) == pytest.approx(1)
    assert plan["mixture_condition_weights"] == [.25, .75]
    for group in plan["temperature_mixtures"]:
        assert sum(r["probability"] for r in group["requests"] if r["temperature_K"] == 493.15) == pytest.approx(.75)
    with pytest.raises(ValueError, match="unsupported"):
        bin_probabilities([.5, .5], ["low", "high"], ["low"])


def test_prefix_sensitivity_retains_complete_raw_frequencies():
    result = reference_pool_sensitivity([0., .02], ["low", "high"], [3, 1],
        433.15, ["low", "high"], raw_sample_indices=[0, 1, 0, 0], sizes=[2, 4])
    assert result["rows"][-1]["raw_pool_size"] == 4
    assert result["rows"][-1]["l1_difference_to_full_pool"] == pytest.approx(0)
    assert result["rows"][0]["unique_orbits"] == 2
    with pytest.raises(ValueError, match="disagrees"):
        reference_pool_sensitivity([0., .02], ["low", "high"], [3, 1],
            433.15, ["low", "high"], raw_sample_indices=[0, 1, 1, 0])


def test_hu12_finite_support_multiplicities_and_exact_weights():
    root = Path(__file__).resolve().parents[1] / "artifacts/current/extras/learning_all_parents"
    with np.load(root / "parents/Hu_O3_12.npz", allow_pickle=False) as z:
        parent = {k: z[k] for k in z.files}
    with np.load(root / "learning_Hu_O3_12_uma.npz", allow_pickle=False) as z:
        occupancy, energy, n = z["occupancy"], z["energy_eV"], z["n_Li"]
    support = validate_exhaustive_support(occupancy, parent["permutations"])
    assert support["representatives"] == 87
    assert [r["representatives"] for r in support["counts"]] == [1, 1, 3, 6, 11, 13, 17, 13, 11, 6, 3, 1, 1]
    assert [r["raw_patterns"] for r in support["counts"]] == [math.comb(12, i) for i in range(13)]
    selected = n == 6
    keys = canonical_keys(occupancy[selected], parent["permutations"])
    by_key = dict(zip(keys, energy[selected]))
    masks = np.zeros((math.comb(12, 6), 12), np.uint8)
    for row, sites in zip(masks, itertools.combinations(range(12), 6)):
        row[list(sites)] = 1
    raw_e = np.asarray([by_key[k] for k in canonical_keys(masks, parent["permutations"])])
    actual = static_weights(energy[selected], np.asarray(support["multiplicities"])[selected],
                            433.15, measure_kind="exhaustive_orbits")
    raw_p = np.exp(-(raw_e - raw_e.min()) / (KB_EV_K * 433.15)); raw_p /= raw_p.sum()
    assert actual["mean_energy_eV"] == pytest.approx(float(raw_p @ raw_e), abs=1e-10)
    assert actual["microstate_or_particle_ess"] == pytest.approx(1 / np.dot(raw_p, raw_p), rel=1e-10)


def test_reference_total_energy_conversion_and_double_count_guard(tmp_path):
    parent = tmp_path / "parent.npz"
    np.savez(parent, numbers=[27, 3, 3], positions_A=np.zeros((3, 3)), cell_A=np.eye(3),
             target_indices=[1, 2], permutations=[[0, 1]])
    labels = tmp_path / "reference.npz"
    np.savez(labels, occupancy=[[1, 0]], energy_eV_atom=[-5.], n_atoms=[2], raw_frequency=[3])
    loaded = load_reference(labels, parent, "uniform_raw_sample")
    assert loaded["energy_eV"].tolist() == [-10.]
    np.savez(labels, occupancy=[[1, 0]], energy_eV_atom=[-5.], n_atoms=[2], raw_frequency=[3], orbit_multiplicity=[2])
    with pytest.raises(ValueError, match="additional orbit multiplicity"):
        load_reference(labels, parent, "uniform_raw_sample")
    np.savez(labels, occupancy=[[1, 0]], energy_eV=[np.nan], n_atoms=[2], raw_frequency=[3])
    with pytest.raises(ValueError, match="Incomplete/nonfinite"):
        load_reference(labels, parent, "uniform_raw_sample")


def test_full_cli_writes_flattened_requests_without_duplicate_metadata(tmp_path, monkeypatch):
    import csv
    from cathode_cvae.acquisition import main
    from cathode_cvae.conditions import fit_schema, save_schema
    # Only two synthetic scalar rows: this tests CLI wiring, not physical inference.
    monkeypatch.setattr("cathode_cvae.runtime.require_compute_allocation", lambda: None)
    parent = tmp_path/"parent.npz"
    np.savez(parent, numbers=[27,3,3], positions_A=np.zeros((3,3)), cell_A=np.eye(3),
             target_indices=[1,2], permutations=[[0,1]])
    reference = tmp_path/"reference.npz"
    np.savez(reference,occupancy=[[1,0],[0,1]],n_atoms=[2,2],n_Li=[1,1],
             energy_eV=[-1.,-.98],raw_frequency=[2,1])
    schema=fit_schema(np.ones(10),np.linspace(-2,2,10),["train"]*10,n_sites=2)
    schema_path=tmp_path/"schema.json";save_schema(schema,schema_path)
    out=tmp_path/"out"
    result=main(["--reference",str(reference),"--parent",str(parent),"--schema",str(schema_path),
                 "--reference-kind","uniform_raw_sample","--temperatures","400", "--budget","7", "--output-dir",str(out)])
    assert len(result["plans"])==1
    with (out/"requests.csv").open() as f: rows=list(csv.DictReader(f))
    assert len(rows)==12
    assert all(r["n_target"]=="1" and r["allocation_id"] for r in rows)
    for policy in {r["allocation_policy"] for r in rows}:
        assert sum(int(r["attempts"]) for r in rows if r["allocation_policy"]==policy)==7
