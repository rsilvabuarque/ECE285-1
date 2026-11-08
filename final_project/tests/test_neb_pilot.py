"""Synthetic checks of event identity, periodic branch, and force/call accounting.

Toy potentials below are software fixtures, not LCO/MLIP evidence.
"""
from pathlib import Path
import sys

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.calculator import Calculator, all_changes
from ase.mep import NEB

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107"))
from neb_common import (BudgetExpired, CallLedger, CountingCalculator, continuous_hop_displacement, discrete_barriers, mapping,
    digest, snapshot, validate_endpoint_event, write_json)
from neb_run import fmax, resample_band, safe_idpp
import neb_run


class ToyHarmonic(Calculator):
    implemented_properties = ["energy", "forces", "stress"]
    def __init__(self): super().__init__(); self.actual_calls = 0
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes); self.actual_calls += 1
        self.results = dict(energy=float((atoms.positions**2).sum()), forces=-2*atoms.positions,
                            stress=np.zeros(6))


def test_actual_calculate_calls_exclude_property_cache_hits_and_keep_image_caches(tmp_path):
    backend = ToyHarmonic(); ledger = CallLedger(tmp_path/"calls.jsonl", 30)
    a, b = Atoms("Li",positions=[[1,0,0]]), Atoms("Li",positions=[[2,0,0]])
    a.calc, b.calc = [CountingCalculator(backend,ledger,str(i)) for i in range(2)]
    assert a.get_potential_energy() == 1.
    np.testing.assert_array_equal(a.get_forces(),[[-2,0,0]])
    assert ledger.attempts == 1
    assert b.get_potential_energy() == 4.
    np.testing.assert_array_equal(a.get_forces(),[[-2,0,0]])
    assert ledger.attempts == backend.actual_calls == 2
    a.positions[0,0] = 3
    assert a.get_potential_energy() == 9.
    assert ledger.attempts == 3
    assert ledger.summary()["successful_calculator_EF"] == 3


def test_budget_guard_prevents_new_calculator_call_and_unknown_snapshot_stays_unknown(tmp_path):
    backend = ToyHarmonic(); ledger = CallLedger(tmp_path/"calls.jsonl", -1)
    atom = Atoms("Li",positions=[[0,0,0]]); atom.calc = CountingCalculator(backend,ledger,"image")
    with pytest.raises(BudgetExpired): atom.get_forces()
    assert ledger.attempts == backend.actual_calls == 0
    saved = snapshot([atom],tmp_path/"failed.npz")
    assert not saved["labels_match_geometry"].any()
    assert np.isnan(saved["energy_eV"]).all()


def test_parent_site_mapping_and_physical_event_validation_without_canonicalization():
    parent = dict(numbers=np.asarray([3,3,27,8]), positions_A=np.asarray([[0,0,0],[1,0,0],[0,2,0],[1,2,0.]]),
        cell_A=np.eye(3)*4,li_indices=np.asarray([0,1]))
    indices = np.asarray([0,2,3])
    a = Atoms(numbers=parent["numbers"][indices],positions=parent["positions_A"][indices],cell=parent["cell_A"],pbc=True)
    b = a.copy(); b.positions[0,0] = 1.
    pre, post = mapping(a,parent,indices), mapping(b,parent,indices)
    assert validate_endpoint_event(pre,post,[0],[1],0,0,1)[0]
    collapsed = mapping(a.copy(),parent,indices)
    valid, reason = validate_endpoint_event(pre,collapsed,[0],[1],0,0,1)
    assert not valid and reason == "both_endpoints_returned_to_same_site_assignment"


def test_arclength_refinement_preserves_continuous_periodic_branch():
    images = [Atoms("Li",positions=[[x,0,0]],cell=np.eye(3)*4,pbc=True) for x in (3.5,4.,4.5)]
    refined = resample_band(images,7)
    assert len(refined) == 9
    assert refined[0].positions[0,0] == 3.5
    assert refined[-1].positions[0,0] == 4.5
    assert np.all(np.diff([im.positions[0,0] for im in refined]) > 0)


def test_continuous_frame_increments_resolve_long_branch_instead_of_endpoint_MIC():
    positions=np.zeros((6,2,3)); positions[:,0,0]=np.linspace(0,2.5,6); positions[:,1,1]=2
    audit=continuous_hop_displacement(positions,[3,27],positions[0],[0,1],np.eye(3)*4,0)
    assert audit["displacement_A"] == [2.5,0.,0.]
    assert audit["maximum_migrating_frame_step_A"] == .5
    # Endpoint MIC would return -1.5 A, the other winding, rather than +2.5 A.
    from ase.geometry import find_mic
    assert find_mic(positions[-1,0]-positions[0,0],np.eye(3)*4,pbc=True)[0][0] == -1.5


class ToyDoubleWell(Calculator):
    implemented_properties = ["energy", "forces"]
    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms,properties,system_changes)
        x,y,z = atoms.positions[0]
        self.results = dict(energy=float((x*x-1)**2+y*y+z*z),
            forces=np.asarray([[-4*x*(x*x-1),-2*y,-2*z]]))


def test_projected_neb_force_differs_from_physical_image_forces_and_barrier_is_per_hop():
    images = [Atoms("Li",positions=[[x,0,0]]) for x in np.linspace(-1,1,7)]
    for atom in images: atom.calc = ToyDoubleWell()
    neb = NEB(images,method="improvedtangent",k=.1,climb=True,parallel=False)
    projected = neb.get_forces()
    physical = np.asarray([im.get_forces() for im in images])
    assert np.linalg.norm(projected,axis=1).max() < 1e-12
    assert np.linalg.norm(physical,axis=2).max() > 1.
    barrier = discrete_barriers([im.get_potential_energy() for im in images])
    assert barrier["internal_peak"]
    assert barrier["discrete_forward_span_eV"] == barrier["discrete_reverse_span_eV"] == 1.


def test_interrupted_idpp_restores_real_calculator_provenance(monkeypatch):
    images = [Atoms("Li",positions=[[x,0,0]]) for x in (-1,0,1)]
    originals = [ToyHarmonic() for _ in images]
    for image,calc in zip(images,originals): image.calc = calc
    band = NEB(images,method="improvedtangent")
    def interrupted(neb, **kwargs):
        for image in neb.images: image.calc = ToyDoubleWell()
        raise BudgetExpired("synthetic IDPP interruption")
    monkeypatch.setattr(neb_run,"idpp_interpolate",interrupted)
    with pytest.raises(BudgetExpired): safe_idpp(band)
    assert all(im.calc is calc for im,calc in zip(images,originals))
    assert sum(calc.actual_calls for calc in originals) == 0


def test_nonfinite_projected_force_cannot_be_reported_as_converged():
    with pytest.raises(ValueError,match="Nonfinite"): fmax([[np.nan,0,0]])


def test_complete_neb_runner_with_synthetic_potential_only(tmp_path,monkeypatch):
    """Exercise exports/receipts/optimizer APIs without loading any MLIP."""
    import json
    from cathode_cvae.mlip import calculators
    class ToyEvent(Calculator):
        implemented_properties = ["energy","forces","stress"]
        def calculate(self,atoms=None,properties=("energy",),system_changes=all_changes):
            super().calculate(atoms,properties,system_changes)
            x,y,z = atoms.positions[0]
            reference = np.asarray([[0,2,0],[1,2,0.]])
            delta = atoms.positions[1:]-reference
            self.results = dict(energy=float((x*x-1)**2+y*y+z*z+(delta**2).sum()),
                forces=np.vstack([[-4*x*(x*x-1),-2*y,-2*z],-2*delta]),stress=np.zeros(6))
    prepared = tmp_path/"prepared"; prepared.mkdir()
    parent = dict(numbers=np.asarray([3,3,27,8]),positions_A=np.asarray([[-1,0,0],[1,0,0],[0,2,0],[1,2,0.]]),
        cell_A=np.eye(3)*8,li_indices=np.asarray([0,1]))
    parent_file = prepared/"parent.npz"; np.savez_compressed(parent_file,**parent)
    indices = np.asarray([0,2,3]); bindings={}
    for name,x in (("initial",-1.),("final",1.)):
        positions = parent["positions_A"][indices].copy(); positions[0,0] = x
        path = prepared/(name+".npz")
        np.savez_compressed(path,positions_A=positions,numbers=parent["numbers"][indices],cell_A=parent["cell_A"],
            pbc=np.ones(3,bool),parent_indices=indices)
        bindings[name]=dict(path=path.name,sha256=digest(path))
    event=dict(event_id="synthetic",endpoint_inputs=bindings,expected_initial_assignment=[0],expected_final_assignment=[1],
        migrating_atom_index=0,source_parent_Li_site=0,destination_parent_Li_site=1)
    fire=dict(dt_ASE=.1,dtmax_ASE=1.,maxstep_A=.05)
    protocol=dict(per_event_wall_seconds=2400,maximum_calculator_EF_attempts_per_event=10000,
        refinement=dict(event_id="synthetic"),parent=dict(path=str(parent_file),sha256=digest(parent_file)),events=[event],
        endpoint_relaxation=dict(**fire,Fmax_eV_A=.03,max_steps_per_endpoint=500),
        initial_band=dict(**fire,interior_images=5,k_eV_A2=.1,method="improvedtangent",IDPP_max_steps=5,
            IDPP_optimizer_fmax=.1,IDPP_threshold_scope="synthetic auxiliary fixture",initial_Fmax_eV_A=.1,
            initial_max_steps=300,climbing_Fmax_eV_A=.05,climbing_max_steps=700))
    write_json(prepared/"protocol.json",protocol)
    monkeypatch.setattr(neb_run,"verify_prepared",lambda _:protocol)
    monkeypatch.setattr(calculators,"make_calculator",lambda *a,**k:(ToyEvent(),{"model":"synthetic_fixture_not_MLIP"}))
    output=tmp_path/"result"
    code=neb_run.main(["--prepared",str(prepared),"--event","synthetic","--output-dir",str(output),
        "--checkpoint",str(tmp_path/"unused"),"--device","cpu","--budget-seconds","30"])
    result=json.loads((output/"receipt.json").read_text())
    assert code == 0 and result["status"] == "CONVERGED_NUMERICALLY_UNVALIDATED"
    assert result["teacher"]["model"] == "synthetic_fixture_not_MLIP"
    assert result["climbing_band"]["NEB_projected_Fmax_eV_A"] <= .05
    assert result["forward_barrier_eV"] == pytest.approx(1.,abs=.01)
    assert (output/"DFT_candidates_UMA_band5/image_03.POSCAR").is_file()
    assert result["call_accounting"]["actual_calculator_EF_attempts"] > 0
