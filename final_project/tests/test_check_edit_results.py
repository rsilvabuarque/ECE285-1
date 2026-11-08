"""Independent editor audit arithmetic and exact-denominator fixtures."""
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

path=Path(__file__).resolve().parents[1]/"scripts/conditioning_20261107/check_edit_results.py"
spec=importlib.util.spec_from_file_location("check_edit_results",path)
check=importlib.util.module_from_spec(spec);spec.loader.exec_module(check)


def test_independent_all192_canonicalization_matches_direct_lexicographic_action():
    permutations=[]
    for flip12 in (-1,1):
        for rotation12 in range(12):
            for flip4 in (-1,1):
                for rotation4 in range(4):
                    permutations.append([((flip4*a+rotation4)%4)*12+(flip12*b+rotation12)%12
                                         for a in range(4) for b in range(12)])
    permutations=np.asarray(permutations)
    assert len({tuple(p) for p in permutations})==192
    row=np.random.default_rng(1).integers(0,2,48,dtype=np.uint8)
    expected=min("".join(map(str,(1-row[np.argsort(p)]).tolist())) for p in permutations)
    result=check.canonical(np.stack([row,row[np.argsort(permutations[57])]]),permutations)
    assert result.tolist()==[expected,expected]


def test_uniform_subset_and_actual_edit_are_independently_reconstructed():
    seed=np.zeros(48,np.uint8);seed[:12]=1
    u=np.full(48,.9);u[1]=.01;u[4]=.01
    remove=check.uniform_subset(seed==1,2,u)
    add=check.uniform_subset(seed==0,2,np.linspace(0,.99,48))
    child=seed-remove+add
    check.exact_edit(seed,child,remove,add,2)
    assert remove.sum()==add.sum()==2 and np.count_nonzero(seed!=child)==4
    with pytest.raises(ValueError):check.exact_edit(seed,seed,remove,add,2)


def test_frozen_bin_ties_and_exact_metric_denominators(tmp_path):
    schema={"compositions":{"12":{"edges_eV_atom":[-5.,-4.]}}}
    assert check.measured_bin(schema,12,-5.)=="low"
    assert check.measured_bin(schema,12,-4.)=="middle"
    assert check.measured_bin(schema,12,-3.9)=="high"
    file=tmp_path/"metrics.csv"
    file.write_text("method,attempts,yield\nmutation,128,0.5\n")
    check.compare_table(file,[dict(method="mutation",attempts=128,yield_value=.5)], ["method"],["attempts"])
    with pytest.raises(ValueError,match="denominator/count"):
        check.compare_table(file,[dict(method="mutation",attempts=127)],["method"],["attempts"])


def test_seed_label_parity_reports_bin_changes_without_reselection(tmp_path):
    counts=np.asarray([12,24,36]);occupancy=np.asarray([np.arange(48)<n for n in counts],np.uint8)
    keys=np.asarray(["".join(map(str,(1-row).tolist())) for row in occupancy])
    source_ids=np.asarray(["synthetic_H0_"+str(n) for n in counts])
    scoring_ids=np.asarray([f"mask48_{int(key,2):012x}" for key in keys])
    h0=tmp_path/"H0.npz"
    np.savez_compressed(h0,occupancy=occupancy,structure_ids=source_ids,energy_eV_atom=np.full(3,-5.2))
    bank=tmp_path/"bank.npz"
    np.savez_compressed(bank,occupancy=occupancy,seed_ids=source_ids,H0_source_rows=np.arange(3),
        canonical_seed_keys=keys,n_Li=counts,seed_bin=np.full(3,"low"))
    prep=tmp_path/"preparation.json"
    prep.write_text(json.dumps(dict(source_hashes=dict(H0=dict(path=str(h0),sha256=check.digest(h0))))))
    protocol=dict(sources=dict(preparation=dict(path=str(prep)),seedbank=dict(path=str(bank))))
    scores=dict(structure_ids=scoring_ids,n_atoms=144+counts,n_Li=counts,energy_eV=-4.9*(144+counts))
    schema=dict(compositions={str(n):dict(edges_eV_atom=[-5.,-4.]) for n in counts})
    out=tmp_path/"audit";out.mkdir()
    result=check.seed_parity(protocol,[dict(heldout_seed_bank_row=i) for i in range(3)],scores,
        {sid:i for i,sid in enumerate(scoring_ids)},schema,out)
    assert result["classification_changed_seeds"]==3
    assert result["median_absolute_difference_meV_atom"]==pytest.approx(300.)
    assert result["no_arbitrary_pass_tolerance"] and result["no_reselection"]
    assert all(r["median_absolute_difference_over_binwidth"]==pytest.approx(.3) for r in result["by_composition"])
    confusion=check.read_csv(out/"seed_bin_confusion.csv")
    assert len(confusion)==27
    assert sum(int(r["count"]) for r in confusion if r["H0_bin"]=="low" and r["current_UMA_bin"]=="middle")==3
