"""Identity-only geometry/history extraction and independent-lineage parallelism."""
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from pathlib import Path
import sys

import numpy as np
import pytest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/"scripts/overnight_20261108"))
import paired_common as common
import paired_history as history
import paired_mapping_audit as mapping
from test_paired_relaxation import parent


def test_coordinate_and_pool_identity_readers_never_access_poisoned_energy_arrays(tmp_path):
    p=parent();slots=np.array([0,1,2,3]);xyz=p["positions_A"][slots].copy()
    end=xyz.copy();end[-1]=p["positions_A"][4]
    path=tmp_path/"TEST_geometry.npz"
    np.savez(path,positions_A=[xyz,end],numbers=p["numbers"][slots],cell_A=p["cell_A"],parent_indices=slots,
        energy_eV=np.array([object()],dtype=object))
    keys,audit=history.coordinate_keys(path,p,mapping.nearest_parent_separations(p))
    assert keys=={"0011","0101"} and audit["frames_with_mobile_reassignment"]==1
    with np.load(path,allow_pickle=False) as z:
        with pytest.raises(ValueError,match="Object arrays"):_=z["energy_eV"]
    pool=tmp_path/"TEST_pool.npz";np.savez(pool,canonical_masks=["0011"],energy_eV=np.array([object()],dtype=object))
    assert history.pool_keys(pool,p)[0]=={"0011"}


def test_per_lineage_process_worker_matches_serial_with_unreadable_energy(tmp_path):
    p=parent();slots=np.array([0,1,2,3]);xyz=p["positions_A"][slots].copy()
    end=xyz.copy();end[-1]=p["positions_A"][4]
    queue=tmp_path/"queue";directory=queue/"results/synthetic_task";directory.mkdir(parents=True)
    np.savez(directory/"arrays.npz",trajectory_positions_A=[xyz,end],parent_indices=slots,
        energy_eV=np.array([object()],dtype=object))
    common.atomic_json(directory/"status.json",dict(arrays_sha256=common.sha256(directory/"arrays.npz"),context_hash="synthetic"))
    payload=(dict(task_id="synthetic_task"),0,"test",str(queue),p,mapping.nearest_parent_separations(p),"synthetic")
    expected=mapping.audit_task(payload)
    with ProcessPoolExecutor(max_workers=2,mp_context=multiprocessing.get_context("spawn")) as executor:
        actual=list(executor.map(mapping.audit_task,[payload,payload]))
    assert actual==[expected,expected]
    assert expected[0]["frames_with_mobile_reassignment"]==1
