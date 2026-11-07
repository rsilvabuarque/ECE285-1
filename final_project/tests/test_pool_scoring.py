"""Exact-geometry deduplication and completion checks, with no real MLIP calls."""
import copy
import json
import threading

import numpy as np
import pytest

from cathode_cvae import pool_scoring as ps
from cathode_cvae.data import sha256


def parent_file(tmp_path, reverse=False):
    path = tmp_path / 'parent.npz'
    perms = [np.arange(48)]
    if reverse:
        perms.append(np.arange(47, -1, -1))
    np.savez(path, numbers=np.r_[np.full(48,27),np.full(48,3),np.full(96,8)],
             positions_A=np.arange(576).reshape(192,3)/100,
             cell_A=np.eye(3)*20, li_indices=np.arange(48,96), permutations=np.asarray(perms))
    return path


def mask(count):
    return np.r_[np.zeros(count, dtype=np.uint8), np.ones(48-count, dtype=np.uint8)]


def pool_file(tmp_path, parent, counts, name='pool.npz', occupancy_only=False, noncanonical=False):
    masks = np.array([mask(n) for n in counts])
    if noncanonical:
        masks = masks[:, ::-1]
    keys = [''.join(map(str, row)) for row in masks]
    ids = np.array([f'mask48_{int(k,2):012x}' for k in keys])
    p = tmp_path / name
    arrays = dict(structure_ids=ids, parent_sha256=sha256(parent))
    arrays['occupancy' if occupancy_only else 'masks'] = 1-masks if occupancy_only else masks
    np.savez(p, **arrays)
    return p, ids


def cannot_certify(root, output):
    """An invalid/missing result may fail loudly or return an incomplete receipt."""
    try:
        receipt = ps.collect(root, output)
    except (ValueError, RuntimeError, FileNotFoundError):
        return
    assert receipt['status'] != 'PASS', 'Missing or mismatched physical labels must not receive a PASS receipt'


def test_overlapping_requests_score_each_physical_model_pair_once(tmp_path):
    parent = parent_file(tmp_path)
    first, _ = pool_file(tmp_path, parent, [12, 24])
    second, _ = pool_file(tmp_path, parent, [24, 36], name='second.npz')
    queue = tmp_path / 'queue'
    a = ps.register_pool(queue, first, parent, models=('uma','mace'), batch_size=1)
    b = ps.register_pool(queue, second, parent, models=('uma','mace'), batch_size=1)
    c = ps.register_pool(queue, first, parent, models=('uma','mace'), batch_size=1)
    assert a['added_geometry_model_pairs'] == 4
    assert b['added_geometry_model_pairs'] == 2
    assert c['added_geometry_model_pairs'] == 0
    registry = json.loads((queue/'registry.json').read_text())
    assert len(registry['physical_keys']) == 6
    assert len({t['task_id'] for t in registry['tasks']}) == len(registry['tasks'])


def test_identity_separates_parent_mask_teacher_head_checkpoint_and_protocol(monkeypatch):
    key = ps.physical_key('parent', '0'*48, 'uma')
    assert key != ps.physical_key('other-parent', '0'*48, 'uma')
    assert key != ps.physical_key('parent', '1'+'0'*47, 'uma')
    assert key != ps.physical_key('parent', '0'*48, 'mace')
    original = copy.deepcopy(ps.MODELS['uma'])
    altered = copy.deepcopy(original); altered['head'] = 'other-head'
    monkeypatch.setitem(ps.MODELS, 'uma', altered)
    assert key != ps.physical_key('parent', '0'*48, 'uma')
    altered = copy.deepcopy(original); altered['files'] = {'weights.pt':'other-checkpoint-sha'}
    monkeypatch.setitem(ps.MODELS, 'uma', altered)
    assert key != ps.physical_key('parent', '0'*48, 'uma')
    monkeypatch.setitem(ps.MODELS, 'uma', original)
    monkeypatch.setattr(ps, 'PROTOCOL', 'relaxed coordinates; a different physical question')
    assert key != ps.physical_key('parent', '0'*48, 'uma')


def test_noncanonical_masks_and_nonpositive_batch_size_are_rejected(tmp_path):
    parent = parent_file(tmp_path, reverse=True)
    invalid, _ = pool_file(tmp_path, parent, [12], noncanonical=True)
    with pytest.raises(ValueError):
        ps.register_pool(tmp_path/'bad-canonical', invalid, parent, models=('uma',))
    valid, _ = pool_file(tmp_path, parent, [12], name='valid.npz')
    for batch in [0, -1]:
        with pytest.raises(ValueError):
            ps.register_pool(tmp_path/f'bad-batch{batch}', valid, parent, models=('uma',), batch_size=batch)


def test_uncertified_id_only_cache_cannot_avoid_a_physical_evaluation(tmp_path):
    parent = parent_file(tmp_path)
    pool, ids = pool_file(tmp_path, parent, [12])
    queue = tmp_path / 'queue'; queue.mkdir()
    # This could be from a different head, cell, precision or geometry. Matching
    # a mask ID alone is deliberately insufficient evidence for an energy label.
    np.savez(queue/'cache_uma.npz', structure_ids=ids, energy_eV=[-123.], n_atoms=[156],
             n_Li=[12], force_component_rms_eV_A=[0.], force_max_eV_A=[0.],
             stress_eV_A3=np.zeros((1,6)), seconds=[0.])
    try:
        receipt = ps.register_pool(queue, pool, parent, models=('uma',))
    except ValueError:
        return
    assert receipt['added_geometry_model_pairs'] == 1, 'Uncertified cache entries cannot silently skip inference'


@pytest.mark.parametrize('field,value', [('tf32', True), ('dtype', 'float64'), ('head', 'wrong-head'), ('geometry_protocol', 'relaxed')])
def test_cache_certificate_rejects_different_precision_head_or_geometry(tmp_path, field, value):
    parent = parent_file(tmp_path)
    pool, ids = pool_file(tmp_path, parent, [12])
    queue = tmp_path/'queue'; queue.mkdir()
    cache = queue/'cache_uma.npz'
    np.savez(cache, structure_ids=ids, energy_eV=[-123.], n_atoms=[156], n_Li=[12])
    context = ps.cache_context('uma', sha256(parent))
    assert context['dtype'] == 'float32' and context['tf32'] is False
    context[field] = value
    (queue/'cache_uma.json').write_text(json.dumps(dict(context=context, npz_sha256=sha256(cache))))
    with pytest.raises(ValueError, match='context'):
        ps.register_pool(queue,pool,parent,models=('uma',))


def test_missing_cache_is_not_a_complete_result(tmp_path):
    parent = parent_file(tmp_path)
    pool, ids = pool_file(tmp_path, parent, [12])
    queue = tmp_path/'queue'
    ps.register_pool(queue, pool, parent, models=('uma',))
    p = queue/'registry.json'; registry = json.loads(p.read_text())
    registry['tasks'] = []
    for value in registry['physical_keys'].values():
        value['provenance'] = 'exact_geometry_cache'
    p.write_text(json.dumps(registry))
    cannot_certify(queue, tmp_path/'missing-cache-collect')


@pytest.mark.parametrize('bad_ids', ['partial', 'wrong-identities'])
def test_task_rows_must_match_the_complete_registered_candidate_set(tmp_path, bad_ids):
    parent = parent_file(tmp_path)
    pool, ids = pool_file(tmp_path, parent, [12,24])
    queue = tmp_path/'queue'
    ps.register_pool(queue, pool, parent, models=('uma',), batch_size=64)
    task = json.loads((queue/'registry.json').read_text())['tasks'][0]
    result_ids = ids[:1] if bad_ids == 'partial' else np.array(['wrong-a','wrong-b'])
    n = len(result_ids)
    p = queue/'scores/uma'/f"{task['task_id']}.npz"; p.parent.mkdir(parents=True)
    np.savez(p, structure_ids=result_ids, status=np.full(n,'success'), energy_eV=np.ones(n),
        n_atoms=np.asarray([156,168])[:n], n_Li=np.asarray([12,24])[:n], force_component_rms_eV_A=np.zeros(n),
        force_max_eV_A=np.zeros(n), stress_eV_A3=np.zeros((n,6)), seconds=np.zeros(n))
    p.with_suffix('.json').write_text(json.dumps(dict(task=task,output_sha256=sha256(p),status='complete')))
    cannot_certify(queue, tmp_path/'incomplete-collect')


@pytest.mark.parametrize('occupancy_only', [False, True])
def test_mock_worker_keeps_geometry_mapping_and_completes_without_real_mlip(tmp_path, monkeypatch, occupancy_only):
    from cathode_cvae.mlip import calculators
    parent = parent_file(tmp_path)
    pool, ids = pool_file(tmp_path, parent, [12], occupancy_only=occupancy_only)
    queue = tmp_path/'queue'; ps.register_pool(queue, pool, parent, models=('uma',))
    calls = []
    monkeypatch.setattr(calculators, 'make_calculator', lambda *args: (object(),dict(test_only=True)))

    def fake_evaluate(calc, atoms):
        calls.append(atoms.copy())
        (queue/'STOP').touch()
        return -float(len(atoms)), np.ones((len(atoms),3)), np.zeros(6)

    monkeypatch.setattr(calculators, 'evaluate', fake_evaluate)
    # Prevent a regression from hanging the suite if loading a pool fails before
    # reaching the fake calculator. This is a unit-test bound, not a retry policy.
    timer = threading.Timer(2., lambda: (queue/'STOP').touch()); timer.start()
    try:
        ps.worker(queue,'uma','unused-test-checkpoint',device='cpu',threads=1)
    finally:
        timer.cancel()
    assert len(calls) == 1
    assert len(calls[0]) == 156
    receipt = ps.collect(queue,tmp_path/'collected')
    assert receipt['status'] == 'PASS' and receipt['model_counts'] == {'uma':1}
    task = json.loads((queue/'registry.json').read_text())['tasks'][0]
    with np.load(queue/'scores/uma'/f"{task['task_id']}.npz") as z:
        # Li occupy parent rows 48..95. Retained target rows are 48..59.
        assert z['forces_parent_eV_A'].shape == (1,192,3)
        assert np.all(z['forces_parent_eV_A'][0,:60] == 1)
        assert np.all(z['forces_parent_eV_A'][0,60:96] == 0)
        assert np.all(z['forces_parent_eV_A'][0,96:] == 1)
