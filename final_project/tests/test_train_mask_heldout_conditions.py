"""TRAIN/VAL condition equivalence and protected-label perturbation tests."""
import json

import numpy as np
import pytest
import torch

from cathode_cvae.conditions import classify, encode_observations, fit_schema
from cathode_cvae.descriptor import fit_descriptor_schema, encode_observations as encode_descriptor
from cathode_cvae.learning import train_mask


@pytest.mark.parametrize('mode', ['composition', 'energy', 'multibin'])
def test_train_val_conditions_match_previous_all_row_encoding_byte_for_byte(mode):
    counts = np.array([2, 2, 2, 2, 2, 2])
    energies = np.array([-2., -1., 0., .5, 2., 3.])
    splits = np.array(['train'] * 3 + ['val'] * 2 + ['test'])
    old_schema = fit_schema(counts, energies, splits, n_sites=4)
    train = splits == 'train'
    new_schema = fit_schema(counts[train], energies[train], splits[train], n_sites=4)
    assert old_schema == new_schema
    old_conditions = encode_observations(old_schema, counts, energies, mode=mode)
    old_bins = classify(old_schema, counts, energies)
    got, bins, active = train_mask.encode_train_val_conditions(new_schema, counts, energies, splits, mode)
    assert got[active].tobytes() == old_conditions[active].tobytes()
    assert bins[active].tobytes() == old_bins[active].tobytes()
    assert np.all(got[splits == 'test'] == 0) and np.all(bins[splits == 'test'] == '')
    perturbed = energies.copy(); perturbed[-1] = np.nan
    changed, changed_bins, _ = train_mask.encode_train_val_conditions(new_schema, counts, perturbed, splits, mode)
    assert changed.tobytes() == got.tobytes()
    assert changed_bins.tobytes() == bins.tobytes()


def test_descriptor_schema_and_train_val_conditions_match_previous_all_row_calculation():
    counts = np.array([2] * 6); splits = np.array(['train'] * 3 + ['val'] * 2 + ['test'])
    energies = np.array([-2., -1., 0., .5, 2., 3.]); descriptor = np.array([.1, .2, .3, .4, .5, .6])
    schema = fit_schema(counts, energies, splits, n_sites=4)
    old_bins = classify(schema, counts, energies)
    old_descriptor = fit_descriptor_schema(descriptor, counts, old_bins, splits)
    old_extra = encode_descriptor(old_descriptor, counts, old_bins, descriptor)
    _, bins, active = train_mask.encode_train_val_conditions(schema, counts, energies, splits, 'multibin')
    train = splits == 'train'
    new_descriptor = fit_descriptor_schema(descriptor[train], counts[train], bins[train], splits[train])
    assert new_descriptor == old_descriptor
    descriptor[-1] = np.nan
    extra = train_mask.encode_train_val_descriptors(new_descriptor, counts, bins, descriptor, active)
    assert extra[active].tobytes() == old_extra[active].tobytes()
    assert np.all(extra[splits == 'test'] == 0)


@pytest.mark.parametrize('with_descriptor', [False, True])
def test_test_energy_perturbation_does_not_change_tiny_training(monkeypatch, tmp_path, with_descriptor):
    monkeypatch.setattr(train_mask, 'require_compute_allocation', lambda: None)
    parent = tmp_path / 'parent.npz'; np.savez(parent, permutations=np.arange(4)[None])
    occupancy = np.asarray([[1, 1, 0, 0], [1, 0, 1, 0], [1, 0, 0, 1],
                            [0, 1, 1, 0], [0, 1, 0, 1], [0, 0, 1, 1]], np.uint8)
    ids = np.asarray([f'synthetic_{i}' for i in range(6)])
    data_paths = []
    for name, test_energy in [('before', 3.), ('after', np.nan)]:
        data = tmp_path / f'{name}.npz'
        np.savez(data, occupancy=occupancy, n_target=np.full(6, 2),
                 energy_eV_atom=np.array([-2., -1., 0., .5, 2., test_energy]),
                 weight=np.array([1., 2., .7, 1., 1., 1.]),
                 split=np.asarray(['train']*3 + ['val']*2 + ['test']),
                 structure_ids=ids, parent_sha256=np.asarray(train_mask.sha256(parent)))
        data_paths.append(data)
    descriptor_path = tmp_path / 'descriptor.npz'
    # TEST has no descriptor entry: it must not be needed for fitting/validation.
    np.savez(descriptor_path, structure_ids=ids[:5], descriptor_values=np.array([.1, .2, .3, .4, .5]))
    checkpoints = []
    for idx, data in enumerate(data_paths):
        out = tmp_path / f'fit_{idx}'
        args = ['--data', str(data), '--parent', str(parent), '--output-dir', str(out),
                '--kind', 'ar_cvae', '--condition-mode', 'multibin', '--latent-dim', '4',
                '--max-updates', '2', '--epochs', '10', '--minimum-epochs', '10',
                '--patience', '10', '--batch-size', '2', '--validation-samples', '1',
                '--class-balance', '--device', 'cpu', '--threads', '1', '--max-seconds', '60']
        if with_descriptor: args += ['--descriptor-values', str(descriptor_path)]
        train_mask.main(args)
        config = json.loads((out / 'configuration.json').read_text())
        assert set(config['observed_rows_outside_training_bin_support']) == {'train', 'val'}
        assert 'fully decompressed' in config['heldout_energy_handling']
        checkpoints.append(torch.load(out / 'best.pt', map_location='cpu', weights_only=False))
    assert checkpoints[0]['state_dict'].keys() == checkpoints[1]['state_dict'].keys()
    for name in checkpoints[0]['state_dict']:
        assert torch.equal(checkpoints[0]['state_dict'][name], checkpoints[1]['state_dict'][name]), name
    assert checkpoints[0]['val_objective'] == checkpoints[1]['val_objective']
