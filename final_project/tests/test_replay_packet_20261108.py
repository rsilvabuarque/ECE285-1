"""Cached arithmetic allowances must not hide changed identities or counts."""
import importlib.util
from pathlib import Path
import pytest

spec = importlib.util.spec_from_file_location('replay_packet',
    Path(__file__).parents[1] / 'scripts/overnight_20261108/replay_packet.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)


def write(path, header, value):
    path.write_text(header + '\n' + value + '\n')


def test_roundoff_does_not_hide_one_changed_attempt(tmp_path):
    a, b = tmp_path/'a.csv', tmp_path/'b.csv'
    write(b, 'mean_yield', '0.5')
    write(a, 'mean_yield', '0.5000000000000001')
    assert replay.compare_csv(a, b)['roundoff_only_cells'] == 1
    write(a, 'mean_yield', str(.5 + 1/44928))
    with pytest.raises(ValueError): replay.compare_csv(a, b)


def test_large_binary_identity_and_counts_remain_exact(tmp_path):
    a, b = tmp_path/'a.csv', tmp_path/'b.csv'
    write(a, 'canonical_mask', '111111111111111111111111111111111111111111111110')
    write(b, 'canonical_mask', '111111111111111111111111111111111111111111111111')
    with pytest.raises(ValueError): replay.compare_csv(a, b)
    write(a, 'attempts', '44927'); write(b, 'attempts', '44928')
    with pytest.raises(ValueError): replay.compare_csv(a, b)


def test_packet_paths_cannot_escape(tmp_path):
    with pytest.raises(ValueError): replay.contained(tmp_path, '../another-packet')
