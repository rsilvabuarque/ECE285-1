"""Cached arithmetic allowances must not hide changed identities or counts."""
import importlib.util
import json
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


def test_portable_pair_replay_uses_the_actual_exclusive_source_parser(tmp_path, monkeypatch):
    """The real parser must accept compact facts without any queue argument."""
    import cathode_cvae.runtime
    monkeypatch.setattr(cathode_cvae.runtime, 'require_compute_allocation', lambda: None)
    monkeypatch.syspath_prepend(str(Path(__file__).parents[1]/'scripts/overnight_20261108'))
    paired = replay.load_helper('paired_evaluate')
    seen = []
    def record(args):
        assert args.facts is not None and args.queue is None
        args.output_dir.mkdir(parents=True)
        seen.append(args)
    monkeypatch.setattr(paired, 'evaluate', record)
    monkeypatch.setattr(replay, 'load_helper', lambda name: paired)
    packet = tmp_path/'packet'; packet.mkdir()
    stage = dict(name='paired',kind='paired',facts_sha256='0'*64,expected_tables={})
    for key in ['prepared','registration','attempts','history','deletion_schema',
                'endpoint_schema','fixed_thresholds','facts']:
        stage[key] = key
    (packet/'replay_manifest.json').write_text(json.dumps(dict(
        status='FROZEN_COMPLETE_REPLAY_PACKET',files={},stages=[stage])))
    result = replay.main(['--packet',str(packet),'--output-dir',str(tmp_path/'out')])
    assert result['status'] == 'PASS' and len(seen) == 1
