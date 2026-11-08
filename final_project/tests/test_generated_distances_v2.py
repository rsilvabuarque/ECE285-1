"""The cached-parent optimization must preserve exact reference and distance tables."""
import importlib.util
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
def module(name,path):
    s=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(s);s.loader.exec_module(m);return m
fixture_module=module('distance_fixture_v2',ROOT/'tests/test_generated_distances.py')
new=module('distance_cached_v2',ROOT/'scripts/overnight_20261108/audit_generated_distances_v2.py')

def test_cached_parent_path_preserves_reference_and_all_distances(tmp_path):
    generated,parent,data=fixture_module.fixture(tmp_path)
    for implementation,name in [(fixture_module.m,'old'),(new,'new')]:
        r=implementation.main(['--generated',str(generated),'--parent',str(parent),'--training-data',str(data),'--output-dir',str(tmp_path/name),'--max-seconds','30'])
        assert r['status']=='PASS'
    for name in ['TRAIN_reference.csv','attempt_distances.csv','metrics_by_condition_seed.csv','metrics_by_seed.csv']:
        assert (tmp_path/'old'/name).read_bytes()==(tmp_path/'new'/name).read_bytes()
