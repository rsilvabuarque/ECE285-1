"""Tiny synthetic tests; no protected project energies or accelerator are used."""
import csv
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

SCRIPT = Path(__file__).resolve().parents[1] / 'scripts/angular_20261110/development_boundary.py'
spec = importlib.util.spec_from_file_location('development_boundary', SCRIPT)
b = importlib.util.module_from_spec(spec)
spec.loader.exec_module(b)


class BoundaryTest(unittest.TestCase):
    def test_canonical_orbits_and_role_extraction(self):
        perm = np.tile(np.arange(48), (2, 1))
        perm[1, :2] = [1, 0]
        occ = np.zeros((3, 48), np.uint8)
        occ[np.arange(3), [0, 1, 2]] = 1
        keys = b.canonical_keys(occ, perm)
        self.assertEqual(keys[0], keys[1])
        self.assertNotEqual(keys[0], keys[2])
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'cohort.npz'
            np.savez(path, occupancy=occ, canonical_masks=keys,
                     split=np.array(['train', 'val', 'test']), parent_sha256=np.array('parent'))
            known, roles = b._npz_masks(path, {'permutations': perm}, 'parent', ('val', 'test'))
            self.assertEqual(known, set(keys))
            self.assertEqual(roles, {'val': {keys[1]}, 'test': {keys[2]}})

    def test_filter_keeps_attempt_denominator_and_raw_evidence(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); boundary = root/'boundary'; boundary.mkdir()
            perm = np.tile(np.arange(48), (2, 1)); perm[1, :2] = [1, 0]
            occ = np.zeros((6, 48), np.uint8)
            occ[np.arange(6), [0, 2, 0, 2, 0, 2]] = 1
            keys = b.canonical_keys(occ, perm)
            parent = root/'parent.npz'
            np.savez(parent, permutations=perm, numbers=np.ones(192))
            dummy = root/'input.txt'; dummy.write_text('frozen input')
            b.save_json(boundary/'known_history.json', [keys[0]])
            b.save_json(boundary/'protected.json', [keys[0]])
            r = {'status': 'FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY',
                 'history': b.binding(boundary/'known_history.json'),
                 'protected': b.binding(boundary/'protected.json'),
                 'parent': b.binding(parent), 'data': b.binding(dummy),
                 'request_schema': b.binding(dummy), 'original_history': b.binding(dummy),
                 'protected_orbits': 1, 'known_history_orbits': 1}
            b.save_json(boundary/'receipt.json', r)
            src1 = root/'generate_controls.py'; src1.write_text('generator')
            src2 = root/'generate_mask.py'; src2.write_text('loader')
            manifest = {'status':'FROZEN_DEVELOPMENT_BEFORE_GENERATION',
                'inputs': {'parent': r['parent'], 'data': r['data'], 'history': r['history']},
                'teacher_schemas':{}, 'fixed_thresholds':b.binding(dummy),
                'boundary_receipt':b.binding(boundary/'receipt.json'), 'plan':b.binding(dummy),
                'source_bindings':[b.binding(src1),b.binding(src2)],
                'models':[{'method':'uniform','replicate_id':1,'sampler':'uniform'}],
                'expected_primary_attempts':6, 'counts':[1],
                'requested_bins':['low','middle','high'], 'count_per_group':2}
            b.save_json(root/'manifest.json',manifest)
            gen=root/'generated'; gen.mkdir()
            protocol={'manifest':b.binding(root/'manifest.json'), 'inputs':manifest['inputs']}
            b.save_json(gen/'protocol.json',protocol)
            rows=[]
            for i, (bin_, key) in enumerate(zip(['low']*2+['middle']*2+['high']*2, keys)):
                rows.append(dict(attempt_id=str(i),method='uniform',replicate_id='1',n_target='1',
                    requested_bin=bin_,attempt=str(i % 2),diagnostic='primary',raw_row=str(i),status='generated',
                    canonical_mask=key,scoring_structure_id=f'mask48_{int(key,2):012x}',latent_dim='0',error=''))
            b.save_csv(gen/'attempts_all.csv',rows)
            uniforms=np.tile(np.arange(2)[:,None],(3,48))
            np.savez(gen/'raw_attempts.npz',occupancy=occ,uniforms=uniforms,latents=np.zeros((6,32)))
            np.savez(gen/'raw_primary.npz', occupancy=occ)
            np.savez(gen/'raw_diagnostics.npz',occupancy=occ[:0])
            files=['attempts_all.csv','raw_attempts.npz','raw_primary.npz','raw_diagnostics.npz']
            receipt={'status':'FROZEN_BEFORE_SCORING','protocol_sha256':b.digest(gen/'protocol.json'),
                'source_sha256':b.digest(src1),'runtime_loader_source':b.binding(src2), 'total_attempts':6,
                'outputs':{n:b.digest(gen/n) for n in files}}
            b.save_json(gen/'receipt.json',receipt)
            result=b.filter_bundle(gen,boundary,root/'derived')
            self.assertEqual(result['protected_attempts_excluded_before_scoring'],3)
            self.assertEqual(result['total_attempts'],6)
            self.assertEqual(result['unique_geometries'],1)
            self.assertEqual(b.digest(gen/'raw_attempts.npz'), b.digest(root/'derived/raw_attempts.npz'))
            derived=b.read_csv(root/'derived/attempts_all.csv')
            self.assertEqual(sum(r['status']=='protected_boundary' for r in derived),3)
            self.assertEqual(sum(r['status']=='generated' for r in derived),3)
            self.assertEqual(sum(r['status']=='generated' for r in b.read_csv(gen/'attempts_all.csv')),6)
            src2.write_text('tampered')
            with self.assertRaises(ValueError):
                b.verify_manifest(root/'manifest.json')


if __name__ == '__main__':
    unittest.main()
