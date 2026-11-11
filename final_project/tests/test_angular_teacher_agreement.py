"""Synthetic tie and threshold tests; no project results or GPU work."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.stats import kendalltau, spearmanr

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'scripts/angular_20261110'),str(PROJECT/'src')]
spec=importlib.util.spec_from_file_location('agree',PROJECT/'scripts/angular_20261110/teacher_agreement.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)
from cathode_cvae.conditions import fit_schema


def schema(values):
    return fit_schema(np.full(5,12),np.asarray(values,float),np.array(['train']*5),n_sites=48)


class TeacherAgreementTest(unittest.TestCase):
    def test_fixed_schema_and_empirical_bottom_are_distinct(self):
        # Synthetic training cutoffs are UMA <=1 and MACE <=0. These are not
        # fitted on the following five generated orbit energies.
        u=np.array([1,2,2,4,5.]);m=np.array([1,3,2,4,5.]);ids=[f'orbit_{i}' for i in range(5)]
        summary,conf,rows=a.analyze_count(12,ids,u,m,schema([1,2,3,4,5]),schema([0,1,2,3,4]))
        self.assertAlmostEqual(summary['spearman_average_rank'],spearmanr(u,m).statistic)
        self.assertAlmostEqual(summary['kendall_tau_b'],kendalltau(u,m,variant='b').statistic)
        self.assertEqual(summary['schema_low_UMA'],1)
        self.assertEqual(summary['schema_low_MACE'],0)
        self.assertEqual(summary['schema_low_jaccard'],0)
        self.assertEqual(summary['empirical_low_UMA'],1)
        self.assertEqual(summary['empirical_low_MACE'],1)
        self.assertEqual(summary['empirical_low_jaccard'],1)
        self.assertEqual(sum(r['orbits'] for r in conf),5)
        self.assertEqual(len(rows),5)
        with tempfile.TemporaryDirectory() as td:
            a.plot(Path(td),rows,conf,[12])
            self.assertTrue((Path(td)/'teacher_ranks.pdf').exists())
            self.assertTrue((Path(td)/'teacher_schema_confusion.pdf').exists())

    def test_cutoff_ties_are_never_split(self):
        low,cut=a.empirical_low([1,1,2,3,4])
        self.assertEqual(cut,1)
        self.assertEqual(low.tolist(),[True,True,False,False,False])
        with self.assertRaises(ValueError):a.empirical_low([float('nan')])

    def test_constant_rank_and_empty_overlap_are_explicit(self):
        s=schema([1,2,3,4,5])
        summary,_,_=a.analyze_count(12,['a','b','c'],[8,8,8],[9,9,9],s,s)
        self.assertEqual(summary['spearman_average_rank'],'')
        self.assertEqual(summary['kendall_tau_b'],'')
        self.assertEqual(summary['schema_low_jaccard'],'')
        with self.assertRaises(ValueError):a.analyze_count(12,['a','a'],[1,2],[2,1],s,s)


class TeacherReceiptTest(unittest.TestCase):
    def test_synthetic_frozen_shared_union_and_tampering(self):
        import json
        import cathode_cvae.conditions, cathode_cvae.pool_scoring
        from cathode_cvae.conditions import save_schema
        from development_boundary import binding, save_json
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bd=root/'boundary';bd.mkdir();gen=root/'gen';gen.mkdir();col=root/'collected';col.mkdir()
            f=root/'input';f.write_text('synthetic')
            parent=root/'parent.npz';np.savez(parent,numbers=np.ones(192),li_indices=np.arange(48))
            save_json(bd/'history.json',[]);save_json(bd/'protected.json',[])
            save_json(bd/'receipt.json',dict(status='FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY',
                history=binding(bd/'history.json'),protected=binding(bd/'protected.json'),
                protected_orbits=0,known_history_orbits=0))
            schemas={}
            for teacher,values in [('uma',[1,2,3,4,5]),('mace',[0,1,2,3,4])]:
                path=root/f'{teacher}_schema.json';save_schema(schema(values),path);schemas[teacher]=binding(path)
            manifest=dict(status='FROZEN_DEVELOPMENT_BEFORE_GENERATION',models=[dict(method='u',replicate_id='1',sampler='uniform')],
                counts=[12], inputs=dict(parent=binding(parent),data=binding(f),schema=binding(f),history=binding(bd/'history.json')),
                teacher_schemas=schemas,fixed_thresholds={'path':str(root/'DO_NOT_OPEN_OLD_THRESHOLD'),'sha256':'unused'},
                boundary_receipt=binding(bd/'receipt.json'),plan=binding(f),
                source_bindings=[binding(cathode_cvae.conditions.__file__),binding(cathode_cvae.pool_scoring.__file__),
                                 binding(PROJECT/'scripts/overnight_20261108/evaluate_controls.py')])
            save_json(root/'manifest.json',manifest);save_json(root/'generation_original.json',{'status':'synthetic'})
            masks=np.ones((2,48),np.uint8);masks[0,:12]=0;masks[1,12:24]=0
            ids=np.array([f'mask48_{int("".join(map(str,row)),2):012x}' for row in masks])
            np.savez(gen/'candidates.npz',structure_ids=ids,masks=masks,n_Li=np.array([12,12]))
            save_json(gen/'protocol.json',dict(manifest=binding(root/'manifest.json')))
            save_json(gen/'receipt.json',dict(status='FROZEN_BEFORE_SCORING',protocol_sha256=binding(gen/'protocol.json')['sha256'],
                manifest=binding(root/'manifest.json'),boundary_receipt=binding(bd/'receipt.json'),
                original_generation_receipt=binding(root/'generation_original.json'),
                outputs={'candidates.npz':binding(gen/'candidates.npz')['sha256']}))
            keys={}
            for teacher in ('uma','mace'):
                for sid in ids:keys[f'{teacher}_{sid}']=dict(model=teacher,structure_id=sid,provenance='new_scoring')
                np.savez(col/f'{teacher}.npz',structure_ids=ids,status=np.array(['success','success']),
                    provenance=np.array(['new','new']),energy_eV=np.array([156.,312.]),
                    n_atoms=np.array([156,156]),n_Li=np.array([12,12]))
            registry=root/'registry.json';save_json(registry,dict(parent_sha256=binding(parent)['sha256'],physical_keys=keys,
                tasks=[{'model':'uma'},{'model':'mace'}]))
            save_json(col/'receipt.json',dict(status='PASS',registry_sha256=binding(registry)['sha256']))
            evaluation=root/'evaluation_receipt.json'
            save_json(evaluation,dict(status='PASS',terminal_failed_candidate_labels={'uma':0,'mace':0},
                proposal_protocol_sha256=binding(gen/'protocol.json')['sha256'],
                source_sha256=binding(PROJECT/'scripts/overnight_20261108/evaluate_controls.py')['sha256'],
                score_sources={teacher:dict(sha256=binding(col/f'{teacher}.npz')['sha256'],schema_sha256=schemas[teacher]['sha256'])
                               for teacher in ('uma','mace')}))
            common,sources,counts,labels,loaded,_=a.validate_inputs([(root/'manifest.json',gen)],col,registry,evaluation)
            self.assertEqual(len(counts),2);self.assertEqual(set(labels),{'uma','mace'})
            self.assertEqual(common['counts'],[12]);self.assertEqual(len(sources),1)
            with (col/'uma.npz').open('ab') as stream:stream.write(b'tamper')
            with self.assertRaises(ValueError):
                a.validate_inputs([(root/'manifest.json',gen)],col,registry,evaluation)


if __name__=='__main__':unittest.main()
