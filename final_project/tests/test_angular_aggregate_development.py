"""Small synthetic estimand tests; no project labels or GPU calls."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]/'scripts/angular_20261110'
sys.path.insert(0,str(ROOT))
spec=importlib.util.spec_from_file_location('agg',ROOT/'aggregate_development.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def design():
    return {'teachers':{'uma':'hash'},'methods':['a','b'],'counts':[12,24],
        'requested_bins':['low','high'],'budget':2,'diagnostics':False,
        'latent_dims':{'a':0,'b':0},'contrasts':[{'method':'a','reference':'b'}]}


def metric(rep,method,n,bin_,good):
    return dict(teacher='uma',method=method,replicate_id=rep,n_Li=str(n),requested_bin=bin_,attempts='2',
        scored_attempts='2',distinct_novel_inbin_orbits=str(good),novel_inbin_yield_per_attempt=str(good/2),
        fixed_q20_available='True',novel_fixed_q20_yield_per_attempt='0.5')


def pairs(rep, missing=False):
    rows=[]
    for method in ['a','b']:
        for n in [12,24]:
            for i in range(2):
                known=not (missing and method=='a' and n==12 and i==0)
                rows.append(dict(teacher='uma',method=method,replicate_id=rep,n_Li=str(n),paired_id=str(i),
                    contrast='condition_high_minus_low',both_energies_known=str(known),
                    energy_shift_meV_atom=('2' if known else ''),AR_uniforms_equal='True',actual_z_equal='True'))
    return rows


class AggregateTest(unittest.TestCase):
    def test_macro_and_paired_control_are_replicate_based(self):
        d=design();allrows=[];allcells={}
        for rep in ['1','2']:
            metrics=[metric(rep,m,n,b,2 if m=='a' else 1) for m in d['methods'] for n in d['counts'] for b in d['requested_bins']]
            result,cells=a.macro_yields(metrics,d,rep);allrows.extend(result);allcells.update(cells)
        paired,aggregate=a.contrasts_and_summary(allrows,d)
        self.assertEqual([r['difference_pp'] for r in paired],[50.,50.])
        self.assertEqual(aggregate[0]['mean_difference_pp'],50.)
        self.assertEqual(aggregate[0]['descriptive_SD_pp'],0.)
        self.assertEqual(len(a.cell_contrasts(allcells,d,['1','2'])),8)
        with self.assertRaises(ValueError):a.macro_yields(metrics[:-1],d,'2')
        with self.assertRaises(ValueError):a.macro_yields(metrics+[metrics[0]],d,'2')

    def test_missing_pairs_are_not_filled_with_zero(self):
        d=design();rows=[]
        for rep in ['1','2']:rows+=a.sensitivity(pairs(rep,missing=rep=='1'),rep,d)
        fits,summary=a.sensitivity_summary(rows,d,['1','2'])
        target=next(r for r in fits if r['method']=='a' and r['replicate_id']=='1')
        self.assertEqual(target['pair_coverage'],0.75)
        self.assertFalse(target['all_pairs_complete'])
        self.assertEqual(target['macro_signed_meV_atom'],2.)
        result=next(r for r in summary if r['method']=='a')
        self.assertEqual(result['mean_macro_signed_meV_atom'],2.)
        self.assertEqual(result['mean_pair_coverage'],0.875)
        broken=pairs('1')[:-1]
        with self.assertRaises(ValueError):a.sensitivity(broken,'1',d)

    def test_compact_plot_creation(self):
        d=design(); seeds=[];paired=[]
        for rep in ['1','2']:
            metrics=[metric(rep,m,n,b,2 if m=='a' else 1) for m in d['methods'] for n in d['counts'] for b in d['requested_bins']]
            result,_=a.macro_yields(metrics,d,rep);seeds.extend(result)
        _,paired=a.contrasts_and_summary(seeds,d)
        with tempfile.TemporaryDirectory() as td:
            a.plot_results(Path(td),seeds,paired,[],d)
            self.assertTrue((Path(td)/'macro_yield.pdf').is_file())
            self.assertTrue((Path(td)/'paired_control_differences.pdf').is_file())


class ReceiptChainTest(unittest.TestCase):
    def test_two_synthetic_shards_and_tamper_detection(self):
        import json
        import numpy as np
        b=a.__dict__['binding']
        from development_boundary import save_json,save_csv
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);bd=root/'boundary';bd.mkdir()
            f=root/'input';f.write_text('synthetic input')
            save_json(bd/'known_history.json',[]);save_json(bd/'protected.json',[])
            boundary=dict(status='FROZEN_DEVELOPMENT_KEY_ONLY_BOUNDARY',history=b(bd/'known_history.json'),
                protected=b(bd/'protected.json'),parent=b(f),data=b(f),request_schema=b(f),original_history=b(f),
                protected_orbits=0,known_history_orbits=0)
            save_json(bd/'receipt.json',boundary)
            source=root/'evaluate_controls.py';source.write_text('synthetic evaluator')
            oldreceipt=root/'oldreceipt.json';save_json(oldreceipt,{'status':'synthetic'})
            registry=root/'registry.json';collector=root/'collected';collector.mkdir()
            sid='mask48_000000000000'
            save_json(registry,dict(parent_sha256=b(f)['sha256'],physical_keys={
                'unique-key':dict(model='uma',structure_id=sid,provenance='new_scoring')},tasks=[dict(model='uma')]))
            np.savez(collector/'uma.npz',structure_ids=np.array([sid]),status=np.array(['success']),provenance=np.array(['new']))
            save_json(collector/'receipt.json',dict(status='PASS',registry_sha256=b(registry)['sha256']))
            paths=[]
            for rep in ['1','2']:
                shard=root/f'rep{rep}';shard.mkdir();gen=shard/'gen';gen.mkdir();ev=shard/'ev';ev.mkdir()
                m=dict(status='FROZEN_DEVELOPMENT_BEFORE_GENERATION',namespace='synthetic',counts=[12,24],
                    requested_bins=['low','high'],count_per_group=2,diagnostics=False,latent_stream_width=4,
                    contrasts=[{'method':'a','reference':'b'}],
                    models=[dict(method=x,replicate_id=rep,sampler='uniform') for x in ['a','b']],
                    inputs={'parent':b(f),'data':b(f),'schema':b(f),'history':b(bd/'known_history.json')},
                    teacher_schemas={'uma':b(f)},fixed_thresholds=b(f),boundary_receipt=b(bd/'receipt.json'),
                    plan=b(f),source_bindings=[b(source)])
                save_json(shard/'manifest.json',m)
                save_json(gen/'protocol.json',dict(manifest=b(shard/'manifest.json')))
                attempts=[]
                for method in ['a','b']:
                    for n in [12,24]:
                        for name in ['low','high']:
                            for i in range(2):
                                j=len(attempts)
                                attempts.append(dict(attempt_id=f'{rep}-{j}',status='generated',canonical_mask='0'*48,
                                    scoring_structure_id=sid,method=method,replicate_id=rep,n_target=str(n),requested_bin=name,
                                    diagnostic='primary',raw_row=str(j)))
                save_csv(gen/'attempts_all.csv',attempts)
                np.savez(gen/'candidates.npz',structure_ids=np.array([sid]))
                save_json(gen/'receipt.json',dict(status='FROZEN_BEFORE_SCORING',protocol_sha256=b(gen/'protocol.json')['sha256'],
                    manifest=b(shard/'manifest.json'),boundary_receipt=b(bd/'receipt.json'),original_generation_receipt=b(oldreceipt),
                    outputs={k:b(gen/k)['sha256'] for k in ['attempts_all.csv','candidates.npz']}))
                outcomes=[dict(row,teacher='uma') for row in attempts];save_csv(ev/'attempt_outcomes.csv',outcomes)
                metrics=[metric(rep,method,n,bin_,1) for method in ['a','b'] for n in [12,24] for bin_ in ['low','high']]
                save_csv(ev/'metrics.csv',metrics)
                save_csv(ev/'seed_metrics.csv',[dict(teacher='uma',method=x,replicate_id=rep,
                    macro_composition_bin_yield=0.5,fixed_q20_LOW_request_yield=0.5) for x in ['a','b']])
                save_csv(ev/'paired_attempts.csv',pairs(rep))
                evnames=['attempt_outcomes.csv','metrics.csv','seed_metrics.csv','paired_attempts.csv']
                save_json(ev/'receipt.json',dict(status='PASS',terminal_failed_candidate_labels={'uma':0},
                    proposal_protocol_sha256=b(gen/'protocol.json')['sha256'],source_sha256=b(source)['sha256'],
                    outputs={k:b(ev/k)['sha256'] for k in evnames},score_sources={'uma':dict(
                        sha256=b(collector/'uma.npz')['sha256'],schema_sha256=b(f)['sha256'])},
                    fixed_thresholds=dict(sha256=b(f)['sha256'],source='main_train')))
                paths.append((shard/'manifest.json',gen,ev))
            config,parsed,_,receipt=a.validate_shards(paths,collector)
            costs=a.validate_registry(registry,collector,receipt,parsed,config['teachers'])
            self.assertEqual(costs[0]['unique_physical_queries'],1)
            self.assertEqual(costs[0]['new_queries'],1)
            rows=a.read_csv(paths[0][2]/'attempt_outcomes.csv');rows[0]['status']='protected_boundary'
            (paths[0][2]/'attempt_outcomes.csv').unlink();save_csv(paths[0][2]/'attempt_outcomes.csv',rows)
            with self.assertRaises(ValueError):a.validate_shards(paths,collector)
            # Even an internally consistent receipt cannot turn a mismatched
            # evaluation into evidence from the filtered candidate pool.
            rp=paths[0][2]/'receipt.json';r=json.loads(rp.read_text())
            r['outputs']['attempt_outcomes.csv']=b(paths[0][2]/'attempt_outcomes.csv')['sha256']
            rp.unlink();save_json(rp,r)
            with self.assertRaisesRegex(ValueError,'Evaluation changed filtered attempt'):
                a.validate_shards(paths,collector)


if __name__=='__main__':unittest.main()
