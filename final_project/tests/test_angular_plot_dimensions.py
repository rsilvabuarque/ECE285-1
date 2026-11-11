"""Synthetic tests for the fresh angular dimension plots; no GPU jobs."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'scripts/angular_20261110'),str(PROJECT/'src')]
spec=importlib.util.spec_from_file_location('angular_dim',PROJECT/'scripts/angular_20261110/plot_angular_dimensions.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def synthetic():
    seeds=('1','2','3');y=[];masks={};latent={}
    for s in seeds:
        masks[s]=[]
        for family,_ in a.FAMILIES:
            for d in a.DIMENSIONS:
                method=a.method_for(family,d)
                for teacher in ('uma','mace'):
                    y.append(dict(teacher=teacher,method=method,replicate_id=s,fit_seed=s,
                                  macro_novel_inbin_yield='.5',score_coverage='1'))
                for n in a.COUNTS:
                    masks[s].append(dict(method=method,replicate_id=s,n_Li=str(n),contrast='alternate_latent_minus_low',
                        valid_pairs='8',attempted_pairs='8',valid_coverage='1',same_orbit_fraction='.25'))
                latent[family,d,s]=dict(mean_KL=.005,total_KL=.005*d,active_001=0,active_01=0,active_1=0,epoch=3)
    return seeds,y,masks,latent


class AngularDimensionPlotTest(unittest.TestCase):
    def test_three_families_five_sizes_three_fits_and_plot(self):
        seeds,y,masks,latent=synthetic();p=a.extract(seeds,y,masks,latent);s=a.summarize(p)
        self.assertEqual(len(p),45);self.assertEqual(len(s),15)
        self.assertEqual(a.method_for('angle_hist_cvae',4),'angle_hist_cvae')
        self.assertEqual(a.method_for('angle_hist_cvae',8),'angle_hist_cvae_d8')
        self.assertEqual({r['same_orbit_percent'] for r in p},{25.})
        with tempfile.TemporaryDirectory() as td:
            a.plot(p,Path(td));self.assertTrue((Path(td)/'angular_dimension_sweeps.pdf').is_file())
            self.assertTrue((Path(td)/'angular_dimension_sweeps.png').is_file())

    def test_unmatched_seed_and_missing_coverage_are_errors(self):
        seeds,y,masks,latent=synthetic();y[0]['fit_seed']='2'
        with self.assertRaises(ValueError):a.extract(seeds,y,masks,latent)
        seeds,y,masks,latent=synthetic();masks['1'][0]['valid_pairs']='0'
        with self.assertRaises(ValueError):a.extract(seeds,y,masks,latent)

    def test_fresh_sidecars_receipts_and_fit_root(self):
        from development_boundary import binding,save_csv,save_json
        seeds,y,masks,_=synthetic()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);agg=root/'aggregate';agg.mkdir();fitroot=root/'fresh_fits';fitroot.mkdir()
            save_csv(agg/'per_seed_yield.csv',y);shards=[];receipts=[]
            for seed in seeds:
                sub=root/f'rep{seed}';sub.mkdir();models=[]
                for family,_ in a.FAMILIES:
                    for d in a.DIMENSIONS:
                        method=a.method_for(family,d);fit=fitroot/f'{method}_{seed}';fit.mkdir()
                        ck=fit/'selected.pt';ck.write_text(f'synthetic-{method}-{seed}')
                        cfg=fit/'configuration.json';save_json(cfg,dict(seed=int(seed),latent_dim=d,val_rows=10))
                        models.append(dict(method=method,replicate_id=seed,checkpoint=str(ck),
                            expected_checkpoint_sha256=binding(ck)['sha256'],configuration=binding(cfg),declared_latent_dim=d))
                        pooled=dict(rows=10,latent_dim=d,KL_nats_by_dimension=[.005]*d,posterior_mean_variance_by_dimension=[.0001]*d,
                            mean_KL_nats_per_dimension=.005,total_KL_nats=.005*d,
                            active_components_by_cutoff={'0.001':0,'0.01':0,'0.1':0})
                        save_json(Path(str(ck)+'.latent.json'),dict(epoch=3,pooled=pooled))
                manifest=sub/'manifest.json';save_json(manifest,dict(status='FROZEN_DEVELOPMENT_BEFORE_GENERATION',models=models))
                shards.append(dict(replicate_id=seed,manifest=binding(manifest)))
                ms=sub/'mask_summary.csv';save_csv(ms,masks[seed]);rec=sub/'mask_receipt.json'
                save_json(rec,dict(status='PASS',scope='all originally valid masks',manifest=binding(manifest),
                                   outputs={'mask_summary.csv':binding(ms)['sha256']}));receipts.append(rec)
            ar=agg/'receipt.json';save_json(ar,dict(status='PASS_DEVELOPMENT_AGGREGATION',replicates=seeds,shards=shards,
                outputs={'per_seed_yield.csv':binding(agg/'per_seed_yield.csv')['sha256']}))
            validated=a.load_inputs(ar,receipts,fitroot)
            self.assertEqual(len(validated[-1]),45)
            self.assertEqual(len(a.extract(*validated[:4])),45)
            with self.assertRaisesRegex(ValueError,'outside fresh fit root'):a.load_inputs(ar,receipts,root/'other')


if __name__=='__main__':unittest.main()
