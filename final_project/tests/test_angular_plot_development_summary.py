"""Small, explicitly synthetic tests; no prior assessment or GPU work."""
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path

PROJECT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(PROJECT/'scripts/angular_20261110'),str(PROJECT/'src')]
spec=importlib.util.spec_from_file_location('plot_dev',PROJECT/'scripts/angular_20261110/plot_development_summary.py')
a=importlib.util.module_from_spec(spec);spec.loader.exec_module(a)


def synthetic():
    seeds=('1','2','3');y=[];paired=[];masks={};latent={}
    methods={x for m,r,_ in a.CONTRASTS for x in (m,r)}|{f'radial_cvae_d{d}' for d in a.DIMENSIONS}
    for s in seeds:
        for t in a.TEACHERS:
            for method in methods:
                y.append(dict(teacher=t,method=method,replicate_id=s,macro_novel_inbin_yield='0.5',score_coverage='1'))
            for method,reference,_ in a.CONTRASTS:
                paired.append(dict(teacher=t,method=method,reference=reference,replicate_id=s,difference_pp='0'))
        masks[s]=[]
        for d in a.DIMENSIONS:
            latent[s,d]=dict(mean_kl=.005,active_fraction=0,active_001=0,active_01=0,active_1=0,epoch=1)
            for n in a.COUNTS:
                masks[s].append(dict(method=f'radial_cvae_d{d}',replicate_id=s,n_Li=str(n),
                    contrast='alternate_latent_minus_low',valid_pairs='8',attempted_pairs='8',valid_coverage='1',same_orbit_fraction='0.25'))
    return seeds,paired,y,masks,latent


class FigureTest(unittest.TestCase):
    def test_matched_points_and_plot(self):
        points=a.extract_data(*synthetic())
        self.assertEqual(len(points),141)
        self.assertEqual({r['value'] for r in points if r['panel']=='latent_resampling'},{25.})
        with tempfile.TemporaryDirectory() as td:
            a.plot(points,Path(td))
            self.assertTrue((Path(td)/'development_summary.pdf').is_file())
            self.assertTrue((Path(td)/'development_summary.png').is_file())

    def test_rejects_mismatched_difference_and_missing_seed(self):
        seeds,paired,y,masks,latent=synthetic();paired[0]['difference_pp']='1'
        with self.assertRaises(ValueError):a.extract_data(seeds,paired,y,masks,latent)
        with self.assertRaises(ValueError):a.mean_sd([1,2])

    def test_rejects_invalid_mask_coverage(self):
        seeds,paired,y,masks,latent=synthetic();masks['1'][0]['valid_pairs']='0'
        with self.assertRaises(ValueError):a.extract_data(seeds,paired,y,masks,latent)

    def test_synthetic_receipts_match_selected_checkpoint_sidecars(self):
        from development_boundary import binding,save_csv,save_json
        seeds,paired,y,masks,_=synthetic()
        with tempfile.TemporaryDirectory() as td:
            root=Path(td);agg=root/'aggregate';agg.mkdir()
            save_csv(agg/'paired_control_differences.csv',paired);save_csv(agg/'per_seed_yield.csv',y)
            shards=[];mask_receipts=[]
            for seed in seeds:
                sub=root/f'rep{seed}';sub.mkdir();models=[]
                for d in a.DIMENSIONS:
                    fit=sub/f'd{d}';fit.mkdir();checkpoint=fit/'selected.pt';checkpoint.write_text(f'synthetic-{seed}-{d}')
                    cfg=fit/'configuration.json';save_json(cfg,dict(seed=int(seed),latent_dim=d,val_rows=10))
                    models.append(dict(method=f'radial_cvae_d{d}',replicate_id=seed,checkpoint=str(checkpoint),
                        expected_checkpoint_sha256=binding(checkpoint)['sha256'],configuration=binding(cfg),declared_latent_dim=d))
                    pooled=dict(rows=10,latent_dim=d,KL_nats_by_dimension=[.005]*d,posterior_mean_variance_by_dimension=[.0001]*d,
                        mean_KL_nats_per_dimension=.005,total_KL_nats=.005*d,
                        active_components_by_cutoff={'0.001':0,'0.01':0,'0.1':0})
                    save_json(Path(str(checkpoint)+'.latent.json'),dict(epoch=3,pooled=pooled))
                manifest=sub/'manifest.json';save_json(manifest,dict(status='FROZEN_DEVELOPMENT_BEFORE_GENERATION',models=models))
                shards.append(dict(replicate_id=seed,manifest=binding(manifest)))
                ms=sub/'mask_summary.csv';save_csv(ms,masks[seed])
                receipt=sub/'mask_receipt.json';save_json(receipt,dict(status='PASS',scope='all originally valid masks',
                    manifest=binding(manifest),outputs={'mask_summary.csv':binding(ms)['sha256']}));mask_receipts.append(receipt)
            agg_receipt=agg/'receipt.json';save_json(agg_receipt,dict(status='PASS_DEVELOPMENT_AGGREGATION',replicates=seeds,shards=shards,
                outputs={name:binding(agg/name)['sha256'] for name in ['paired_control_differences.csv','per_seed_yield.csv']}))
            args=a.validate_inputs(agg_receipt,mask_receipts)
            self.assertEqual(len(args[-1]),15)
            self.assertEqual(len(a.extract_data(*args[:5])),141)
            # A selected checkpoint cannot be silently changed after manifest freeze.
            Path(args[-1][0]['checkpoint']['path']).write_text('tampered checkpoint')
            with self.assertRaises(ValueError):a.validate_inputs(agg_receipt,mask_receipts)


if __name__=='__main__':unittest.main()
