"""Snapshot entropy source and commands before fitting the tiny AR proposal."""
import argparse
import json
from pathlib import Path
import shutil
from cathode_cvae.data import sha256


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--project-source',type=Path,required=True)
    p.add_argument('--root',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args();root=a.root.resolve();source=a.project_source.resolve();out=a.output_dir.resolve()
    if out.exists():raise FileExistsError(out)
    roles=root/'entropy_ar_roles/receipt.json';catalog=root/'entropy_input/orbit_representatives.npz';parent=root/'entropy_input/parent.npz'
    if not roles.is_file() or json.loads(roles.read_text())['status']!='FROZEN_BEFORE_ENERGY_IMPORT':raise ValueError('Unfrozen roles')
    out.mkdir(parents=True)
    shutil.copytree(source/'src/cathode_cvae',out/'src/cathode_cvae',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    shutil.copytree(source/'scripts/entropy_20261112',out/'scripts/entropy_20261112',ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    files={str(p.relative_to(out)):sha256(p) for p in sorted(out.rglob('*')) if p.is_file()}
    protocol=dict(status='FROZEN_BEFORE_TINY_AR_FITTING',root=str(root),execution_project=str(out),
        catalog_sha256=sha256(catalog),parent_sha256=sha256(parent),role_receipt_sha256=sha256(roles),
        role_sha256=sha256(root/'entropy_ar_roles/orbit_roles.npz'),source_files=files,
        seeds=[2026111301,2026111302,2026111303],model='ar_baseline',condition_mode='composition',
        hidden_dim=32,orientation='augment',graph_context='radial_shells; cutoff5Å; shell tolerance1e-5Å',
        training_weight='Orbit multiplicity normalized by trainer mean TRAIN weight; no class balancing',
        max_updates=4000,epochs=5000,minimum_epochs=5000,patience=5000,batch_size=64,
        learning_rate=.001,primary_checkpoint='last_update.pt',sensitivity_checkpoint='best_validation_beta_0p1.pt',
        thermodynamic_temperatures_K=[100,200,300,500,800,1200,2000,5000,10000],
        wang_landau_steps=200000,wang_landau_bins=[16,32],wang_landau_seeds=[2026111201,2026111202,2026111203],
        importance_draw_budgets=[128,512,2048,8192],uniform_support_mixture=.1,
        prior_scope='Learned count-conditioned structural proposal, not trained Boltzmann probabilities',
        assessment='Exact N12 fixed-lattice total-energy reference; TEST remains absent from training labels; no N48 transfer claim')
    (out/'protocol.json').write_text(json.dumps(protocol,indent=2)+'\n')
    print(json.dumps({'status':protocol['status'],'execution':str(out),'files':len(files)},indent=2))


if __name__=='__main__':main()
