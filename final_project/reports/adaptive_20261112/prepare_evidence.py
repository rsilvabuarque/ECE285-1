"""Copy an explicit compact evidence allowlist; preserve raw runs unchanged."""
import argparse
import hashlib
import json
import shutil
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run-root', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    files = []
    for phase in ('latent_diagnostics', 'latent_phase2', 'latent_mean_supervision'):
        files += [f'{phase}/analysis_oracle/{name}' for name in (
            'oracle_yield_seed_balanced.csv', 'oracle_yield_mean_SD.csv',
            'paired_primary_effects.csv', 'guided_descriptor_effects_seed_balanced.csv',
            'checkpoint_information_per_fit.csv', 'VAL_probes_per_fit.csv',
            'optimization_curves.csv', 'receipt.json')]
    files += ['latent_mean_supervision/' + name for name in (
        'LATENT_RESULT_PACKET.md', 'mean_family_five_fit_summary.csv',
        'posterior_information_bound.json', 'posterior_information_bound_validation.json',
        'posterior_information_bound_highprecision_validation.json', 'latent_cost_receipt.json')]
    files += ['latent_mean_supervision/conditional_information_audit/' + name for name in (
        'conditional_probes_cells.csv', 'conditional_probes_seed_balanced.csv',
        'fixed_condition_random_comparison.csv', 'receipt.json')]
    for folder, names in {
        'iteration/study_v2/analysis': ['paired_fit_summary.csv', 'paired_fit_results.csv', 'training_bin_support.csv', 'fit_weight_concentration.csv', 'corrected_aggregate_metrics.csv', 'metric_semantics_repair.json', 'receipt.json', 'interpretation.md'],
        'iteration/study_v2/interpretation_diagnostics': ['range_R2_decomposition_summary.csv', 'fresh_uniform_mutation_baselines.csv', 'range_R2_per_cell_summary.csv', 'range_TRAIN_dynamics.csv', 'R1_range_minus_fixed_same_data.csv', 'receipt.json', 'interpretation.md'],
        'iteration/study_v2/acquisition_coverage': ['summary.csv', 'by_fit_count.csv', 'UMA_own_moving_target_fidelity.csv', 'receipt.json', 'interpretation.md'],
        'iteration/causal_range': ['paired_summary.csv', 'paired_results.csv', 'aggregate_metrics.csv', 'plan.json', 'protocol.json', 'completion.json'],
        'iteration/frontier': ['completion.json', 'protocol.json', 'first_step_paired_control_check.json'],
        'iteration/frontier/presentation_v2': ['final_summary.csv', 'captions.md', 'receipt.json'],
        'entropy_review': ['RESULTS.md', 'terminal_AR_uniform_comparison.csv', 'terminal_AR_target_mismatch.csv', 'symmetry_energy_temperature_scale.csv', 'receipt.json'],
        'entropy_independent_audit': ['exact_binned_WL_comparison.csv', 'REFERENCES.md', 'receipt.json'],
        'entropy_boltzmann_1000K/summary': ['RESULTS.md', 'matched_budget_comparison.csv', 'physical_target_role_mass.csv', 'receipt.json'],
        'integration_verification': ['receipt.json'],
        'integration_verification/causal_review': ['audit.json', 'AUDIT.md', 'independent_paired_summary.csv', 'audit.py'],
        'integration_verification/fit_role_audit': ['receipt_final.json', 'AUDIT_COMPLETION.json', 'fit_metadata_rows_final.csv', 'criterion_erratum_final.json', 'append_certificate_rows.json', 'boundary_role_rows.json']
    }.items():
        files += [folder + '/' + name for name in names]
    files += ['iteration/RESULT_AND_COST_RECEIPT.json', 'iteration/final_protected_role_audit.json', 'entropy_pipeline_receipt.json', 'entropy_boltzmann_1000K/pipeline_receipt.json', 'entropy_boltzmann_1000K/independent_audit/receipt.json']
    inventory = {}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for rel in files:
        source = args.run_root / rel
        raw = source.read_bytes()
        if len(raw) > 5_000_000:
            raise ValueError(f'Compact-file limit exceeded: {source}')
        target = args.output_dir / rel
        if target.exists() and target.read_bytes() != raw:
            raise ValueError(f'Preserve differing existing artifact: {target}')
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        inventory[rel] = {'source': str(source), 'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    manifest = {'scope': 'Aggregate evidence and receipts only. Full attempts, identities, checkpoints and raw MLIP outputs remain in the HPC run archive. This compact packet replays report tables, not complete training or inference.', 'files': inventory}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'files': len(inventory), 'bytes': sum(x['bytes'] for x in inventory.values())}))


if __name__ == '__main__':
    main()
