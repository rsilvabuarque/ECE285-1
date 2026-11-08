"""Freeze the next campaign contract before generating/scoring development pools.

This preparation records actual input hashes and separates old, development and
new final evidence. It never changes historical artifacts or submits jobs.
"""
import argparse
from datetime import datetime, timedelta, timezone
import hashlib
import json
from pathlib import Path
import shutil


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for b in iter(lambda: f.read(8 * 1024**2), b''):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--previous', type=Path, required=True)
    p.add_argument('--goal-created-unix', type=int, required=True)
    a = p.parse_args()
    root = a.root.resolve()
    out = root / 'artifacts/conditioning_20261107'
    reports = root / 'reports/conditioning_20261107'
    run = root / 'runs/conditioning_20261107'
    for d in (out, reports, run / 'logs', run / 'executed_sources'):
        d.mkdir(parents=True, exist_ok=True)
    protocol = out / 'protocol.json'
    if protocol.exists():
        raise FileExistsError('Do not overwrite a frozen campaign protocol')
    start = datetime.fromtimestamp(a.goal_created_unix, timezone.utc)
    source = Path('/global/cfs/cdirs/m5397/rsb/2026/classes/ECE285/for_chat_gpt')
    parents = root / 'artifacts/current/core/data/parents/O3_N48_pilot.npz'
    data = root / 'artifacts/current/improvement_v1/training_round0.npz'
    hypotheses = [
        ('policy', 'Class balancing and checkpoint selection affect conditional control',
         'balancing off/on; two selection weights reuse each trajectory', 6),
        ('latent', 'A larger latent improves useful prior generation only if actually used',
         'latent 4/8/16/32, reuse selected-policy 8; hidden64 unchanged', 9),
        ('no_latent', 'The latent contributes beyond stochastic AR decoding',
         'AR baseline, no latent and no KL', 3),
        ('symmetry', 'Training orientation affects generalization',
         'canonical versus joint parent-group augmentation', 3),
        ('graph_encoder', 'Physical neighbors improve posterior feature extraction',
         'distance message-passing encoder; otherwise matched policy', 3),
        ('graph_context', 'Causal physical neighbor features help generation',
         'one bounded generation-context pilot, no future occupations', 1),
        ('descriptor', 'Supported joint structural/energy requests can be controlled',
         'one supported energy+vacancy-pair descriptor pilot', 1),
        ('edit', 'A seed-conditioned variational editor improves equal-budget swaps',
         'same seed families, exact one/two/three swaps versus mutation', 3),
    ]
    contract = dict(
        schema_version=1, status='FROZEN_BEFORE_NEW_PROPOSALS_AND_SCORING',
        base_commit='73e6cee5fa43a26a1ac9e8700f81baaa3e40619f',
        branch='midterm/conditioning-graph-edit-20261107',
        start_UTC=start.isoformat(), hard_stop_UTC=(start+timedelta(hours=12)).isoformat(),
        finalization_reserve_minutes=45,
        user_resource_override=dict(account='m5397_g', maximum_concurrent_GPUs=36,
            maximum_new_GPU_node_hours=250, maximum_elapsed_hours=12,
            static_candidate_cap=None, old_prompt_accounting_limits_superseded=True),
        nonresource_science_bounds=dict(maximum_full_training_trajectories=30,
            planned_full_training_trajectories=29, optional_NEB_maximum_events=3,
            new_DFT=False, new_MLIP_finetuning=False),
        data=dict(parent_sha256=digest(parents), round0_sha256=digest(data),
            protected_split_counts=dict(train=40157,validation=4967,test=5123),
            original_role_preservation=True, counts=[12,24,36], occupancy_one='present',
            canonical_string_one='vacancy',
            geometry_protocol='pristine-derived deletion; unrelaxed; exact canonical parent coordinates'),
        sources={n: digest(source/n) for n in [
            'ECE285_next_goal_prompt_20261107.txt','Evaluate Generative Model Project.pdf']},
        teacher=dict(primary='UMA-S-1.2.1', head='omat', precision='float32', tf32=False,
            audit='MACE-MH-1/omat_pbe, prespecified final matched subset'),
        fit=dict(seeds=[20261106,20261107,20261108], max_updates=4000,
            batch_size=512, learning_rate=.001, training_beta_final=.1,
            beta_warmup_epochs=20, validation_selection_betas=[1.,.1],
            hidden_dim=64, maximum_seconds_each=1200,
            note='Graph pilot time caps recorded; no criterion relaxed for success'),
        development=dict(attempts_per_count_bin_fit=64,
            paired_conditions=True, common_latents=True, common_AR_uniforms=True,
            sampling_namespace='conditioning-development-20261107-v1',
            permitted_for_selection=True, protected_test_used_for_selection=False),
        final=dict(attempts_per_count_bin_fit=128,
            sampling_namespace='conditioning-final-20261107-v1',
            history_frozen_after_development_before_any_final_generation=True,
            final_method_list_frozen_after_development_before_generation=True,
            no_final_labels_reused_for_training_or_selection=True),
        controls=dict(mutation='uniform training seed in requested bin; uniform1/2/3 distinct occupied/vacant swaps; frozen before scoring',
            uniform='raw fixed-count masks, all attempts counted',
            editor='paired identical seed keys, edit counts and proposal budgets; preserve joint orientation'),
        outcomes=['exact-count validity','requested-bin confusion','novel in-bin orbits/all attempts',
            'original-fixed-q20 low requests','energy distributions/budget curves',
            'novelty to train and history separately','structural diversity/edit distances',
            'posterior/prior and latent/condition sensitivity','measured allocated cost'],
        stages=[dict(id=i,hypothesis=h,changed_factor=c,full_trajectories=n) for i,h,c,n in hypotheses],
        gates=dict(optional_extensions='Required experiments, audits and compiled supported midterm packet first',
            NEB='only after required packet and editor comparison; bounded event/force/cost feasibility'),
    )
    protocol.write_text(json.dumps(contract,indent=2)+'\n')
    ledger = dict(schema_version=1, protocol_sha256=digest(protocol),
        requirements={i:dict(status='PLANNED',hypothesis=h,changed_factor=c,
            outputs=[],full_training_trajectories=n) for i,h,c,n in hypotheses})
    for k in ['cached_reproduction','paired_relaxation_audit','descriptor_audit',
              'parent_graph_checks','required_report','final_assessment','NEB','publication']:
        ledger['requirements'][k]=dict(status='PLANNED',outputs=[])
    (reports/'experiment_ledger.json').write_text(json.dumps(ledger,indent=2)+'\n')
    shutil.copy2(source/'ECE285_next_goal_prompt_20261107.txt',run/'instructions.txt')
    (run/'execution.json').write_text(json.dumps(dict(status='PREPARING',
        protocol=str(protocol), owned_jobs=[60985189], instructions_sha256=contract['sources'],
        parent=str(parents),data=str(data)),indent=2)+'\n')
    print(json.dumps(dict(protocol=str(protocol),sha256=digest(protocol),
        ledger=str(reports/'experiment_ledger.json'),run=str(run)),indent=2))


if __name__ == '__main__':
    main()
