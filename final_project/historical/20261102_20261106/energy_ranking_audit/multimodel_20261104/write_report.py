"""Write the human-readable result from saved, completed comparison tables."""
from pathlib import Path
import csv
import json
import math

HERE=Path(__file__).resolve().parent
OUT=HERE/'comparison'
NAMES={'uma':'UMA 1.2.1 / omat','mace':'MACE-MH-1 / omat_pbe','m3gnet':'M3GNet / MatPES-PBE'}


def table(headers,rows):
    return '\n'.join(['| '+' | '.join(headers)+' |','| '+' | '.join(['---']*len(headers))+' |']
                     +['| '+' | '.join(map(str,row))+' |' for row in rows])


def main():
    summary=list(csv.DictReader((OUT/'model_summary.csv').open()))
    ranks=list(csv.DictReader((OUT/'ranking_metrics.csv').open()))
    lines=['# Three frozen MLIPs versus historical and Hu DFT references','',
           'Completed exact-geometry single-point comparisons. No relaxation, new DFT, MD, or training is included.',
           '', '[Main comparison figure](comparison/model_comparison.pdf) · [Side-by-side structure ranks](comparison/side_by_side_structure_ranks.csv) · [All ranking metrics](comparison/ranking_metrics.csv)',
           '', '## Which energy rankings agree most?', '',
           'The rows below use unweighted medians across phase/composition groups within each DFT source. Precision measures the fraction of the MLIP low-energy quintile also in the DFT low-energy quintile. Exact order agreement is stricter than selecting useful low-energy candidates. Mean regret is the DFT penalty of selecting a model minimum in the tested pool, in meV/atom.', '']
    ranking=[];errors=[]
    for dataset in ['historical','hu']:
        for row in [r for r in summary if r['dataset']==dataset and r['phase']=='ALL']:
            ranking.append([dataset,NAMES[row['model']],row['n_rank_groups'],f"{float(row['median_spearman']):.5f}",
                            f"{100*float(row['median_low20_precision']):.1f}%",f"{float(row['mean_regret_meV_atom']):.4f}",
                            row['exact_order_groups']])
            errors.append([dataset,NAMES[row['model']],f"{float(row['energy_MAE_meV_atom']):.3f}",
                           f"{float(row['median_pair_energy_RMSE_meV_atom']):.3f}",f"{float(row['force_RMSE_eV_A']):.5f}",
                           f"{float(row['stress_RMSE_GPa']):.5f}"])
        rows=[r for r in summary if r['dataset']==dataset and r['phase']=='ALL']
        best=max(float(r['median_spearman']) for r in rows)
        names=[NAMES[r['model']] for r in rows if math.isclose(float(r['median_spearman']),best,abs_tol=1e-12,rel_tol=0)]
        lines.append(f"In **{dataset}**, the largest median Spearman correlation is **{', '.join(names)}**, {best:.5f}.")
    lines+=['',table(['Source','Model','Groups','Median ρ','Median low-20% precision','Mean regret (meV/atom)','Exact-order groups'],ranking),'',
            'These source-wide summaries do not replace the phase and optimized-cohort tables. Hu has many correlated migration frames; phase-specific comparisons and the [optimized-only comparison](comparison/hu_optimized_ranking_comparison.pdf) are needed before choosing a labeling model.',
            '', '## Energy, force, and stress accuracy', '',
            table(['Source','Model','Native energy MAE (meV/atom)','Median pair-energy RMSE (meV/atom)','Force RMSE (eV/Å)','Stress RMSE (GPa)'],errors),'',
            'Native energy errors retain all reference offsets. Pair-energy RMSE compares differences between structures of the same source, phase, and composition; a common energy shift cancels. Force RMSE weights all Cartesian atomic components; stress RMSE weights the six ASE components equally. Per-frame and Li/Co/O force errors are retained in [per_structure_EFS.csv](comparison/per_structure_EFS.csv).',
            '', '## Dataset and model provenance', '',
            '- **524 historical geometries** and **10,932 unique Hu geometries** were evaluated by each model: **34,368 production model–geometry evaluations**. All 15,468 original entries are retained in `raw_frame_map.csv`; 4,012 redundant Hu geometries share predictions. Exact duplicates are accepted only when their DFT energy, force, and stress labels also match.',
            '- **UMA:** patched `uma-s-1p2p1`, `omat`, FAIRChem 2.25.0, FP64/general inference, TF32/Nitro/compilation/dispersion disabled. [Official changelog](https://github.com/facebookresearch/fairchem/blob/main/docs/core/uma_changelog.md). The newer r²SCAN fine-tune is a different reference and was not substituted for the corrected base model.',
            '- **MACE:** `MACE-MH-1`, explicitly asserted `omat_pbe` head, FP64, MACE 0.3.16, no dispersion or CuEq compilation. [Official model card](https://huggingface.co/mace-foundations/mace-mh-1).',
            '- **M3GNet:** corrected `M3GNet-PES-MatPES-PBE-2025.2`, MatGL 4.1.4, FP64. The checkpoint and code include the corrected three-body graph mapping. [Official checkpoint card](https://huggingface.co/materialyze/M3GNet-PES-MatPES-PBE-2025.2).',
            '- All processes used isolated Torch 2.13/CUDA 13 environments on A100 GPUs. Model hashes, software/settings, GPU assignments, and source hashes are recorded in `results/*/shard_*/model.json` and copied to `comparison/*_receipts.json`. Original shared environments were preserved.',
            '- DFT reference energies are used exactly as archived. The historical and Hu DFT protocols differ; their absolute energies are not merged into one reference. Hu SI reports its spin protocol, but original frame-level SCF/spin records and exporter metadata are unavailable. The audit does not recertify every archived electronic calculation.',
            '- Hu virials use the conventional DeepMD conversion `stress = -sym(virial)/volume`. Every calculator returns ASE tensile-positive `xx yy zz yz xz xy` stress in eV/Å³, converted to GPa for reporting. Atomic positions and species order are preserved exactly.',
            '', '## Statistical definitions and checks', '',
            'Ranks use average ranks for exact ties. The low-energy set includes the lowest `ceil(0.2*n)` entries and all ties at its boundary. Selection regret averages the DFT penalty across any exactly tied model minima. Strict pair inversions exclude pairs tied in either score. Single-frame/fully tied groups are not treated as evidence of correct ordering.',
            '', 'For errors δᵢ = E_model,i − E_DFT,i within a fixed-composition pool of n equal-size frames, the all-pair energy RMSE is sqrt(2n/(n−1)) times the population standard deviation of δ, divided by atom count. The code checks this identity and all historical inversion/rank calculations against direct quadratic calculations for all three models.',
            '', 'Each model completed an 11-frame pilot covering source, phase, optimized/displaced cohorts, and Li-free data. Two-step central energy differences checked one force component and hydrostatic stress on a hot historical and an optimized Hu geometry. These validate numerical consistency, signs, and units; they do not establish DFT accuracy.',
            '', 'The initial Torch 2.8/CUDA 12 stack failed even a standalone 3×3 CUDA determinant on this node. Torch 2.13/CUDA 13 passed the same control; MACE/MatGL were moved to private upgraded environments and their pilots then passed. Old startup failures are preserved. UMA also had a console-only NumPy-bool serialization failure after its pilot predictions/checks were saved; the receipt records that correction.',
            '', 'No framewise p-values or independent-atom confidence intervals are claimed. Exact duplicate removal does not remove trajectory/parent correlation. Very small DFT energy spans and float32 archive quantization can make detailed ranks numerically fragile; cohort tables retain spans, ties, and sample sizes. Static accuracy does not establish MD stability or a global ground state.',
            '', '## Files to use', '',
            '- [Main model comparison](comparison/model_comparison.pdf), [historical ranking comparison](comparison/historical_ranking_comparison.pdf), [Hu ranking comparison](comparison/hu_ranking_comparison.pdf), [Hu optimized-only ranking comparison](comparison/hu_optimized_ranking_comparison.pdf).',
            '- [Historical E/F/stress errors](comparison/historical_EFS_errors.pdf), [Hu E/F/stress errors](comparison/hu_EFS_errors.pdf). Each model also has source-specific `*_parity.pdf` and `*_rankings.pdf` figures, plus PNG copies.',
            '- [Side-by-side ranks for every exact geometry](comparison/side_by_side_structure_ranks.csv), [complete model-specific ordered structures](comparison/ranked_structures.csv), [model summary](comparison/model_summary.csv), [phase/composition/cohort ranks](comparison/cohort_rank_metrics.csv), [per-group winners](comparison/group_winners.csv).',
            '- [Five optimized Hu Li10 structures](comparison/Hu_Li10_optimized_counterexample.csv) gives the same structures under all three potentials.',
            '- Raw predicted force arrays, energies, and stresses are retained in `results/<model>/shard_*/chunk_*.npz`. `frames.csv` identifies every unique geometry; `raw_frame_map.csv` restores every original archive row. `reference_frames.npz` contains the exact corresponding DFT labels.',
            '', '## Execution and accounting', '',
            'Account `m5397_g`. UMA and MACE use the original four-GPU interactive allocation `60804157`. At the user’s request for maximum parallelism, M3GNet is split into 20 disjoint single-GPU batches: 16 GPUs across four nodes in `60804966`, plus two GPUs each in shared allocations `60804967` and `60804969`. Peak concurrent reservation is 24 GPUs. The final `resources.json` records released allocations and includes setup, failures, idle time, inference, and analysis in its charge calculation. GPU-device-hours and charged node-hours are distinct.', '']
    if (HERE/'resources.json').exists():
        res=json.loads((HERE/'resources.json').read_text())
        lines.append(f"Calculated charge: **{res['charged_node_hours']:.5f} GPU node-hours**; reservation: **{res['reserved_GPU_device_hours']:.5f} GPU-device-hours**. Shared allocations use half-node GPU fractions. Final posted Iris charges may differ. Allocation states: **{res['state']}**.")
    (HERE/'REPORT.md').write_text('\n'.join(lines)+'\n')
    print('Wrote',HERE/'REPORT.md')


if __name__=='__main__':main()
