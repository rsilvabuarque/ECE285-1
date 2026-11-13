# Pristine-referenced energies and within-composition ordering

**November 4 update:** The completed [three-model benchmark](multimodel_20261104/REPORT.md) evaluates corrected UMA 1.2.1/omat, MACE-MH-1/omat_pbe, and corrected M3GNet on every historical and unique Hu geometry. Use that report for the current comparison. The analysis below preserves the original November 2 UMA 1.2 results and their provenance.

Prepared November 2, 2026. This analysis concerns archived DFT-labeled atomic frames and frozen UMA inference, not new DFT, relaxation, MD or training. It does not alter the CVAE work split or scientific scope.

## Measured answer

**The differences are not a single constant offset, and the rank orders are not universally preserved.** UMA remains a useful approximate selector in much of the historical pool, but the Hu optimized subsets provide direct counterexamples to using its low-energy class as an interchangeable DFT label.

| Comparison | Informative phase/composition groups | Exact order matches | Median Spearman rho | Median bottom-20% precision |
|---|---:|---:|---:|---:|
| Original screenshot energies, 524 frames | 7 | 0 | 0.95645 | 0.91667 |
| New UMA versus Hu, 10,932 unique geometries | 27 | 0 | 0.85344 | 0.77011 |

Medians above weight groups equally, not individual frames. They do not replace the composition/cohort tables or quantify independent-parent uncertainty. Historical ranks use the original cached energies; fresh E/F/stress predictions reproduce the old energies to maximum 2.40e−5 eV/cell and old RMS force-vector errors to maximum 4.24e−7 eV/Å. Thus the new component-level diagnostics are consistent with the screenshot baseline to the measured numerical precision.

Historical low-energy selection is substantially better than complete rank agreement:

| Li remaining (%) | Frames | Spearman rho | UMA low-quintile precision | DFT regret of UMA minimum (meV/atom) |
|---:|---:|---:|---:|---:|
| 75.00 | 10 | 0.91515 | 1.000 | 0 |
| 79.17 | 22 | 0.93111 | 0.800 | 0.02669 |
| 83.33 | 44 | 0.94616 | 0.889 | 2.36397 |
| 87.50 | 55 | 0.95808 | 0.818 | 0 |
| 91.67 | 118 | 0.95645 | 0.917 | 0 |
| 95.83 | 132 | 0.97024 | 0.963 | 0 |
| 100.00 | 143 | 0.99967 | 1.000 | 0 |

These values support a bounded approximate-ranking role in that sampled pool. They do not establish a universal tolerance for CVAE label noise or a global ground-state guarantee.

### A counterexample relevant to low-energy configurations

The Hu O3 archive has five optimized Li10Co12O24 structures (x=0.83333). Their subset rank correlation is −0.1, with five inversions in ten pairs and zero bottom-quintile overlap. In particular, UMA selects the structure with the **highest DFT energy among these five**:

| Hu ID | DFT rank | UMA rank | DFT energy above DFT minimum (meV/atom) | UMA energy above UMA minimum (meV/atom) |
|---|---:|---:|---:|---:|
| structure_4158 | 1 | 2 | 0 | 2.96171 |
| structure_4157 | 2 | 3 | 1.74348 | 4.24500 |
| structure_4156 | 3 | 5 | 1.76737 | 8.17900 |
| structure_4160 | 4 | 4 | 2.23972 | 8.13009 |
| structure_4159 | 5 | 1 | 7.97106 | 0 |

The source is `P/for_chat_gpt/LiCoO2_All_data/Data_Set/1_O3/1_Opt_Structures/Li10Co12O24/set.000/`. DFT selects frame 2, energy −263.0699462890625 eV; UMA selects frame 3, whose DFT energy is −262.7032775878906 eV. Regret is 0.366668701171875 eV/cell = 7.9710587211 meV/atom. Source labels, geometry hashes and prediction-cache joins were independently checked on all five frames. Since composition and atom count are identical, neither a common offset nor any per-species reference correction can repair this ordering. This is a finite-pool counterexample, not an estimate of its prevalence in unseen configurations.

Other low-energy cohorts also differ: optimized O1 x=2/12 has n=13, rho=−0.632 and zero low-quintile overlap; optimized O3 x=6/12 has n=19 and rho=0.933, yet only 50% low-quintile precision. Conversely, O3 x=9/12 has n=10, rho=0.960 and full low-quintile overlap. Some nearly degenerate groups span less than 0.002 meV/atom in stored DFT energies; do not interpret their numerical rank differences as physically resolved ordering.

### Offset decomposition and derivatives

For the 524-frame pool, a fitted constant total shift leaves 2.227 meV/atom residual RMSE; a fitted CoO2-host-plus-Li reference leaves 1.601 meV/atom. For the 107 optimized O3 Hu frames, the corresponding residuals are 55.16 and 7.26 meV/atom. The large improvement from the composition term demonstrates why a pristine shift and a composition-dependent reference are different hypotheses. These are descriptive fits using all inspected data, not validated corrections.

Absolute errors remain after pristine subtraction. In the historical replay, force component RMSE is 0.0242 eV/Å at x=1 and 0.1148 eV/Å at x=0.75; stress component RMSE is about 1.82 and 1.93 GPa, respectively. The absolute force/stress panels prevent good-looking differential responses from being mistaken for absolute accuracy.

The cached EFS fine-tuned test comparison improves exact ordering from two to four of its five composition groups, while one four-frame group remains imperfect. Both baseline and fine-tuned models select the DFT minimum in each of these tiny test pools. That 26-frame test omits x=0.75 and 0.79167; it cannot establish full-range rank reliability. EF and EFS cached-model results are retained in `results/cached_finetuned_test_rank_metrics.csv`.

## Questions and estimands

For exactly the same atomic frame i, define d_i = E_UMA,i − E_DFT,i in eV/cell. The proposed constant-offset explanation is d_i ≈ a. It differs from a constant per-atom error d_i ≈ b N_i and from a species-reference difference. At fixed Co/O inventory the latter has form d_i ≈ a + b n_Li,i. The exported offset-model table fits these as **all-data descriptive decompositions**, not held-out predictive corrections.

With a pristine reference p, the difference between the two pristine-referenced energies is

\[
(E^U_i-E^U_p)-(E^D_i-E^D_p)=d_i-d_p.
\]

It removes a pristine offset but does not remove a Li-dependent reference difference. Comparisons across lithiation use energy per fixed CoO2 formula unit, E/n_Co; a changing per-atom denominator can otherwise obscure which offset hypothesis is tested. These are endpoint differences, not balanced vacancy-formation or reaction energies: no removed-Li reservoir is specified.

Within one composition, any composition-only reference cancels exactly:

\[
(E^U_i-E^U_j)-(E^D_i-E^D_j)=d_i-d_j.
\]

Therefore the ranking test is more directly relevant to composition-conditioned CVAE labeling than the pristine-reference test. Report average ranks for exact ties, Spearman rho, Kendall tau-b, strict pair inversion fraction, bottom-quintile precision/recall/Jaccard, and DFT selection regret. Bottom-quintile sets include every tied boundary candidate. Regret averages the DFT penalty over all UMA-tied minima, corresponding to uniform selection among tied minima. Single-frame and fully tied groups are not evidence of correct ordering.

No coordinate-independent energy offset can change forces, F = −∂E/∂R, or fixed-composition stress. Force and stress errors are consequently reported in absolute form as well as differential scalar responses. Force RMS means sqrt(mean over the 3N Cartesian force components); its change from pristine is **not a force-vector subtraction** between different atoms. Pressure is −tr(stress)/3. It supplies a rotationally invariant stress-derived reference comparison across structural families; the lower stress panel retains absolute six-component stress RMSE.

## Data actually used

Cathode root P:

```text
/global/cfs/cdirs/m5056/rsb/2026/cathode_relithiation/cathode_relithiation_project
```

The screenshot comes from:

```text
P/results/publication/default_systems/electrode/LCO_mp-22526/
  structure_generation_tiny/vasp_workflow/analysis_completed_only/
  uma_vs_vasp_completed_cases.csv
```

This has 524 rows, Li18–24Co24O48, baseline UMA-s-1p2 / omat / CUDA. Its plotted bars are frame standard deviations, not confidence intervals. The older unfiltered 768-row CSV is not the screenshot's dataset. Exact matched DFT E/F/stress labels come from `dataset_completed_only_strict_v3_efs_same_cases/extxyz/raw/` in the corresponding UMA fine-tuning workflow. All 524 raw energies match the source CSV within 1e−7 eV; all have stress. Original VASP SCF convergence was not recertified for every archived label.

The original cached energy rankings are preserved as such. Fresh pinned UMA predictions on the same 524 geometries recover the raw F/stress arrays absent from the old scalar-summary cache. The replay table quantifies fresh-minus-cached energy differences rather than assuming retrospective checkpoint identity. Historical pristine subtraction uses separate pristine ensemble means at each recorded T/P. The reference table reports the actual DFT and UMA pristine metrics.

Additional Hu data:

```text
P/for_chat_gpt/LiCoO2_All_data/Data_Set/
P/for_chat_gpt/Final_Dataset/dpgen_work/Total_Data_Set/
```

These contain 14,944 stored frames: 6,122 bulk and 8,822 migration frames. There are 10,932 exact ordered geometries: all 6,122 bulk frames plus 4,810 migration frames. The 4,012 redundant migration entries have identical E/F/virial labels. Raw frame mappings are retained; main metrics use one representative per exact geometry so duplicate storage does not multiply evidence. This does not remove temporal correlation or establish independent vacancy parents.

Bulk families are O3 (Co12O24), spinel (Co16O32) and O1 (Co12O24). The migration archive is Li0Co12O24 only. Actual type maps retain Li/Co/O indices even when Li count is zero. Labels are archival DFT training labels by training-file/publication provenance, rather than predictions from Hu's fitted potential. Original frame-level VASP outputs, magnetic records, PAW hashes, smearing/energy-label convention and exporter version are unavailable here.

Hu's supporting information reports VASP 5.4.4, PBE+U (Co U=5.14 eV), ENCUT 650 eV, k-spacing 0.25 Å−1 and EDIFF 1e−6 eV, with a low-spin labeling protocol. It does not certify every retained frame from its raw arrays, nor imply relaxed nuclei. The historical tiny-cell labels instead use Co U=3.32 eV / 680 eV / Gamma 5×5×1 in the inspected inputs. These references must remain separate. See [Hu et al., JPCL 14, 3677–3684 (2023), DOI 10.1021/acs.jpclett.3c00506](https://doi.org/10.1021/acs.jpclett.3c00506), SI pp. S3–S6; the local SI is `P/for_chat_gpt/jz3c00506_si_001.pdf`.

Only O3 supplies x=1. The actual optimized Li12Co12O24 frame is the Hu pristine reference; its archived DFT energy is −276.2315979003906 eV/cell. Other families use this explicitly named **common O3 endpoint** for the scalar reference plots, not an invented phase-specific pristine state. Their raw/within-phase ranking errors remain separately stratified. The source DPGEN versus optimized-structure cohorts also have separate ranking tables.

Hu virials are interpreted using the standard DeePMD convention: nine row-major virial entries in eV; ASE tensile-positive stress = −sym(virial)/volume in eV/Å³, then xx, yy, zz, yz, xz, xy. Volumes are evaluated for each frame. This is a documented format convention, not verification of the archive's original exporter. Sources: [DeePMD data specification](https://docs.deepmodeling.com/projects/deepmd/en/stable/data/system.html), [official dpdata ASE conversion](https://github.com/deepmodeling/dpdata/blob/master/dpdata/plugins/ase.py).

## Figure and table package

Four main figures are saved as PNG and PDF under `results/`:

1. [Historical reference figure, PDF](results/historical_pristine_referenced.pdf) / [PNG](results/historical_pristine_referenced.png): ΔE, scalar Δforce RMS, Δpressure, plus referenced energy residual and absolute force/stress errors by lithiation. Raw pristine metrics are printed and exported.
2. [Historical rank figure, PDF](results/historical_energy_rankings.pdf) / [PNG](results/historical_energy_rankings.png): original screenshot-source DFT versus cached UMA ranks at each lithiation stage.
3. [Hu reference figure, PDF](results/hu_pristine_referenced.pdf) / [PNG](results/hu_pristine_referenced.png): the same diagnostics with new UMA predictions and the explicit common O3 endpoint.
4. [Hu rank figure, PDF](results/hu_energy_rankings.pdf) / [PNG](results/hu_energy_rankings.png): DFT versus new UMA ranks within structural family and composition.

`structure_N` IDs are deterministic within each dataset. The ranked CSVs provide every structure's source path/frame, energies and both ranks; figure labels abbreviate selected IDs as sN. Complete order lists are preferable to thousands of overlapping text labels. Separate tables retain DPGEN/optimized cohorts and, for the historical data, T/P-stratified rankings. Cached EF/EFS fine-tuned models are assessed only on their actual 26-frame test tables; those do not cover the two lowest historical lithiation fractions.

Data links: [all historical cached ranks](results/historical_cached_ranked_structures.csv), [all Hu ranks](results/hu_ranked_structures.csv), [five-structure counterexample](results/hu_optimized_Li10_counterexample.csv), [historical pristine reference values](results/historical_pristine_references.csv), [Hu pristine reference values](results/hu_pristine_references.csv). Figure correlations are rounded to three decimals; a displayed rho=1.000 is not a claim that every unrounded rank matches.

The model receipt records checkpoint hash, omat task, installed versions and explicit FP32/general inference settings with TF32 and compilation disabled. No checkpoint is updated. The inference cache preserves raw force/stress predictions and reuses only identical geometries under the same scorer signature.

## Decision limits

The seven original mean total-energy errors range from −0.28313 eV/cell at pristine to −0.89779 eV/cell at 75% lithiation. Their difference is approximately −0.61466 eV/cell, or −25.61 meV/CoO2. Thus a single total-energy offset does not account for that composition trend. Whether a composition-dependent bias coexists with useful ordering is a separate empirical result in the rank tables.

The inspected objects are archived atomic frames. Thermal/displacement energies can dominate their rank correlations. Even excellent frame rankings would not alone certify ordering of vacancy masks reconstructed on the CVAE's common pristine framework. Conversely, within-composition rank inversions cannot be repaired by pristine or species-reference offsets. No statistical p-values or atomwise bootstrap confidence intervals are used for these correlated, incompletely traced archives.

A CVAE can be trained with MLIP-derived labels without DFT labels as a mathematical matter. Claiming DFT-relevant low-energy targeting requires evidence of low-tail overlap and low DFT regret on the relevant generated object. Force/stress validation is an additional requirement for subsequent MLIP dynamics; it is not supplied by energy-rank agreement. The optional MLIP-label-versus-DFT-label training comparison must use identical structures, splits, conditioning and budgets to isolate label fidelity.

## Execution

Execution used m5397_g, one shared interactive GPU at a time. Job 60717166 stopped before inference on configuration serialization; job 60717329 completed inference, analysis and estimator checks. The timed pilot forecast 988 seconds for inference; measured prediction/collection time was about 911 seconds. It evaluated 11,456 distinct geometries: 524 historical plus 10,932 Hu, with 4,012 duplicate lookups. Job 60718420 regenerated the two reference figures solely to move their colorbars outside data axes; it made no inference calls. All allocations are terminal and released.

Total reservation time, including startup failures, idle time, analysis and rendering, was 1,617 seconds = 0.44917 GPU-device-hours. At one of four GPUs in shared interactive QOS, the calculated charge is **0.11229 GPU node-hours**, not 0.44917 full-node hours. The allocation-level `sacct -X` records and calculation are in `results/resources.json`, following [NERSC's shared-resource charge policy](https://docs.nersc.gov/jobs/policy/). Unrelated allocations are excluded.

Checks passed for exact label/source joins, duplicate identity, all seven original rank groups against an independent quadratic pair-count/rank-correlation implementation, and small independent examples covering identity, complete reversal, ties and tied-minimum regret. No DFT convergence or global magnetic-state certification is inferred from these code checks. Executed source snapshots are preserved under `results/executed_sources_60717329/`; model and inference signatures are in `results/model.json`.

The current scripts reproduce the workflow from the same archived inputs: `predict.py` evaluates or resumes exact-geometry cache entries; `analyze.py` regenerates statistics, independent checks and figures; `check_analysis.py` checks tie/regret examples. `run_gpu.sh` composes inference and analysis inside a compute allocation. Its default outputs are `results/`. `render_figures.py` only redraws the two reference figures from saved metrics. No installation, dataset download or checkpoint update is required for the recorded environment.
