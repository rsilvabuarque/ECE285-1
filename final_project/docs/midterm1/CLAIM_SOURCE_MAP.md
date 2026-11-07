# Report claim and source map

The [machine-readable map](claim_sources.json) stores exact source hashes and verification scope. Status: **PASS — measured report ready**. This does not mark Beihao’s or Varad’s assigned independent work complete.

| Claim family | Evidence / status | Main source |
| --- | --- | --- |
| parent48 | Li48Co48O96, Li rows48..95, 192 retained site permutations, fixed cell — VERIFIED | [O3_N48_pilot.npz](../../artifacts/current/core/data/parents/O3_N48_pilot.npz) |
| half orbit capacity | At nLi24: 32247603683100 raw masks and168025313938 orbits under this parent action — VERIFIED | [make_examples.py](../../reports/ranking_audit_20261107/make_examples.py) |
| hu12 support | 87 representatives across allcounts; n6 has17orbits covering924rawmasks; group144 — VERIFIED | [finite_support.json](../../reports/ranking_audit_20261107/hu_finite_support/finite_support.json) |
| training round0 | 50247 rows; train40157/val4967/test5123 — VERIFIED | [preparation.json](../../artifacts/current/improvement_v1/data/preparation.json) |
| historical yields | Old UMA/MACE AR35.57/40.06%; uniform18.99/19.75%; strongestmutation52.65/54.08%; adaptive33.81/38.91% — VERIFIED_HISTORICAL | [summary.csv](../../artifacts/current/improvement_v1/reports/final/summary.csv) |
| ranking context | 34368 matched rows/11456geometries; cohort-specific rho/topk/regret/gap metrics — VERIFIED | [receipt.json](../../reports/ranking_audit_20261107/receipt.json) |
| hu five counterexample | CompleteHuoptimizedO3Li10poolN5; UMAstructure4159rank1versusDFTrank5; regret7.9710587meV/atom — VERIFIED | [Hu_O3_Li10_optimized_counterexample.csv](../../reports/ranking_audit_20261107/Hu_O3_Li10_optimized_counterexample.csv) |
| representative examples | Two existing unrelaxedLi24training deletion patterns, not newCVAE samples — VERIFIED_HISTORICAL | [manifest.json](../../reports/ranking_audit_20261107/examples/manifest.json) |
| geometry secondary | Old32maskUMAmedianoptimizersteps40deletion,24regressor,24CVAE; noCVAE-specificadvantage — VERIFIED_HISTORICAL | [decoded_matched_relaxations.csv](../../artifacts/current/reports/decoded_matched_relaxations.csv) |
| energy units example | 168atomcell1meV/atom=.168eV; at433.15K ratioexp(-.168/kBT)approximately.0111 — EXACT_UNIT_CONVERSION_AND_CALCULATION | [acquisition.py](../../src/cathode_cvae/acquisition.py) |
| reactor temperature grid | 393.15,433.15,473.15,493.15K — VERIFIED_EXISTING_TARGETS | Existing reactor input manifests; exact paths/hashes in JSON |
| independent reference acquisition | All 24,575 exact geometries from 24,576 uniform raw draws scored; concentrated finite-pool static weights and prefix sensitivity — VERIFIED_NEW_MEASUREMENT | [receipt.json](../../reports/overnight_20261107/reference_diagnostics/receipt.json) |
| new multibin benchmark | Balanced 41,472-attempt benchmark, 40,662 unique geometries; complete current UMA/MACE scoring; three-bin yield does not beat controls — VERIFIED_NEW_MEASUREMENT | [receipt.json](../../artifacts/overnight_20261107/evaluation/benchmark/receipt.json) |
| new acquisition demonstration | 90 executed schedules; 11,520 attempts/11,080 geometries; requested and measured energy bins differ substantially — VERIFIED_NEW_MEASUREMENT | [generation_receipt.json](../../artifacts/overnight_20261107/demo/generation_receipt.json) |
| new fit receipts | Three completed fits: epochs 53/44/49, updates 4187/3476/3871, wall seconds 30.806/27.761/27.707; selection by validation objective — VERIFIED_NEW_MEASUREMENT | [summary.json](../../artifacts/overnight_20261107/fits/multibin_v2_20261106/summary.json) |
| new cost receipt | 127,481 new logical EFS calls including 24 pilot calls; 12.401111 GPU-device hours / 3.100278 four-GPU-equivalent node-hours; both allocations completed and released — VERIFIED_NEW_MEASUREMENT | [cost_receipt.json](../../artifacts/overnight_20261107/cost_receipt.json) |

The report has five body pages and one reference page. All pages were rendered and reviewed; the final cost paragraph was reviewed in its final page4 layout. No overfull boxes or undefined references remain.

Primary materials references were checked to the scope stated in JSON: RSC publisher abstract/metadata; the author-hosted full Van der Ven paper; and OMat24 v2 §2.1.2. The supplied screenshots require revised proposal, detailed methods, and discussion in approximately five body pages; NeurIPS is explicitly named for the proposal only.

No new DFT, physical entropy, equilibrium population, global minimum or downstream MLIP-improvement claim is implied.
