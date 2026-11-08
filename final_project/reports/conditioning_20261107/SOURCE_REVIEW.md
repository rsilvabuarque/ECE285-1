# Baseline and discussion review

The new goal starts from commit `73e6cee5fa43a26a1ac9e8700f81baaa3e40619f`.
The independent experiment record is the integrating owner's
`experiment_ledger.json` beside this file; the scientific contract is
`artifacts/conditioning_20261107/protocol.json`. This review is a source map,
not a competing status ledger.

The updated 89-page conversation PDF motivates controlled conditioning,
physical-neighbor and latent-size tests, a paired energy-target audit, a
structural descriptor, and later seed editing/defined-hop NEB. Its suggestions
are not evidence that those experiments ran. A separately named
`Pasted text(1).txt` was not present in the designated attachments directory at
preflight. Prior Codex history and actual source/receipts provide execution
context; no missing transcript is represented as inspected.

| Baseline claim checked | Authoritative record | Finding at starting commit |
| --- | --- | --- |
| Three condition families | `src/cathode_cvae/conditions.py`; `learning/models.py` | Composition-only, binary low/rest, and multibin conditions already exist; the AR decoder receives context at each site. |
| Hard composition | `learning/models.py:ARCVAE.sample` | Count-preserving sequential generation already exists; energy membership remains statistical. |
| Balancing, validation-beta, warm start, update cap, orientation augmentation | `learning/train_mask.py` | Existing flags should be reused rather than reimplemented. |
| Latent-size CLI | `learning/train_mask.py` at baseline | Not yet exposed; model constructors already accept dimensions. New plumbing and strict reload checks are required. |
| Physical graph / ParentSpec | Baseline `src/cathode_cvae/` and manifest | No trained graph model or reusable periodic parent schema at baseline. A diagram is not a measured ablation. |
| Frozen mutation | `scripts/overnight_20261107/generate_benchmark.py` | Fixed seed pool, uniformly chosen one/two/three distinct removal-addition swaps, no energy acceptance or online population update. |
| Earlier feedback | `artifacts/current/improvement_v1/adapt_round1.py`; final receipt | One coupled adaptation changed data, initialization, class weighting and beta. Three November7 fit seeds are not three feedback rounds. |
| Historical versus new results | `docs/RESULTS.md`; both campaigns' CSVs | Historical binary UMA yield35.57%; new multibin balanced31.00%, requested-low original-q20 32.99%. These are different scopes. |
| Existing relaxation corpus | `artifacts/current/core/data/geometry_uma_fixed.npz`; executed exporter | 1000 independent starting masks, 6000 frames; fixed-cell deletion/intermediate/endpoint rows. A fresh audit is required before using endpoint labels. |
| Existing geometry CVAE | `docs/RESULTS.md`; frozen decoded geometry CSVs | Separate coordinate predictor; optimizer savings were also attained by regression. No demonstrated CVAE-specific geometry advantage. |
| Temperature layer | `acquisition.py`; allocation receipts | Outer finite-pool static acquisition, not a decoder temperature or physical entropy measurement. |
| Existing Li hops | `reports/thermal_pilot_20261107/numerics/md_audit_events.json` | Two observed events are candidates for a later bounded NEB test, not already calculated barriers. |

The current user's explicit instructions replace the attachment's old resource
defaults: 12 elapsed hours, up to36 simultaneous GPUs, and a new250 GPU-node-hour
ceiling on `m5397_g`, with no fixed cap on UMA/MACE static labels. The bounded
training design, protected roles, no-new-DFT restriction, no-new-MLIP-fine-tuning
restriction, required-report-first gate, and preservation requirements remain.
The integrating cost receipt reports actual allocated time including startup,
idle allocation, and failures; no budget statement is inferred from old receipts.
