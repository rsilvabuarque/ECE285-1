# What changed in this goal

The starting point was commit `d0d93edeefc56003744599a83dc54c45f78577aa`.
The previous report, releases, protected masks and original TRAIN/VAL/TEST roles
were preserved. Updated conversation suggestions were treated as hypotheses;
completed earlier experiments were not automatically repeated.

| Suggestion or unresolved question | Verified implementation and outcome |
|---|---|
| Separate spatial context from a useful variational latent | Fresh matched graph-posterior CVAE, no-latent AR, dense-posterior CVAE, and fixed nonphysical association controls. Common first-three-fit contrasts and the selected five-fit comparison are in `final_static/`. Useful latent benefit remains unresolved. |
| Improve the causal decoder with limited, interpretable changes | Five single-factor screens, independent confirmation of three arms, and frozen FINAL assessment. Radial-shell context gives a modest gain; remaining-count and beta changes do not provide a general rescue. All screens remain in `screens/` and `confirmation/`. |
| Check low-energy structures after relaxation | A separately sampled 6,000-mask cohort, roles assigned before labels, fixed-cell UMA relaxation, and 0.01 eV/Å development controls. Eight interrupted acquisitions were not silently retried or discarded from the record. |
| Train on deletion versus relaxed energies | Three genuine CVAE fits per target with identical initial masks, roles, architecture and update budget. Both generators were evaluated under both label definitions. Relaxed-energy labels did not improve the bounded endpoint comparison. These are still mask generators followed by an external relaxation. |
| Check the current sampler's relaxed LOW advantage | Frozen 2,304-attempt four-method comparison. All 7,616 unique physical queries in the full 10,944-attempt endpoint assessment converged. Raw denominators, shared uniforms and per-fit contrasts are preserved. |
| Add an independent teacher | Every prescribed static candidate was measured with UMA and MACE using separate frozen thresholds. At selected UMA endpoints, MACE energies/forces and independently relaxed MACE controls show teacher and stationarity disagreement. Agreement is not DFT validation. |
| Evaluate diversity and symmetry | Exact orbit novelty uses the full protected TRAIN/history identity sets and all 192 retained N48 operations. Nearest-reference distances use a fixed 1,024-TRAIN-orbit subset per count; they are upper bounds on full-TRAIN nearest distance, not certified distances to every training example. |
| Verify exact small-cell statistics | All 1,364 literal Hu-parent raw masks at counts 3/6/9 were scored under both teachers, with numerical repeats. Approximate parent symmetry was not used to alias energies. `smallcell/` and `smallcell_independent_v2/` distinguish finite static-model entropy from real-material thermodynamics. |
| Resolve saved NEB route ambiguity | Bounded common-endpoint image/tolerance tests, one candidate-saddle curvature check at two displacements, downhill connections and independently optimized MACE/UMA bands. Event 1 remains unconverged; event 2 fails the combined protocol-sensitivity target and is not a validated elementary diffusion barrier. |
| Test thermal-estimator bias cheaply | Twenty-seven synthetic harmonic analyses from the actual saved 537-mode spectrum. Spurious fluidicity and finite-window bias remain visible; no new MD or anharmonic free-energy claim was made. |
| Repeat adaptive training or the learned editor until it wins | Not done. The earlier same-seed, same-edit-budget learned-editor comparison is already documented in the prior packet. New adaptive acquisition, continuous conditioning, unseen-size neural transfer and further editor variants were deferred. |

The compact receipts, source/claim maps, exact command manifests and the
downloadable raw evidence identify which observations are numerical, which are
teacher-defined, and which require later physical validation. The new static
and paired FINAL assessments were frozen before their labels were read, and
their outcomes were not used to tune the models.

Remaining physical work includes compatible DFT checks of selected unrelaxed
and relaxed geometries, electronic/magnetic-state consistency, stable path and
basin identification before interpreting migration kinetics, and a validated
same-potential thermodynamic reference before claiming anharmonic free energy.
No DFT, new MD or MLIP fine-tuning was launched or scheduled in this goal.

Beihao's, Varad's and Ricardo's independent review tasks are in
`docs/COLLABORATOR_TASKS.md`. Automated fits and table checks are evidence for
that review; they are not completed independent student contributions.
