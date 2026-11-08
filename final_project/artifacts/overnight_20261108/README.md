# Frozen November 8 evidence packet

Use the project code at the GitHub release tag `ece285-overnight-20261108-v1`.
Install the core learning environment with its README procedure. Then, from
`final_project/`, run:

```bash
python scripts/overnight_20261108/replay_packet.py --packet artifacts/overnight_20261108 --output-dir runs/varad_replay --plots
```

This checks every manifest SHA and replays eight cached assessment scopes. It
makes no MLIP calls, fits no models and generates no masks. Full trajectories
are optional raw companion assets; they are not required for this replay.

`data/H0.npz` retains the original protected TRAIN/VAL/TEST roles. All37 fitted
models used named validation-beta0.1 payloads, recorded in `fits/` and the
execution task manifests. Do not select default `best.pt` instead.

`paired/training/` contains the4794 usable TRAIN/VAL mask-label pairs and
separate deletion/endpoint schemas. `paired/compact_geometry/` contains the
6000 acquisition rows as separate TRAINVAL4800 and TEST1200 coordinate views.
Interrupted records remain flagged, including a missing endpoint; use the
usability/status fields rather than treating every row as converged. TEST and
all FINAL assessment candidates remain assessment data, not new training data.
These coordinate views do not imply that a geometry CVAE was fitted here.

The main PDF has five body pages plus references; the separate eight-page
research supplement preserves negative path/saddle and thermal diagnostics.
Physical validation still requires later DFT and appropriate thermodynamic and
path convergence checks. No new DFT, MD or MLIP fine-tuning occurred.

This payload is frozen before publication. Final download verification,
resource closure and the current collaborator instructions are in the tagged
Git repository and the release's completion/publication receipts. Private
account balances, conversation PDFs and foundation-model weights are excluded.
