The optional NEB pilot is complete. Both marked events survived endpoint
quenching and met their recorded projected-force criteria. Event1's forward
span changed from 0.638221 to 0.188711 eV per hop when image count and force
target were changed together. This large sensitivity leaves its barrier
unresolved. Event2's five-interior-image span is 0.191544 eV per hop and has no
image-count check. The seven MACE single points use the **coarse event1 UMA
band**, with no MACE optimization. See [the full interpretation](../../../docs/OPTIONAL_EXTENSIONS_20261107.md),
[saved outcomes](results/outcomes.csv), and [independent numerical audit](independent/receipt.json).

There were 3,749 UMA and seven MACE calculator E/F attempts, with no failed
calls. Job60989957 is COMPLETED; its two-GPU allocation used 0.758889
GPU-device-hours, including startup and idle time. No DFT was run or scheduled.
The required report, its assessment boundaries and release packet remain
preserved. The design and exact execution commands below are retained as
provenance; fresh reproductions require new output directories.

The executed preparation is `prepared_v3/protocol.json`; `prepared/` and
`prepared_v2/` preserve earlier drafts before allocation accounting and
interrupted-IDPP label provenance were tightened.

The two candidates come from the saved Li36Co48O96 MD trajectories. Preparation verifies the complete source archive/trajectory/mapping hashes against their original receipts, compares the selected positions with the original trajectory frames, recomputes the atom-to-site assignments, and preserves every atom's original identity and unwrapped coordinates. The before-event and settled post-event snapshots differ in the assignment of one Li atom. These are thermal snapshots to be quenched, rather than prevalidated minimum-energy endpoints. The second event briefly visits assignment site 40 before settling at site 23; that boundary/transit assignment is retained in the protocol.

NEB places intermediate configurations between two minima and optimizes a path. Its spring forces keep images distributed along the path, while physical forces perpendicular to the path move the band toward a minimum-energy route. After an initial band relaxation, the climbing image reverses its physical force along the path and removes its spring force, seeking the saddle. A converged NEB projected force therefore differs from the raw physical force on an ordinary interior image: the latter may have a substantial component along the energy slope. Both quantities are saved. The intended endpoint event must survive all-atom, fixed-cell relaxation; if both snapshots quench to the same actual site assignment, the code reports a failed candidate and leaves the barrier unknown. Equal symmetry-canonical occupation keys alone do not invalidate a diffusion event.

The [official ASE NEB documentation](https://docs.ase-lib.org/ase/neb.html) explains the APIs and the initial-band/climbing-image sequence. The current online default changed after ASE 3.26, so the code explicitly selects `improvedtangent`, serial image evaluation, and FIRE. Installed 3.26 source hashes/signatures are frozen in the protocol. The spring coefficient is 0.1 eV/Å²: the installed implementation multiplies spring length in Å by this coefficient to obtain force in eV/Å. The web parameter label omits that second inverse length; the protocol records the discrepancy.

All atoms move; the common cell and original atom order remain fixed. Endpoint FIRE uses a provisional maximum physical force of 0.03 eV/Å and at most 500 steps per endpoint. The initial five-interior-image band uses an initial projected-force target of 0.10 eV/Å for up to 300 steps, then a climbing target of 0.05 eV/Å for up to 700 steps. FIRE's maximum step is 0.05 Å. Linear interpolation preserves the observed unwrapped MD hop branch; IDPP then uses periodic interatomic distances for at most 100 steps. It never independently wraps/canonicalizes the endpoints. One successful path may be resampled to seven interior images and refined to 0.03 eV/Å for at most 400 steps. These are provisional numerical targets, not DFT-accuracy criteria. The image/tolerance check reports its barrier difference instead of redefining success.

The owner should run the two events concurrently with one distinct GPU assigned to each command, and end the entire two-GPU NEB window within 3,600 wall seconds. Each process allows at most 2,400 seconds including model loading and 10,000 actual ASE calculator calls. An optional MACE profile is bounded to 600 seconds on one GPU after the UMA paths; the other allocated GPU still counts as idle allocation time. This leaves 600 seconds of the two-GPU window for launch/cleanup reserve. The owner must use the actual allocation accounting, not sum process times and assume idle GPUs are free.

From the repository's `final_project/` directory, launch these only inside the assigned GPU processes, using the UMA backend interpreter and checkpoint recorded in the existing worker configuration. `--device cuda` uses the GPU exposed by the caller/scheduler.

```bash
python scripts/conditioning_20261107/neb_run.py \
  --prepared reports/conditioning_20261107/neb_pilot/prepared_v3 \
  --event basin_1_repeat2 --refine --budget-seconds 2400 \
  --checkpoint "$UMA_CHECKPOINT" \
  --output-dir runs/conditioning_20261107/neb/basin_1_repeat2

python scripts/conditioning_20261107/neb_run.py \
  --prepared reports/conditioning_20261107/neb_pilot/prepared_v3 \
  --event basin_2_repeat1 --budget-seconds 2400 \
  --checkpoint "$UMA_CHECKPOINT" \
  --output-dir runs/conditioning_20261107/neb/basin_2_repeat1
```

The owner should also enforce an outer wall timeout and a cleanup grace period. The script catches termination signals, saves incomplete bands and explicit failures, and attempts no new model calls during failure serialization. A timed-out refinement does not erase the saved five-interior-image result. No output directory is overwritten.

If the first UMA path converges and the allocation budget allows it, run the following with the separate MACE backend interpreter and one GPU. It evaluates the frozen five-interior-image UMA path and performs no optimization. Its endpoint-referenced energy profile is a disagreement check, never an independently optimized MACE barrier.

```bash
python scripts/conditioning_20261107/neb_mace_profile.py \
  --prepared reports/conditioning_20261107/neb_pilot/prepared_v3 \
  --uma-run runs/conditioning_20261107/neb/basin_1_repeat2 \
  --checkpoint "$MACE_CHECKPOINT" --budget-seconds 600 \
  --output-dir runs/conditioning_20261107/neb/mace_profile
```

After the jobs finish, this CPU command reads saved arrays/receipts and plots the profiles, including failures; it loads no MLIP:

```bash
python scripts/conditioning_20261107/neb_summarize.py \
  --run-root runs/conditioning_20261107/neb \
  --output-dir reports/conditioning_20261107/neb_pilot/results
```

Receipts distinguish actual calculator E/F attempts, successful/failed calls, optimizer iterations, projected NEB residuals, physical image forces, and uninstrumented neural-backbone forwards. Forward and reverse barriers are in eV per marked Li hop. These are MLIP potential-energy barriers, without finite-temperature entropy or DFT validation; two local events cannot establish a connected transport network. Endpoint/band POSCAR, CIF and NPZ exports are candidates for later independent physical validation. No VASP input decks, DFT jobs, new MD, or barrier-conditioned CVAE training are produced.
