# Thermal-pilot execution record

The midterm gate was completed and pushed before this extension started. The
scientific design is `docs/thermal_pilot/DESIGN.md`; final run status belongs to
the run and analysis receipts. A command being listed here does not imply that
every scientific adequacy check passed.

Root requested allocation **60956668** using the verified current account/QoS:

```bash
salloc -A m5397_g -C gpu -q interactive -N 2 -t 03:30:00 \
  --job-name=ece285_thermal_pilot bash
```

Eight GPUs were reserved. Single-GPU tasks use `srun --exact`,
`--gpus-per-task=1 --gpu-bind=single:1`, and bounded CPU threads; CPU analysis
uses `--overlap --gres=none`. These cluster options are execution provenance,
not requirements of the portable class package.

The executed preparation/physical sequence is:

1. `prepare_thermal.py`: select three existing training masks and write exact
   deletion POSCARs. Two pre-inference adapter failures are retained in
   `selection_attempt0/` and `selection_attempt1/`; no labels were discarded.
2. `thermal_relax_launcher.py`: three concurrent fixed-cell UMA relaxations.
3. `thermal_task_launcher.py --tasks hessian_tasks.json`: eight GPU shards,
   covering three complete 0.01-Å Hessians and basin1's 0.02-Å sensitivity.
4. CPU `assemble_tasks.json`: four assemblies. The first attempt failed before
   mathematics because nested CLI paths were not JSON-serializable. The repair
   changes metadata serialization only; raw forces were reused unchanged.
5. `thermal_teacher_check.py`: three MACE single points on exact UMA minima,
   for disagreement/DFT prioritization, without MACE relaxation or dynamics.
6. `md_pilot_tasks.json` and `md_pilot_basin2_tasks.json`: three 0.75-ps
   startup/stability checks, explicitly not target-temperature thermodynamics.
7. `md_production_tasks.json`: four 20-ps NVE production trajectories after
   5-ps Bussi warmup; separate matched-seed 5-ps NVT and half-timestep controls.
   The primary pair changed to basins1/2 after independent harmonic agreement,
   before production. The old prepared-only pair1/3 task file is retained.

The JSON task files contain exact input paths, seeds, durations and outputs;
each physical receipt additionally stores its command arguments, parent/input
hashes, executing source hash, checkpoint/head, numerical settings and hardware.
Executed source snapshots and logs remain under
`runs/thermal_pilot_20261107/`. Large native results are on scratch and are
archived on CFS only after all writers stop; the public raw manifest gives the
durable archive path, checksum and member hashes.

Posthoc commands and their actual completion/diagnostic status are recorded in
the independent MD audit, pinned ATLAS analysis and transport-sensitivity
receipts. They preserve sampled-temperature, finite-window and estimator limits.
None of these scripts runs, submits or schedules DFT.
