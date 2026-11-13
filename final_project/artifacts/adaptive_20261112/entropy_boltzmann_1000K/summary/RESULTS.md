# Fixed1000K Boltzmann-target follow-up

Six composition-only AR fits change only within-composition TRAIN weights to g exp[−(E−EminTRAIN)/(kB×1000K)]. Each composition preserves its original TRAIN sum(g); VAL/TEST weights remain g and TEST energies remain NaN. E is total eV per periodic cell. The same three fit seeds, architecture, augmentation, 4000updates, draw budgets and draw seeds are retained. Terminal checkpoints are primary; validation-best checkpoints remain sensitivity outputs.

Both teachers evaluate every proposal distribution. TRAIN-only Boltzmann weighting can miss probability carried by held-out orbits. The10%uniform mixture restores support for unbiased shifted Z estimates, but it does not guarantee efficient importance weights. The full reference was already analyzed before this follow-up; these are development comparisons, not a fresh blind TEST.

At Li6/12,1000K and8192draws (mean errors over three draw seeds within each fit, then three independent fits):

- uma, composition_diversity_AR: entropy MAE=0.0165689kB/cell, fit SD=0.00697643, ESS/draws=0.145392.
- uma, Boltzmann_T1000_train_uma: entropy MAE=0.00535037kB/cell, fit SD=0.00126414, ESS/draws=0.682441.
- uma, Boltzmann_T1000_train_mace: entropy MAE=0.0044325kB/cell, fit SD=0.000699166, ESS/draws=0.497273.
- uma, uniform_microstate: entropy MAE=0.0109552kB/cell, fit SD=0, ESS/draws=0.179537.
- mace, composition_diversity_AR: entropy MAE=0.0146049kB/cell, fit SD=0.00699662, ESS/draws=0.235177.
- mace, Boltzmann_T1000_train_uma: entropy MAE=0.00410513kB/cell, fit SD=0.00228477, ESS/draws=0.655089.
- mace, Boltzmann_T1000_train_mace: entropy MAE=0.00456747kB/cell, fit SD=0.00131383, ESS/draws=0.753613.
- mace, uniform_microstate: entropy MAE=0.0103335kB/cell, fit SD=0, ESS/draws=0.270339.

The target-role-mass table is a post-fit diagnostic. The comparison table includes n=3/6/9 and 300/1000/5000K, with1000K own-teacher evaluation primary. All entropy estimators retain finite-budget bias; no finite-lattice entropy is promoted to physical LCO entropy.
