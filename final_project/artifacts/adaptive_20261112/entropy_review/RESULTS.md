# Small-cell entropy validation: measured results

This is an idealized O3 Li12Co12O24 fixed-cell vacancy lattice. Its 4096 labelled masks form 87 symmetry orbits. Every orbit representative was scored by both UMA and MACE, with 36 additional as-written symmetry checks per model. No DFT or relaxation labels are used; the teacher energy is the total eV per periodic cell.

The symmetry checks measure numerical consistency of the MLIP under the lattice operations. They do not validate physical energies, nor bound every untested orientation. Multiplicity-expanded thermodynamics assumes invariance to the numerical scale shown below.

- mace: maximum tested |ΔE| = 3.0517578e-05 eV per cell; |ΔE|/(kB × 100 K) = 0.00354142; single-state Boltzmann-factor ratio ≤ 1.00355 over those tested pairs.
- uma: maximum tested |ΔE| = 0.0017960627 eV per cell; |ΔE|/(kB × 100 K) = 0.208424; single-state Boltzmann-factor ratio ≤ 1.23174 over those tested pairs.

For fixed composition n, Z = Σo go exp(−Eo/kBT), where go is the number of labelled masks in orbit o. S/kB = ln Z + ⟨E⟩/(kBT); constant energy offsets cancel. The infinite-temperature limit is ln[C(12,n)], not ln(number of symmetry orbits). These are finite-cell configurational quantities, excluding vibrations, spin states, strain, phase transformations, and inter-composition exchange.

The bounded Wang–Landau experiment completed 36 runs; 36 met its prospectively specified flatness/modification-factor schedule. Exact nonempty-bin support was supplied. A successful schedule is a numerical diagnostic rather than proof of unknown-space coverage; midpoint binning and DOS errors remain distinct.

Three composition-only autoregressive models were trained on the frozen 57 TRAIN / 15 VAL / 15 TEST orbit split with multiplicity weights and symmetry augmentation. TEST energies are NaN in the learning export. The terminal 4000-update checkpoints are primary; validation-selected checkpoints are retained as sensitivity results. No physical energy target was conditioned on during this tiny entropy test.

Each learned proposal is mixed with 10% uniform labelled-mask probability to guarantee support. Exact AR likelihoods were normalized by exhaustive enumeration, then independently checked against decoder samples. The proposal distribution is not a Boltzmann distribution. Importance weights divide by the actual proposal probability; naive generated-sample frequencies estimate proposal entropy instead. Partition-function sample means are unbiased under full support; self-normalized energies and the resulting entropy retain finite-budget bias.

The comparison table first averages over three draw seeds within each fit, then averages the three independent fit means. The reported fit spread is descriptive; it is not a confidence interval from nine independent fits. Weight ESS measures weight concentration and does not measure Markov-chain independence.

- mace, Li6/12, 300 K, 16 WL bins: midpoint-only entropy bias +0.0224775 kB per cell; mean absolute DOS-estimation error relative to the binned reference 0.0341832 kB. The two contributions may cancel; small total error alone does not establish either convergence.
- mace, Li6/12, 300 K, 32 WL bins: midpoint-only entropy bias +0.0212779 kB per cell; mean absolute DOS-estimation error relative to the binned reference 0.0342982 kB. The two contributions may cancel; small total error alone does not establish either convergence.
- uma, Li6/12, 300 K, 16 WL bins: midpoint-only entropy bias +1.08008 kB per cell; mean absolute DOS-estimation error relative to the binned reference 0.0712064 kB. The two contributions may cancel; small total error alone does not establish either convergence.
- uma, Li6/12, 300 K, 32 WL bins: midpoint-only entropy bias +0.243815 kB per cell; mean absolute DOS-estimation error relative to the binned reference 0.0416578 kB. The two contributions may cancel; small total error alone does not establish either convergence.

- mace, Li6/12, 1000 K, 8192 draws, terminal_AR_mixture: importance entropy MAE 0.014605 kB per cell; naive entropy MAE 1.0223 kB; weight ESS/draws 0.23518.
- mace, Li6/12, 1000 K, 8192 draws, uniform_microstate: importance entropy MAE 0.010334 kB per cell; naive entropy MAE 1.0462 kB; weight ESS/draws 0.27034.
- uma, Li6/12, 1000 K, 8192 draws, terminal_AR_mixture: importance entropy MAE 0.016569 kB per cell; naive entropy MAE 1.2916 kB; weight ESS/draws 0.14539.
- uma, Li6/12, 1000 K, 8192 draws, uniform_microstate: importance entropy MAE 0.010955 kB per cell; naive entropy MAE 1.3155 kB; weight ESS/draws 0.17954.

These checks establish estimator behavior for a complete small lattice and the chosen MLIP energies. They do not establish physical LCO entropy, transfer to N48, or a likelihood-corrected CVAE entropy estimator. A CVAE would require its latent-marginal proposal probability, not the decoder likelihood at one sampled latent vector.
