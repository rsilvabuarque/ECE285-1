# First 10 ps: exploratory restricted-window diagnostic

This common prefix is the first half-window from the original blocking plan.
It was selected after the assignment-event audit, before inspecting primary
free-energy estimates. All four runs retain their initial atom/site assignments
through this interval at the saved 2 fs structural resolution. That observation
does not certify an equilibrium basin ensemble or remove the sampling effects
of the posthoc window choice.

The primary lag is 2.5 ps. The matched blocking control holds lag at 0.625 ps
across the full prefix and its shorter blocks. Raw native-temperature fields,
approximate 433.15 K transports, all warnings and 40 numerical sensitivity runs
remain available. No trajectory was resampled or restrained.

Use `prefix10_transport_ranges_scoped.pdf/png` and the neighboring
`prefix10_cross_seed_ranges/` tables. Their ranges enumerate the four differences
formed from two trajectories per basin; those four derived values are correlated
and are not four independent samples or a confidence interval. Both temperature
transports retain a sign-changing 3PT range. These estimates do not establish a
validated basin ordering or physical populations.
