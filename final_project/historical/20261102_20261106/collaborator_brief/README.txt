ECE 285 collaborator brief -- revision 2, 3 November 2026

Start with ECE285_Collaborator_Brief.pdf. It introduces DFT, MLIPs, and MD before
the detailed vacancy/model discussion. This revision uses only the historical
and Hu DFT archives for the project, proposes occupancy-plus-displacement
generation at a supplied cell, and includes UMA/MACE/M3GNet comparisons,
source-transfer experiments, generic parent/species inputs, and retained roles.

Build the included LaTeX source from this directory:

    module load texlive/2024  # Perlmutter; omit on a machine with TeX installed
    latexmk -pdf -interaction=nonstopmode -halt-on-error ECE285_Collaborator_Brief.tex

The included assets suffice for compilation; neither HPC data access nor Python
is needed. The main source inputs brief_preamble.tex and brief_content.tex.
The rank figures are vector TikZ/PGFPlots populated from saved audit tables.
The Li-plane cartoon is explicitly schematic.

Optional support-file regeneration from the included saved audit tables:

    python build_support.py

This only reads three small saved CSVs under assets/source_tables/. To refresh
from the original sibling audit instead, pass:

    python build_support.py --audit-results ../energy_ranking_audit/results

The script makes no predictions, submits no jobs, and does not modify raw data.
assets/evidence_extract.json records input SHA256 hashes. Full UMA evidence
and limitations remain in the sibling energy_ranking_audit/REPORT.md and results.

New symmetry-distinct pattern counts can be reproduced, where the archived
parents are accessible, with ASE, NumPy, and spglib:

    /global/cfs/cdirs/m4537/rsb/envs/fairchem/bin/python symmetry_counts.py

This is small crystallographic/integer bookkeeping on three pristine parents.
assets/symmetry_counts.json records source hashes, chosen symmetry-identification
tolerances, sensitivity probes, mapping residuals, groups/cycle histograms,
Burnside counts, and independent enumeration checks. Hu counts refer to its
inferred ideal parent symmetry; they do not identify every displaced snapshot
as exactly symmetric. The 144-site example is illustrative and excluded from
the revised training/evaluation data. Existing counts suffice to compile.

Audience tailoring used public indexed information from the exact LinkedIn
profiles supplied by Ricardo, supplemented by public publisher/team-project
records. Direct LinkedIn page retrieval was unavailable; individual contribution
details and atomistic experience were not inferred. The PDF intentionally omits
personal biographies and introduces chemistry while assuming quantitative/ML
and programming familiarity.

Public background sources consulted (3 November 2026):
https://www.linkedin.com/in/beihao-gu-b50795268/
https://www.linkedin.com/in/varadkottawar/
https://doi.org/10.1016/j.jfranklin.2024.107128
https://devpost.com/software/seismic-risk-atlas
https://devpost.com/software/cryptocurrency-prediction

Scientific references and artifact locations are listed on the final PDF page.
The statistical and scientific review checked the new symmetry counting,
joint occupancy/displacement loss, symmetry handling, and transfer interpretation.
The three frozen-potential comparison is planned: UMA is complete; no MACE,
M3GNet, new DFT, MD, or model training was executed for this document revision.
The original brief/source bundle is preserved under revisions/.
