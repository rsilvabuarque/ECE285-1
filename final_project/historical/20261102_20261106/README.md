# Historical project sources and evidence: November 2–6

This directory preserves a selective, byte-for-byte copy of earlier project
documents, authored analysis code, and compact result tables and figures. See
`manifest.json` for source-relative paths and SHA-256 hashes. The original
working directories and raw outputs remain in the original `ECE285/final_project`
tree on the shared filesystem; they were not moved or removed.

**These documents describe earlier project scope.** In particular, the
November 2 design proposes a 144-Li-site main study, while the later controlled
generative experiments use a fixed 48-site O3 parent. The collaborator brief
discusses broader structural and geometry-generation ideas; it is useful
background, not the current experiment protocol.

The subdirectories retain their original names:

- `collaborator_brief/`: the early explanatory PDF and source.
- `energy_ranking_audit/multimodel_20261104/`: compact historical and Hu-archive
  energy/ranking summaries and structure-mapping checks.
- `energy_ranking_audit/vacancy144_symmetry/`: 144-site configuration analysis.
- `energy_ranking_audit/relaxation_pilot48_20261105/`: the 48-site model timing
  and relaxation pilot.
- `preliminary_dataset_20261106/`: earlier preparation, model, and analysis
  scripts; the later release is the recommended starting point.

The scripts are preserved as **historical executed-source evidence**. Many
contain old absolute paths, allocation settings, and environment assumptions;
read and adapt them before use. Published files here are not a complete replay
bundle. The large local environments, checkpoints, third-party per-frame data,
run queues, and raw structures are not copied into this repository. The
November 8 [release](https://github.com/rsilvabuarque/ECE285-1/releases/tag/ece285-overnight-20261108-v1)
contains the earlier curated training and evaluation packet.
