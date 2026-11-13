# Repository layout and consolidated branches

The shared project is `rsilvabuarque/ECE285-1`, branch `main`. The consolidated
November 13 branch `project/consolidated-20261113` records the same published
integration point. Collaborators should clone `main` and create personal branches.

The extra `ECE285_*` directories were linked Git worktrees: separate checkouts
of branches sharing one Git object database and branch history. They contained
overlapping source by design. A file edit belongs to its checkout until committed
and integrated. Shared raw-data paths and output directories can still interfere
if two processes write to them, so use one owner per checkout and fresh output
directories. The consolidation audit found no missing historical final-project
source paths; patch-equivalent and superseded variants were reviewed.

All 14 historical worktrees now physically reside below
`ECE285/runs/archived_worktrees/`. Original sibling paths are compatibility
symlinks, retained because immutable receipts, saved commands, and four oracle
links reference them. No raw result, untracked historical output, branch, or
checkpoint was deleted. A pre-existing scratch scoring queue remains on
`/pscratch`; it is an external storage dependency, not another source branch.

## Active and preserved content

| Path inside ECE285 | Purpose |
|---|---|
| `final_project/src`, `tests`, `configs`, `scripts` | Integrated package and experiment methods |
| `final_project/reports/adaptive_20261112` | Latest overview, detailed report, figures, LaTeX, and build instructions |
| `final_project/artifacts/adaptive_20261112` | Source-bound aggregate results, protocols, integrity/resource receipts |
| `final_project/docs` | Collaboration, installation, scientific methods, and preserved midterm documents |
| `final_project/historical/20261102_20261106` | Original design, audits, and early work preserved separately |
| `final_project/runs/adaptive_20261112` | Full latest raw campaign, checkpoints, inputs, attempts, exclusions, and failures; HPC only |
| `runs/archived_worktrees` | Historical branch checkouts and their raw/untracked outputs; HPC only |
| `runs/adaptive_goal_20261112` | Isolated development/integration worktrees for the latest campaign; HPC only |

Git contains code, reports, compact scientific evidence, and data-release manifests.
Foundation weights, software environments, full raw force/stress arrays, and large
run archives stay outside Git. Older verified releases remain downloadable using
their committed manifests. The new compact packet replays report tables; full
latest-training or raw-attempt replay additionally requires the HPC archive and
pinned provider weights. Absolute paths in historical receipts are provenance.

## Relocated worktrees

| Historical name | Preserved branch |
|---|---|
| `ECE285_angular_audit_goal_20261111` | `goal/angular-assessment-audit-20261111` |
| `ECE285_angular_directional_agent_20261110` | `agent/angular-directional-20261110` |
| `ECE285_angular_evaluation_agent_20261110` | `agent/angular-evaluation-20261110` |
| `ECE285_angular_goal_20261110` | `goal/angular-latent-iteration-20261110` |
| `ECE285_angular_iteration_agent_20261110` | `agent/angular-iteration-20261110` |
| `ECE285_angular_iteration_goal_20261111` | `goal/angular-iteration-20261111` |
| `ECE285_angular_round2_goal_20261111` | `goal/angular-iteration-round2-20261111` |
| `ECE285_midterm_submission_20261109` | `midterm/submission-neurips-20261109` |
| `ECE285_midterm_worktree` | `overnight/multibin-acquisition-20261107` |
| `ECE285_next_goal_worktree` | `midterm/conditioning-graph-edit-20261107` |
| `ECE285_overnight_20261108` | `overnight/latent-relaxed-neb-20261108` |
| `ECE285_overnight_models_20261108` | `agent/overnight-20261108-models` |
| `ECE285_overnight_paired_20261108` | `agent/overnight-20261108-paired` |
| `ECE285_overnight_research_20261108` | `agent/overnight-20261108-research` |

The [completeness audit](../artifacts/adaptive_20261112/consolidation/AUDIT.md)
and [relocation receipt](../artifacts/adaptive_20261112/consolidation/relocation_receipt.json)
record exact tips, destinations, unchanged directory inodes, tracked-status checks,
and compatibility links. Earlier negative, failed, and superseded results remain
in their archived locations; they are not presented as the current result.

Unrelated homework edits in the root checkout were preserved and excluded from
project commits. The project consolidation does not merge homework branches.
