# Compact evidence for the November 12–13 update

This directory contains aggregate CSVs, protocols, and audit receipts used by
[the overview and technical report](../../reports/adaptive_20261112/README.md).
`manifest.json` records each original source path, byte size, and SHA256.
`resource_receipt.json` records final allocation usage after closure.
`report_build_receipt.json` records the successful report build.

Rebuild all seven report tables without training or MLIP access:

```bash
python reports/adaptive_20261112/build_tables.py --run-root artifacts/adaptive_20261112
```

The complete raw campaign remains in the ignored HPC run archive at
`final_project/runs/adaptive_20261112/`. Compact aggregates are sufficient for
the report's table arithmetic; full raw-attempt replay and retraining require
that archive, the pinned input packet, and separately authorized model access.
Do not interpret this directory as a newly independent assessment dataset.
