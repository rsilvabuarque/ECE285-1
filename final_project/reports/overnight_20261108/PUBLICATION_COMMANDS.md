# Executed publication record

These are provenance commands, not instructions to repeat existing uploads.
All mutations targeted the new release and review branch. No force push, merge,
coursework submission or collaborator message occurred.

```bash
git push origin HEAD
git tag -a ece285-overnight-20261108-v1 -m 'Frozen November8 matched model, endpoint and bounded research evidence'
git push origin refs/tags/ece285-overnight-20261108-v1
gh release create ece285-overnight-20261108-v1 \
  --repo rsilvabuarque/ECE285-1 --verify-tag --draft \
  --title 'ECE285 November8: matched CVAE controls and relaxed-target assessment' \
  --notes-file final_project/runs/overnight_20261108/release_notes.md
```

The 15 distinct payload upload arguments are recorded exactly in
`publication_commands.json`. The upload ran within allocation61003452;
`--clobber` was never used. The final core is the V2 ZIP specified in
`artifacts/overnight_20261108/download.json`; the internal first assembly
remains preserved locally and was not uploaded.

```bash
gh release upload ece285-overnight-20261108-v1 \
  final_project/reports/overnight_20261108/resources_final.json \
  --repo rsilvabuarque/ECE285-1
gh release edit ece285-overnight-20261108-v1 \
  --repo rsilvabuarque/ECE285-1 --draft=false
gh release edit ece285-overnight-20261108-v1 \
  --repo rsilvabuarque/ECE285-1 \
  --target e0f2af771e816f611df1fe1baa517db1153ff840 --verify-tag
gh release upload ece285-overnight-20261108-v1 \
  final_project/reports/overnight_20261108/publication_receipt.json \
  final_project/reports/overnight_20261108/completion_receipt.json \
  --repo rsilvabuarque/ECE285-1
```

The release target metadata was set to the existing tag's actual source SHA;
the Git tag was not moved. Its annotated object is
`93b9b5a76e83f0f8227faf711c08f9c0ed132d9f`, resolving to source commit
`e0f2af771e816f611df1fe1baa517db1153ff840`.

`publication_final_verification.json` verifies all18 remote asset digests,
public visibility and anonymous access to the core URL. The actual fresh
download, nested report snapshots and 36-table CPU replay have separate
receipts. The two protected prior-archive digests were checked unchanged.

All six top-level allocations were terminal before closing receipts were
uploaded. No private account balances or foundation-model weights were
published.
