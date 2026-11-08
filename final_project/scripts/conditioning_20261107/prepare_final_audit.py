"""Select the prespecified MACE audit by method, before reading any final labels.

The UMA assessment remains the full frozen pool. Both teachers assess exactly
the same attempts in this subset, using their own TRAIN-fitted energy bins.
"""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np

from cathode_cvae.data import sha256
from cathode_cvae.runtime import require_compute_allocation


def prepare(root):
    run = root/"runs/conditioning_20261107"
    protocol_path = run/"final_protocol.json"
    protocol = json.loads(protocol_path.read_text())
    generated = run/"final"
    receipt = json.loads((generated/"receipt.json").read_text())
    if receipt["manifest_sha256"] != protocol["generation_manifest_sha256"]:
        raise ValueError("Generation differs from the frozen final manifest")
    for name, expected in receipt["outputs"].items():
        if sha256(generated/name) != expected:
            raise ValueError("Final generation output changed: "+name)
    with (generated/"attempts.csv").open() as stream:
        reader = csv.DictReader(stream);fields = reader.fieldnames;rows = list(reader)
    if len(rows) != protocol["UMA"]["expected_attempts"]:
        raise ValueError("Full final attempt budget differs")
    families = set(protocol["MACE"]["families"])
    selected = [r for r in rows if r["method"] in families]
    counts = Counter((r["method"],r["replicate_id"],r["n_Li"],r["requested_bin"]) for r in selected)
    if len(selected) != protocol["MACE"]["expected_attempts"] or set(counts.values()) != {protocol["attempts_per_cell"]}:
        raise ValueError("Matched audit attempt budget differs")
    wanted = {r["scoring_structure_id"] for r in selected}
    with np.load(generated/"candidates.npz",allow_pickle=False) as z:
        ids = z["structure_ids"].astype(str)
        keep = np.asarray([s in wanted for s in ids])
        if set(ids[keep]) != wanted:
            raise ValueError("An audited attempt has no exact candidate geometry")
        values = {k:z[k][keep] for k in ["masks","occupancy","structure_ids","n_Li","n_atoms"]}
        values["parent_sha256"] = z["parent_sha256"]
    output = run/"final_matched_audit";output.mkdir(exist_ok=False)
    with (output/"attempts.csv").open("x",newline="") as stream:
        writer = csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(selected)
    np.savez_compressed(output/"candidates.npz",**values)
    result = dict(status="MATCHED_AUDIT_FROZEN_BEFORE_LABEL_ACCESS",attempts=len(selected),
        unique_geometries=len(wanted),families=sorted(families),condition_groups=len(counts),
        selection="Method membership only, exactly as in the pre-generation final protocol",
        raw_attempts_reference=str((generated/"raw_attempts.npz").resolve()),
        source_protocol_sha256=sha256(protocol_path),generation_receipt_sha256=sha256(generated/"receipt.json"),
        source_attempts_sha256=sha256(generated/"attempts.csv"),source_candidates_sha256=sha256(generated/"candidates.npz"),
        outputs={n:sha256(output/n) for n in ["attempts.csv","candidates.npz"]},
        source_sha256=sha256(__file__),energy_values_read=0)
    (output/"receipt.json").write_text(json.dumps(result,indent=2)+"\n")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root",type=Path,required=True)
    args = parser.parse_args();require_compute_allocation()
    print(json.dumps(prepare(args.root.resolve()),indent=2))
