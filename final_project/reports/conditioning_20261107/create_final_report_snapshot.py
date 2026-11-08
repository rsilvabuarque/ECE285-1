"""Verify an isolated report rebuild and freeze all project-local build inputs.

This is a report artifact operation. It changes no model, labels, assessment,
required checkpoint, publication status or allocation state.
"""
import hashlib
import json
from pathlib import Path
import re
import subprocess
import sys

import pymupdf

ROOT=Path(__file__).resolve().parents[2]


def digest(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    report=ROOT/"docs/midterm1";preview=ROOT/"runs/conditioning_20261107/report_final_preview"
    verification=json.loads((preview/"receipt.json").read_text())
    visual=json.loads((preview/"visual_review.json").read_text())
    assert verification["status"]=="PASS" and verification["phase"]=="final"
    assert verification["body_pages"]==5 and verification["reference_pages"]==1
    assert visual["pages_reviewed"]==[1,2,3,4,5,6]
    assert verification["pdf_sha256"]==visual["pdf_sha256"]==digest(report/"main.pdf")
    gate_path=ROOT/"artifacts/conditioning_20261107/required_gate.json"
    gate=json.loads(gate_path.read_text())
    assert digest(gate_path)=="c6a7105dd9578c5a1caac1c8bc6acfb2bc134e8fd1ce04e2a0c8def232f91bf3"
    old_map=ROOT/"reports/conditioning_20261107/public_source_map.json"
    assert digest(old_map)==gate["bindings"][str(old_map.relative_to(ROOT))]
    required=ROOT/"reports/conditioning_20261107/required_report_snapshot"
    required_manifest=json.loads((required/"manifest.json").read_text())
    for name,value in required_manifest["files"].items():assert digest(required/name)==value
    # Recursively collect actual project-local TeX inputs; standard LaTeX
    # classes/packages/plain.bst remain supplied by the recorded distribution.
    pending=["main.tex"];inputs={}
    while pending:
        name=pending.pop()
        if name in inputs:continue
        text=(report/name).read_text();inputs[name]=report/name
        pending.extend(value if Path(value).suffix else value+".tex" for value in re.findall(r"\\input\{([^}]+)\}",text))
    for value in re.findall(r"\\bibliography\{([^}]+)\}",(report/"main.tex").read_text()):
        for name in value.split(","):inputs[name+".bib"]=report/(name+".bib")
    inputs["build.sh"]=report/"build.sh"
    rebuild=ROOT/"runs/conditioning_20261107/report_selfcontained_rebuild"
    rebuild.mkdir(parents=True,exist_ok=False)
    for name,path in inputs.items():
        target=rebuild/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    with (rebuild/"build.log").open("w") as log:
        subprocess.run(["bash","build.sh"],cwd=rebuild,stdout=log,stderr=subprocess.STDOUT,check=True)
    original=pymupdf.open(report/"main.pdf");repeat=pymupdf.open(rebuild/"main.pdf")
    assert len(original)==len(repeat)==6
    assert [p.get_text() for p in original]==[p.get_text() for p in repeat]
    log=(rebuild/"main.log").read_text()
    assert "Overfull" not in log and "undefined" not in log.lower()
    rebuild_receipt=dict(status="PASS_SELF_CONTAINED_REBUILD",body_pages=5,reference_pages=1,
        all_six_page_texts_equal=True,project_build_inputs={name:digest(path) for name,path in inputs.items()},
        original_pdf_sha256=digest(report/"main.pdf"),rebuilt_pdf_sha256=digest(rebuild/"main.pdf"),
        standard_distribution="texlive/2024; declared standard article/packages/plain bibliography",
        build_command="bash build.sh",new_model_or_oracle_calls=0)
    (rebuild/"receipt.json").write_text(json.dumps(rebuild_receipt,indent=2)+"\n")
    command=[sys.executable,str(ROOT/"reports/conditioning_20261107/refresh_report_claims.py"),
        "--verification",str(preview/"receipt.json"),"--visual-review",str(preview/"visual_review.json"),
        "--neb-receipt",str(ROOT/"runs/conditioning_20261107/neb/basin_1_repeat2/receipt.json"),
        "--neb-receipt",str(ROOT/"runs/conditioning_20261107/neb/basin_2_repeat1/receipt.json"),"--final"]
    heavy=ROOT/"reports/conditioning_20261107/execution_heavy.json"
    if heavy.exists():command.extend(["--cost-receipt",str(heavy)])
    subprocess.run(command,cwd=ROOT,check=True)
    snapshot=ROOT/"reports/conditioning_20261107/final_report_snapshot"
    snapshot.mkdir(parents=True,exist_ok=False)
    sources={**inputs,"main.pdf":report/"main.pdf","main.bbl":report/"main.bbl",
        "claim_sources.json":report/"claim_sources.json","CLAIM_SOURCE_MAP.md":report/"CLAIM_SOURCE_MAP.md",
        "OPTIONAL_EXTENSIONS_20261107.md":ROOT/"docs/OPTIONAL_EXTENSIONS_20261107.md",
        "build_render_receipt.json":preview/"receipt.json","visual_review.json":preview/"visual_review.json",
        "selfcontained_rebuild_receipt.json":rebuild/"receipt.json"}
    for name,path in sources.items():
        target=snapshot/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes(path.read_bytes())
    assert digest(gate_path)==gate["bindings"].get(str(gate_path.relative_to(ROOT)),"c6a7105dd9578c5a1caac1c8bc6acfb2bc134e8fd1ce04e2a0c8def232f91bf3")
    assert digest(old_map)==gate["bindings"][str(old_map.relative_to(ROOT))]
    for name,value in required_manifest["files"].items():assert digest(required/name)==value
    manifest=dict(status="FROZEN_REVIEWED_FINAL_REPORT_SNAPSHOT",pdf_sha256=digest(snapshot/"main.pdf"),
        body_pages=5,reference_pages=1,all_project_local_build_inputs_included=True,
        isolated_rebuild_page_texts_equal=True,files={name:digest(snapshot/name) for name in sources},
        source_paths={name:str(path.relative_to(ROOT)) for name,path in sources.items()},
        build=dict(command="bash build.sh",latex_environment="texlive/2024 or compatible standard LaTeX distribution"),
        required_gate_sha256=digest(gate_path),required_public_source_map_sha256=digest(old_map),
        required_snapshot_unchanged=True,
        scope="Updated report includes completed optional evidence. Required snapshot/gate/map remain unchanged; publication and post-release cost are root-owned separate receipts.")
    (snapshot/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(json.dumps(dict(status=manifest["status"],snapshot=str(snapshot),pdf_sha256=manifest["pdf_sha256"],files=len(sources)),indent=2))


if __name__=="__main__":main()
