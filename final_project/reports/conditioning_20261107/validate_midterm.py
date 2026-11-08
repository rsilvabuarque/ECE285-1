"""Render the compiled report and check mechanical delivery properties.

These checks do not validate scientific claims. The claim/source manifest and
human visual/interpretive review provide separate evidence.
"""
import argparse
import hashlib
import json
import platform
from pathlib import Path


def main():
    import pymupdf
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--report-dir", type=Path, default=Path("docs/midterm1"))
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--phase", choices=["interim", "required", "final"], required=True)
    a = p.parse_args()
    a.output_dir.mkdir(parents=True, exist_ok=False)
    source = a.report_dir / "main.pdf"
    doc = pymupdf.open(source)
    texts = [page.get_text() for page in doc]
    ref_pages = [i for i, text in enumerate(texts) if text.lstrip().startswith("References")]
    assert ref_pages == [5], f"Expected five body pages then references, found {ref_pages}"
    assert len(doc) in [6,7], f"Unexpected report page count {len(doc)}"
    assert all(len(text.strip()) > 900 for text in texts[:5]), "Blank/spillover body page"
    assert all(len(text.strip()) > 250 for text in texts[5:]), "Blank/spillover reference page"
    log = (a.report_dir / "main.log").read_text()
    assert "Overfull" not in log and "undefined" not in log.lower()
    combined = "\n".join(texts)
    if a.phase != "interim":
        for phrase in ["Interim scope", "comparisons are executing", "will be inserted", "PENDING"]:
            assert phrase not in combined, f"Unfinished placeholder: {phrase}"
    for i, page in enumerate(doc):
        page.get_pixmap(dpi=110, alpha=False).save(a.output_dir / f"page_{i+1:02d}.png")
    receipt = dict(status="PASS", phase=a.phase, body_pages=5,
        reference_pages=len(doc)-5, page_text_characters=[len(x) for x in texts],
        pdf_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        tex_sha256=hashlib.sha256((a.report_dir/"main.tex").read_bytes()).hexdigest(),
        renderer=pymupdf.VersionBind, python=platform.python_version(),
        no_overfull_or_undefined_diagnostics=True, all_pages_rendered=True,
        scientific_claims_validated_by_this_check=False,
        visual_review="Required separately after rendering")
    (a.output_dir / "receipt.json").write_text(json.dumps(receipt, indent=2)+"\n")
    print(json.dumps(receipt,indent=2))


if __name__ == "__main__":
    main()
