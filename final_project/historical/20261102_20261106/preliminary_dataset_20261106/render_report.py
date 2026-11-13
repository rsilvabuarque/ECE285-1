"""Render the finished report and check document-level completeness."""
import hashlib,json,os,importlib.metadata
from pathlib import Path
import pymupdf
HERE=Path(__file__).resolve().parent
assert os.environ.get('SLURMD_NODENAME','').startswith('nid')
path=HERE/'reports/preliminary_report.pdf';doc=pymupdf.open(path);assert len(doc)==10
text='\n'.join(p.get_text() for p in doc);assert 'DRAFT' not in text and 'PENDING' not in text
for token in ['4000','54,810','636/2,000','440/2,000','Nov. 9','Dec. 12']:assert token in text,token
preview=HERE/'reports/previews_final';preview.mkdir(exist_ok=True)
lengths=[]
for i,page in enumerate(doc):
    lengths.append(len(page.get_text()));assert lengths[-1]>1000
    page.get_pixmap(dpi=110,alpha=False).save(preview/f'page_{i+1:02d}.png')
log=(HERE/'reports/preliminary_report.log').read_text();assert 'Overfull' not in log and 'undefined' not in log.lower()
r=dict(status='PASS',pages=len(doc),page_text_characters=lengths,pdf_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),renderer=importlib.metadata.version('pymupdf'),checks=['All10pages render','No draft/pending placeholders','Expected measured counts and deadlines present','No blank/spillover-only pages','No overfull/undefined TeX diagnostics'],remaining='Root visual inspection of rendered pages; final scheduler accounting snapshot after allocation release.')
(HERE/'validation/report_render.json').write_text(json.dumps(r,indent=2)+'\n');print(json.dumps(r),flush=True)
