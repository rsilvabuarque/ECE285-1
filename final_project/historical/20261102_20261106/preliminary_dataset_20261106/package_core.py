"""Package the already validated learning inputs; no chemistry or training runs."""
import gzip,hashlib,json,tarfile
from pathlib import Path
HERE=Path(__file__).resolve().parent;PACK=HERE/'collaborator_package'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def build_package():
    files=sorted(p for p in PACK.rglob('*') if p.is_file() and p.name!='MANIFEST.json' and '__pycache__' not in p.parts and p.suffix!='.pyc')
    manifest=dict(files={str(p.relative_to(PACK)):sha(p) for p in files},scope='MLIP-free 48-site learning package. Runtime bytecode is excluded. Raw chemistry outputs and thermal data remain in the full campaign.')
    archive=HERE/'collaborator_package.tar.gz';receipt_path=HERE/'validation/core_package.json'
    if archive.exists() and receipt_path.exists() and json.loads((PACK/'MANIFEST.json').read_text())==manifest:
        previous=json.loads(receipt_path.read_text())
        if previous['sha256']==sha(archive) and previous['bytes']==archive.stat().st_size:return archive
    tmp=PACK/'MANIFEST.tmp';tmp.write_text(json.dumps(manifest,indent=2)+'\n');tmp.replace(PACK/'MANIFEST.json')
    temp=HERE/'collaborator_package.tmp.tar.gz'
    with temp.open('wb') as raw,gzip.GzipFile(filename='',mode='wb',fileobj=raw,mtime=0) as gz,tarfile.open(fileobj=gz,mode='w') as tar:
        for p in files+[PACK/'MANIFEST.json']:tar.add(p,arcname=str(Path('collaborator_package')/p.relative_to(PACK)),recursive=False)
    temp.replace(archive)
    receipt=dict(status='PASS',archive=str(archive),bytes=archive.stat().st_size,sha256=sha(archive),files=len(files)+1,scope='Portable core package only; separate full-campaign and thermal audits remain required.')
    path=HERE/'validation/core_package.json';tmp=path.with_suffix('.tmp');tmp.write_text(json.dumps(receipt,indent=2)+'\n');tmp.replace(path);return archive
if __name__=='__main__':print('CORE_PACKAGE_READY',build_package(),flush=True)
