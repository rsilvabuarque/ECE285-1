"""Independently verify the durable tar without extracting or loading models."""
import argparse,hashlib,json,tarfile,time
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True)
p.add_argument('--archive',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
m=json.loads(a.manifest.read_text());archive=a.archive or Path(m['archive']);start=time.perf_counter()
def checksum(stream):
 h=hashlib.sha256()
 while block:=stream.read(1024*1024):h.update(block)
 return h.hexdigest()
assert archive.stat().st_size==m['archive_bytes']
with archive.open('rb') as stream:actual=checksum(stream)
assert actual==m['archive_sha256'],'Whole archive checksum mismatch'
seen=set();count=0;total=0
with tarfile.open(archive,'r:') as tar:
 for item in tar:
  assert item.isfile() and item.name not in seen,'Unexpected nonregular or duplicate member'
  seen.add(item.name)
  stream=tar.extractfile(item)
  if item.name==m['internal_manifest']:
   payload=stream.read();assert hashlib.sha256(payload).hexdigest()==m['content_manifest_sha256']
   inner=json.loads(payload);assert inner['files']==m['files']
   continue
  entry=m['files'][item.name]
  assert item.size==entry['bytes'],item.name
  assert checksum(stream)==entry['sha256'],item.name
  count+=1;total+=item.size
assert seen==set(m['files'])|{m['internal_manifest']}
assert count==m['file_count'] and total==m['total_source_bytes']
out=dict(status='PASS',archive=str(archive),archive_sha256=actual,archive_bytes=archive.stat().st_size,
         payload_files=count,total_payload_bytes=total,internal_manifest_verified=True,
         all_member_hashes_verified=True,source_manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
         verification_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
         elapsed_seconds=time.perf_counter()-start,scope='Archive integrity only, not scientific accuracy')
a.output.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
