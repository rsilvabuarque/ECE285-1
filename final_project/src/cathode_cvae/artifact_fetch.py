"""Download a hash-verified run bundle without overwriting collaborator edits."""
import argparse,hashlib,stat,tempfile,urllib.request,zipfile
from pathlib import Path,PurePosixPath


def extract_verified(archive,output):
 output=Path(output).resolve();members=[]
 with zipfile.ZipFile(archive) as z:
  for info in z.infolist():
   rel=PurePosixPath(info.filename)
   if rel.is_absolute() or '..' in rel.parts or any(':' in p for p in rel.parts) or stat.S_ISLNK(info.external_attr>>16):raise ValueError('Unsafe archive member')
   if info.is_dir():continue
   target=output.joinpath(*rel.parts)
   if not target.resolve().is_relative_to(output):raise ValueError('Archive member escapes destination')
   data=z.read(info)
   if target.exists() and (not target.is_file() or hashlib.sha256(target.read_bytes()).digest()!=hashlib.sha256(data).digest()):raise FileExistsError(f'Preserving different local file: {target}')
   members.append((target,data))
  # Validate every existing destination before writing any payload.
  for target,data in members:
   if target.exists():continue
   target.parent.mkdir(parents=True,exist_ok=True)
   with target.open('xb') as f:f.write(data)
 return len(members)


def main(argv=None):
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--url',required=True);p.add_argument('--sha256',required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args(argv)
 if not a.url.startswith('https://'):p.error('Use an HTTPS artifact URL')
 if len(a.sha256)!=64 or any(c not in '0123456789abcdefABCDEF' for c in a.sha256):p.error('A full SHA256 checksum is required')
 with tempfile.TemporaryDirectory(prefix='cathode-run-download-') as tmp:
  archive=Path(tmp)/'run.zip';h=hashlib.sha256()
  with urllib.request.urlopen(a.url,timeout=120) as source,archive.open('wb') as dest:
   while block:=source.read(8*1024**2):h.update(block);dest.write(block)
  if h.hexdigest()!=a.sha256.lower():raise ValueError('Artifact SHA256 mismatch; no files extracted')
  count=extract_verified(archive,a.output_dir)
 print(f'VERIFIED_ARTIFACT {count} files {h.hexdigest()}')
 return 0


if __name__=='__main__':raise SystemExit(main())
