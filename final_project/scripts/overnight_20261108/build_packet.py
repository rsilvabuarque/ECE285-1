"""Freeze explicitly selected scientific evidence and split large raw archives.

The JSON specification contains only explicit source/destination mappings. Files
are hashed before and after copying; active/changing evidence is rejected. Model
foundation weights and private account balances are deliberately never selected.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import time
import zipfile


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(2**20), b''):
            h.update(block)
    return h.hexdigest()


def target(root, relative):
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Destination escapes the packet')
    return path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--spec', type=Path, required=True)
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--archive-dir', type=Path, required=True)
    p.add_argument('--max-raw-bytes', type=int, default=900_000_000)
    a = p.parse_args()
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    started = time.monotonic()
    spec = json.loads(a.spec.read_text())
    if spec['status'] != 'FROZEN_COMPLETE_PACKAGE_INPUTS':
        raise ValueError('Only complete frozen evidence may be packaged')
    if a.output_dir.is_symlink() or (a.output_dir.exists() and any(a.output_dir.iterdir())):
        raise FileExistsError('Preserve the existing nonempty packet or symlink')
    a.output_dir.mkdir(parents=True, exist_ok=True)
    a.archive_dir.mkdir(parents=True, exist_ok=True)
    records = {}
    for item in spec['mappings']:
        source = Path(item['source'])
        paths = sorted(path for path in source.rglob('*') if path.is_file()) if source.is_dir() else [source]
        for path in paths:
            if '__pycache__' in path.parts or path.suffix in {'.pyc', '.aux', '.out', '.blg'}:
                continue
            relative = str(Path(item['destination']) / path.relative_to(source)) if source.is_dir() else item['destination']
            dest = target(a.output_dir, relative)
            before = sha(path)
            if dest.exists():
                if sha(dest) != before:
                    raise ValueError('Two distinct inputs have the same output path')
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, dest)
            if sha(dest) != before or sha(path) != before:
                raise ValueError('Source changed while copied: ' + str(path))
            records[relative] = dict(sha256=before, bytes=dest.stat().st_size, source=str(path))
    replay = dict(status='FROZEN_COMPLETE_REPLAY_PACKET', schema='overnight-replay-v1',
        files={name: row['sha256'] for name, row in records.items()}, stages=spec['stages'],
        scope='Cached assessment replay; original TRAIN/VAL/TEST roles unchanged; no foundation weights')
    (a.output_dir / 'replay_manifest.json').write_text(json.dumps(replay, indent=2) + '\n')
    archives = []
    def write_zip(name, pairs):
        path = a.archive_dir / name
        if path.exists():
            raise FileExistsError(path)
        with zipfile.ZipFile(path, 'x', compression=zipfile.ZIP_DEFLATED, compresslevel=3, allowZip64=True) as archive:
            for source, relative in pairs:
                archive.write(source, relative, compress_type=zipfile.ZIP_STORED if source.suffix in {'.npz', '.pt', '.pdf'} else zipfile.ZIP_DEFLATED)
        archives.append(dict(name=name, sha256=sha(path), bytes=path.stat().st_size, members=len(pairs)))
    core = sorted((path, str(path.relative_to(a.output_dir))) for path in a.output_dir.rglob('*') if path.is_file())
    write_zip('ece285-overnight-20261108-core.zip', core)
    # Full trajectories remain optional downloads. Each chunk contains intact
    # files, so no split-file reassembly or proprietary extractor is required.
    for group in spec.get('raw_groups', []):
        pairs = []
        for item in group['mappings']:
            source = Path(item['source'])
            paths = sorted(path for path in source.rglob('*') if path.is_file()) if source.is_dir() else [source]
            for path in paths:
                if '__pycache__' in path.parts or '.tmp.' in path.name:
                    continue
                relative = str(Path(item['destination']) / path.relative_to(source)) if source.is_dir() else item['destination']
                pairs.append((path, relative))
        part, chunk, size = 1, [], 0
        for path, relative in pairs:
            if chunk and size + path.stat().st_size > a.max_raw_bytes:
                write_zip(f"ece285-overnight-20261108-{group['name']}-{part:02d}.zip", chunk)
                part, chunk, size = part + 1, [], 0
            chunk.append((path, relative)); size += path.stat().st_size
        if chunk:
            write_zip(f"ece285-overnight-20261108-{group['name']}-{part:02d}.zip", chunk)
    result = dict(status='FROZEN_COMPLETE_PACKET', specification_sha256=sha(a.spec),
        replay_manifest_sha256=sha(a.output_dir/'replay_manifest.json'), archives=archives,
        copied_files=len(records), elapsed_seconds=time.monotonic()-started,
        private_balance_and_foundation_weights_excluded=True, provenance=records)
    (a.archive_dir/'packet_receipt.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='provenance'},indent=2))


if __name__ == '__main__':
    main()
