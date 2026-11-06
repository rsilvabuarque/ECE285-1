"""Acquire pinned upstream weights without redistributing gated checkpoints."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import shutil
import urllib.request

from .registry import MODELS


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(block)
    return digest.hexdigest()


def cache_root(value=None):
    return Path(value or os.environ.get("CATHODE_CVAE_MODEL_DIR", Path.home() / ".cache/cathode-cvae/models")).expanduser()


def model_path(name, root=None):
    return cache_root(root) / name / MODELS[name]["entry"]


def verify(name, path=None, root=None):
    spec = MODELS[name]
    path = Path(path) if path is not None else model_path(name, root)
    expected = spec["files"]
    if len(expected) == 1:
        files = {path.name: path}
        wanted = {path.name: next(iter(expected.values()))}
    else:
        files = {filename: path / filename for filename in expected}
        wanted = expected
    hashes = {}
    for filename, source in files.items():
        if not source.is_file():
            raise FileNotFoundError(f"Missing {name} checkpoint file: {source}. Run 'cathode-cvae mlip fetch-models --models {name}'.")
        hashes[filename] = sha256(source)
        if hashes[filename] != wanted[filename]:
            raise ValueError(f"SHA256 mismatch for {source}; expected pinned {spec['name']} weights. Refusing a silent model change.")
    return path, hashes


def fetch(name, root=None):
    spec = MODELS[name]
    directory = cache_root(root) / name
    directory.mkdir(parents=True, exist_ok=True)
    for filename, expected in spec["files"].items():
        destination = directory / filename
        if destination.is_file() and sha256(destination) == expected:
            continue
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(destination.suffix + ".download")
        if "repo" in spec:
            from huggingface_hub import hf_hub_download
            try:
                source = hf_hub_download(spec["repo"], filename=filename, revision=spec["revision"])
            except Exception as error:
                # Do not include tokens, request headers, or the full upstream exception.
                raise RuntimeError(f"Cannot fetch {spec['name']} ({type(error).__name__}). For UMA, obtain model access at https://huggingface.co/facebook/UMA and run 'hf auth login' in this environment. We do not accept licenses on your behalf.") from None
            shutil.copyfile(source, temporary)
        else:
            with urllib.request.urlopen(spec["url"], timeout=120) as source, temporary.open("wb") as target:
                shutil.copyfileobj(source, target)
        if sha256(temporary) != expected:
            raise ValueError(f"Downloaded {name}/{filename} does not match the campaign's SHA256; retained {temporary} for inspection.")
        temporary.replace(destination)
    return verify(name, root=root)[0]
