"""Device checks and provenance shared by the portable learning commands."""
import hashlib
import os
import socket
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(8 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def device_name(requested):
    import torch
    result = ('cuda' if torch.cuda.is_available() else 'cpu') if requested == 'auto' else requested
    if result == 'cuda':
        if not torch.cuda.is_available():
            raise RuntimeError('CUDA was requested but is unavailable in this Python environment')
        # Preserve NERSC's allocation rule without requiring Slurm at other sites.
        host = socket.gethostname().lower()
        if os.environ.get('NERSC_HOST') and ('login' in host) and not os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Use a compute allocation for GPU work on a NERSC login host')
    return result


def trusted_checkpoint(path, device):
    """Load an explicitly supplied, trusted local project checkpoint.

    Torch checkpoints can contain executable pickle data; do not use this helper
    for untrusted downloads. The bundled manifests identify the project files.
    """
    import torch
    return torch.load(path, map_location=device, weights_only=False)
