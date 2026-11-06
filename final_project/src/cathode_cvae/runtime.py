"""Site controls apply on their site; other GPU hosts need no Slurm emulation."""

import os
import socket


def require_compute_allocation():
    """Reject substantial work on known NERSC login hosts, including CPU fits.

    This is a guard for the project's current host, not a portable scheduler
    detector. An allocation on a different site must follow that site's rules.
    CUDA_VISIBLE_DEVICES remains entirely controlled by the caller/scheduler.
    """
    host = socket.gethostname().split(".")[0]
    if host.startswith("perlmutter") or host.startswith("login") and os.environ.get("NERSC_HOST"):
        raise RuntimeError("Use a compute node for training/scoring/large analysis on NERSC; run this command inside your allocation.")
