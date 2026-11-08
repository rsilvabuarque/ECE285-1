"""Supervise the frozen paired queue and collect once every attempt is terminal.

Only status filenames are counted while waiting. The collector independently
verifies identities and keeps TEST energy values sealed. This does not retry
failed physics, alter roles, allocate resources, or wait beyond its deadline.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

from cathode_cvae.runtime import require_compute_allocation


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('prepared', 'queue', 'output-dir', 'collector'):
        p.add_argument('--' + name, type=Path, required=True)
    p.add_argument('--deadline-unix', type=float, required=True)
    a = p.parse_args()
    require_compute_allocation()
    expected = len(json.loads((a.prepared / 'tasks_all.json').read_text())['tasks'])
    while time.time() < a.deadline_unix:
        terminal = len(list((a.queue / 'results').glob('*/status.json')))
        print(json.dumps(dict(unix=time.time(), terminal=terminal, expected=expected)), flush=True)
        if terminal == expected:
            return subprocess.call([sys.executable, str(a.collector), '--prepared', str(a.prepared),
                '--queue', str(a.queue), '--output-dir', str(a.output_dir)])
        if terminal > expected:
            raise ValueError('Queue includes unexpected assessment attempts before acquisition collection')
        time.sleep(min(60, max(0, a.deadline_unix - time.time())))
    raise TimeoutError('Acquisition collection deadline; no retries or partial fit authorized')


if __name__ == '__main__':
    raise SystemExit(main())
