"""Pin a configured task to one scheduler-visible GPU and record its witness.

The integrating owner supplies a frozen command list; this never allocates
resources. The backend interpreter alias is preserved so virtualenv selection
survives. Python source paths are immutable execution snapshots where specified.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import runpy
import socket
import sys
import time


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', type=Path, required=True)
    p.add_argument('--rank', type=int)
    p.add_argument('--child', action='store_true')
    a = p.parse_args()
    rank = a.rank if a.rank is not None else int(os.environ['SLURM_PROCID'])
    cfg_bytes = a.config.read_bytes()
    cfg = json.loads(cfg_bytes)
    task = cfg['tasks_by_rank'][rank]
    if not a.child:
        incoming = os.environ.get('CUDA_VISIBLE_DEVICES', '')
        visible = [x.strip() for x in incoming.split(',') if x.strip()]
        local = task.get('local_gpu_index', int(os.environ.get('SLURM_LOCALID', 0)))
        if len(visible) <= local:
            raise RuntimeError(f'Scheduler-visible GPUs {incoming!r} cannot supply local task {local}')
        os.environ['ECE285_INCOMING_GPU_VISIBILITY'] = incoming
        os.environ['CUDA_VISIBLE_DEVICES'] = visible[local]
        os.environ['PYTHONPATH'] = task['pythonpath']
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS'):
            os.environ[name] = str(task.get('threads', 4))
        # Do not resolve the Python symlink: its alias chooses the virtualenv.
        interpreter = task['python']
        os.execve(interpreter, [interpreter, str(Path(__file__).absolute()),
                  '--config', str(a.config.absolute()), '--rank', str(rank), '--child'], os.environ)
    import torch
    if torch.cuda.device_count() != 1:
        raise RuntimeError('One actual CUDA-visible GPU is required per worker')
    prop = torch.cuda.get_device_properties(0)
    witness = dict(rank=rank, task=task['label'], host=socket.gethostname(),
        job=os.environ.get('SLURM_JOB_ID'), step=os.environ.get('SLURM_STEP_ID'),
        local_rank=os.environ.get('SLURM_LOCALID'), incoming=os.environ.get('ECE285_INCOMING_GPU_VISIBILITY'),
        selected=os.environ.get('CUDA_VISIBLE_DEVICES'), cuda_devices=torch.cuda.device_count(),
        GPU_UUID=str(getattr(prop, 'uuid', 'UNAVAILABLE')), GPU_name=prop.name,
        interpreter=sys.executable, prefix=sys.prefix, config_sha256=hashlib.sha256(cfg_bytes).hexdigest(),
        started_unix=time.time(), argv=task['argv'], script=task['script'],
        entry_source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    destination = Path(cfg['witness_dir'])
    destination.mkdir(parents=True, exist_ok=True)
    path = destination/f"{witness['job']}_{witness['step']}_{rank}.json"
    with path.open('x') as stream:
        json.dump(witness, stream, indent=2)
        stream.write('\n')
    sys.path.insert(0, task['pythonpath'])
    sys.path.insert(0, str(Path(task['script']).parent))
    sys.argv = [task['script'], *task['argv']]
    runpy.run_path(task['script'], run_name='__main__')


if __name__ == '__main__':
    main()
