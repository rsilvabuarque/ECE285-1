"""Execute a frozen fit list, one GPU per Slurm task, with durable failures.

Every task states its full command before fitting. Claim directories prevent
two ranks from starting the same trajectory. No scheduler is invoked here.
"""
import argparse
import contextlib
import hashlib
import json
import os
from pathlib import Path
import sys
import time
import traceback


def main():
    p=argparse.ArgumentParser();p.add_argument('--tasks',type=Path,required=True)
    a=p.parse_args();config=json.loads(a.tasks.read_text())
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG',':4096:8')
    os.environ.setdefault('OMP_NUM_THREADS','4')
    from cathode_cvae.learning.train_mask import main as train
    from cathode_cvae.runtime import require_compute_allocation
    require_compute_allocation()
    claims=a.tasks.with_suffix('.claims');claims.mkdir(exist_ok=True)
    for task in config['tasks']:
        claim=claims/task['id']
        try:claim.mkdir()
        except FileExistsError:continue
        start=time.time()
        owner=dict(pid=os.getpid(),host=os.uname().nodename,
            job=os.environ.get('SLURM_JOB_ID'),rank=os.environ.get('SLURM_PROCID'),
            cuda_visible=os.environ.get('CUDA_VISIBLE_DEVICES'),started_unix=start,
            tasks_sha256=hashlib.sha256(a.tasks.read_bytes()).hexdigest(),
            interpreter=sys.executable,argv=task['argv'])
        (claim/'owner.json').write_text(json.dumps(owner,indent=2)+'\n')
        status='COMPLETE';error=None
        with (claim/'log.txt').open('w',buffering=1) as log:
            with contextlib.redirect_stdout(log),contextlib.redirect_stderr(log):
                try:train(task['argv'])
                except BaseException as e:
                    status='FAILED';error=f'{type(e).__name__}: {e}';traceback.print_exc()
        receipt=dict(owner,status=status,error=error,finished_unix=time.time(),
            elapsed_seconds=time.time()-start)
        (claim/'receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
        print(json.dumps(dict(task=task['id'],status=status,seconds=receipt['elapsed_seconds'],error=error)),flush=True)
    receipts=[json.loads(f.read_text()) for f in claims.glob('*/receipt.json')]
    print(json.dumps(dict(worker_finished=True,completed_receipts=len(receipts))),flush=True)


if __name__=='__main__':main()
