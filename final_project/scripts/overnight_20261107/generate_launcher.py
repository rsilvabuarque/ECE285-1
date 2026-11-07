import os,subprocess,sys
from pathlib import Path
rank=int(os.environ.get('SLURM_PROCID','0'));root=Path.cwd();reps=['0','2'] if rank==0 else ['1'];raise SystemExit(subprocess.call([sys.executable,str(root/'scripts/overnight_20261107/generate_benchmark.py'),'--root',str(root),'--replicates',*reps]))
