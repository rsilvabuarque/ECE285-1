"""Sequential queue, collection, and metric evaluation inside the allocation."""
import argparse,subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
ap=argparse.ArgumentParser();ap.add_argument('--pool',required=True);ap.add_argument('--known-keys',required=True);a=ap.parse_args();prefix=HERE/'data'/a.pool
def run(script,args):subprocess.run([sys.executable,str(HERE/script),*map(str,args)],check=True)
run('enqueue.py',['--prefix',prefix])
run('collect.py',['--prefix',prefix,'--wait'])
run('evaluate.py',['--proposals',str(prefix)+'_attempts.csv','--scores','uma='+str(prefix)+'_scores_uma.npz','mace='+str(prefix)+'_scores_mace.npz','--known-keys',a.known_keys,'--pool',a.pool,'--output-dir',HERE/'reports'/a.pool])
