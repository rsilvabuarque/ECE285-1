"""Append immutable fresh-candidate shards to the resident oracle queues."""
import argparse,hashlib,json,os
from pathlib import Path
import numpy as np
HERE=Path(__file__).resolve().parent
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--prefix',required=True);a=ap.parse_args()
 path=Path(a.prefix+'_fresh.npz').resolve();digest=hashlib.sha256(path.read_bytes()).hexdigest()
 with np.load(path) as z:N=len(z['structure_ids'])
 queue=HERE/'data/score_tasks.jsonl';old=[json.loads(s) for s in queue.read_text().splitlines() if s] if queue.exists() else []
 tag=Path(a.prefix).name;tasks=[]
 for model in ['uma','mace']:
  for start in range(0,N,64):tasks.append({'task_id':f'{tag}_{model}_{start:06d}','model':model,'pool':tag,'input_npz':str(path),'input_sha256':digest,'start':start,'stop':min(start+64,N)})
 assert not ({r['task_id'] for r in tasks}&{r['task_id'] for r in old})
 tmp=queue.with_suffix('.tmp');tmp.write_text(''.join(json.dumps(t)+'\n' for t in old+tasks));tmp.replace(queue)
 print('ENQUEUED',tag,N,len(tasks),flush=True)
if __name__=='__main__':main()
