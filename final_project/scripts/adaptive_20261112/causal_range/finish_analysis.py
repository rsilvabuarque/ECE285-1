from pathlib import Path
import json,subprocess,sys,time
c=Path(__file__).parent
while True:
 if time.time()>1794578068:raise TimeoutError('Goal deadline')
 try:
  if json.loads((c/'completion.json').read_text())['status']=='POSTHOC_CAUSAL_FOLLOWUP_COMPLETE':break
 except (FileNotFoundError,json.JSONDecodeError):pass
 if (c/'failure.json').exists():raise RuntimeError('Causal bounded followup failed; preserve partial outputs')
 time.sleep(2)
with (c/'analyze.log').open('x') as f:v=subprocess.run([sys.executable,str(c/'analyze.py')],stdout=f,stderr=subprocess.STDOUT)
raise SystemExit(v.returncode)
