from pathlib import Path
import hashlib,json,subprocess,sys,time
f=Path(__file__).parent;c=f.parent/'causal_range';base=f.parent/'study_v2';expected="105e303a295811fa90898bc2bb2407c7d2c300c6a32cb9a362f2df17bcc2cf74"
while True:
 if time.time()>1794578068:raise TimeoutError('Goal deadline before causalrawreservation')
 try:
  raw=json.loads((c/'RAW_RESERVATION_READY.json').read_text());inp=json.loads((f/'input_boundary.json').read_text())
  if raw['status']=='CAUSAL_RAW_RESERVED_BEFORE_FRONTIER' and raw['boundary']==inp:break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(2)
assert hashlib.sha256(Path(inp['path']).read_bytes()).hexdigest()==inp['sha256']
assert hashlib.sha256((f/'frontier.py').read_bytes()).hexdigest()==expected,'Prospective frontier source changed'
(f/'start_gate_receipt.json').write_text(json.dumps(dict(status='CORE_READY_CAUSAL_RAW_PROTECTED_BEFORE_FRONTIER',created_unix=time.time(),frontier_source_sha256=expected,causal_raw_reservation_sha256=hashlib.sha256((c/'RAW_RESERVATION_READY.json').read_bytes()).hexdigest(),input_boundary=inp,core_completion_sha256=hashlib.sha256((base/'completion.json').read_bytes()).hexdigest(),core_analysis_sha256=hashlib.sha256((base/'analysis/receipt.json').read_bytes()).hexdigest()),indent=2)+'\n')
with (f/'run.log').open('x') as out:v=subprocess.run([sys.executable,str(f/'frontier.py')],stdout=out,stderr=subprocess.STDOUT)
(f/'driver_exit.json').write_text(json.dumps(dict(status='FRONTIER_DRIVER_EXITED',created_unix=time.time(),exit_code=v.returncode,source_sha256=expected),indent=2)+'\n')
if v.returncode==0:
 with (f/'analyze.log').open('x') as out:a=subprocess.run([sys.executable,str(f/'analyze.py')],stdout=out,stderr=subprocess.STDOUT)
 if a.returncode:raise SystemExit(a.returncode)
raise SystemExit(v.returncode)
