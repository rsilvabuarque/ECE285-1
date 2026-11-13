from pathlib import Path
import hashlib,json,subprocess,sys,time
r=Path(__file__).parent;expected="a2ad403509eac76e60cf9a4ca0b50168ea4d63dfeeb6a9048efc9f2b587dbf61"
while True:
 if time.time()>1794578068:raise TimeoutError('Goal deadline before corecompletion')
 try:
  if json.loads((r/'completion.json').read_text()).get('status')=='TWO_ROUND_ADAPTIVE_STUDY_COMPLETE':break
 except (FileNotFoundError,json.JSONDecodeError):pass
 time.sleep(2)
assert hashlib.sha256((r/'analyze.py').read_bytes()).hexdigest()==expected,'Frozen derived-analysis source changed'
with (r/'analyze.log').open('x') as f:v=subprocess.run([sys.executable,str(r/'analyze.py')],stdout=f,stderr=subprocess.STDOUT)
(r/'core_analysis_driver_exit.json').write_text(json.dumps(dict(status='CORE_ANALYSIS_DRIVER_EXITED',created_unix=time.time(),exit_code=v.returncode,source_sha256=expected,completion_sha256=hashlib.sha256((r/'completion.json').read_bytes()).hexdigest()),indent=2)+'\n')
raise SystemExit(v.returncode)
