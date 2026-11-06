"""Complete diagnostics once the verified final scorer/evaluator is terminal."""
import subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent
while not (HERE/'reports/final/evaluation_receipt.json').exists():
 print('WAIT_FINAL_EVALUATION',flush=True);time.sleep(10)
for script in ['quality_summary.py','plot_primary.py','export_new_examples.py','finalize_science.py']:
 subprocess.run([sys.executable,str(HERE/script)],check=True)
print('FINAL_SCIENCE_PASS',flush=True)
