"""Run successive scientific stages on the same scheduler-assigned GPU slot."""
import subprocess,sys
from pathlib import Path
HERE=Path(__file__).resolve().parent
dispatch={'md_gate':'md_dispatch.py','generated':'score_dispatch.py'}
for stage in sys.argv[1:]:subprocess.run([sys.executable,str(HERE/dispatch[stage]),'--stage',stage],check=True)
