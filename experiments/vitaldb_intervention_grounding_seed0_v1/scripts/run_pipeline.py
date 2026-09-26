"""Run after preparation, or wait for the known preparation PID to finish."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(__file__).resolve().parents[1]
parser=argparse.ArgumentParser(); parser.add_argument('--wait-data',action='store_true'); args=parser.parse_args()
os.chdir(root)
if args.wait_data:
    while not (root/'outputs/data_summary.json').exists(): time.sleep(20)
for name in ['checks','train','evaluate','large_change_audit','report']:
    print('STAGE_START',name,flush=True)
    with open(root/f'logs/{name}.log','w') as log:
        result=subprocess.run([sys.executable,'-u',f'src/{name}.py'],stdout=log,stderr=subprocess.STDOUT)
    (root/'outputs/run_status.json').write_text(json.dumps({'stage':name,'returncode':result.returncode,'time':time.time()},indent=2))
    if result.returncode: raise SystemExit(f'Stage failed: {name}; see logs/{name}.log')
    print('STAGE_COMPLETE',name,flush=True)
print('ALL_COMPUTATION_COMPLETE; conclusion awaits evidence review',flush=True)
