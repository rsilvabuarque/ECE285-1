"""Record owned top-level allocations without counting their child steps twice.

For the verified Perlmutter GPU policy, a full node has billing=128. Shared
allocations use the recorded billing fraction; QoS/architecture factors are one
for this campaign. Running-job costs remain estimates until their final sacct
records arrive. Private project balances are deliberately separate.
"""
import argparse
import csv
from datetime import datetime, timezone
import io
import json
import os
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--jobs', nargs='+', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--private-balance', type=Path)
    a = p.parse_args()
    env = dict(os.environ, TZ='UTC')
    command = ['sacct', '-X', '-j', ','.join(a.jobs), '--units=M', '-P',
        '--format=JobID,Account,QOS,State,Start,End,ElapsedRaw,AllocNodes,AllocTRES']
    result = subprocess.check_output(command, text=True, env=env)
    rows = list(csv.DictReader(io.StringIO(result), delimiter='|'))
    if {r['JobID'] for r in rows} != set(a.jobs):
        raise ValueError('Missing or unexpected top-level allocation')
    totals = dict(charged_GPU_node_hours=0., physical_node_allocation_hours=0.,
                  allocated_GPU_device_hours=0.)
    for row in rows:
        if row['Account'] != 'm5397_g':
            raise ValueError('Unexpected charged account')
        tres = dict(item.split('=', 1) for item in row['AllocTRES'].split(','))
        elapsed = int(row['ElapsedRaw']) / 3600
        row.update(physical_node_allocation_hours=elapsed * int(row['AllocNodes']),
            allocated_GPU_device_hours=elapsed * int(tres['gres/gpu']),
            charged_GPU_node_hours=elapsed * float(tres['billing']) / 128,
            terminal=row['End'] != 'Unknown')
        for key in totals:
            totals[key] += row[key]
    output = dict(created_UTC=datetime.now(timezone.utc).isoformat(),
        status='FINAL' if all(r['terminal'] for r in rows) else 'RUNNING_ESTIMATE',
        command=command, scheduler_date_timezone='UTC', allocations=rows, totals=totals,
        hard_charged_node_hour_cap=250, policy_url='https://docs.nersc.gov/jobs/policy/',
        factors=dict(Perlmutter=1, interactive=1, shared_interactive=1),
        interpretation='Top-level allocated billing includes startup, idle time and failures; child steps excluded. GPU-device hours are allocated capacity, not measured utilization.')
    a.output.parent.mkdir(parents=True, exist_ok=True)
    if a.output.exists():
        raise FileExistsError('Preserve earlier resource receipt; use a new output name')
    a.output.write_text(json.dumps(output, indent=2) + '\n')
    if a.private_balance:
        balance = subprocess.check_output(['iris', 'project', 'm5397', '--json'], text=True)
        with a.private_balance.open('x') as f:
            f.write(balance)
    print(json.dumps(dict(status=output['status'], totals=totals)))


if __name__ == '__main__':
    main()
