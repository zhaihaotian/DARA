"""Run one four-GPU experiment inside two existing two-GPU Slurm allocations."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import re
import signal
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def allocation(job_id):
    text = subprocess.check_output(['scontrol', 'show', 'job', '-o', str(job_id)], text=True)
    fields = dict(re.findall(r'(\S+?)=(\S+)', text))
    resources = dict(part.split('=', 1) for part in fields['AllocTRES'].split(','))
    gpu_types = [(key.split(':', 1)[1], int(value)) for key, value in resources.items()
                 if key.startswith('gres/gpu:')]
    if fields['JobState'] != 'RUNNING' or fields['NumNodes'] != '1' or len(gpu_types) != 1:
        raise ValueError(f'Allocation {job_id} must be running on one node with typed GPUs')
    gpu_type, count = gpu_types[0]
    if count != 2 or int(resources['gres/gpu']) != 2:
        raise ValueError(f'Allocation {job_id} must contain exactly two GPUs')
    return dict(id=str(job_id), node=fields['NodeList'], gpu_type=gpu_type,
                end=datetime.fromisoformat(fields['EndTime']).timestamp())


def step_command(job, plan, role):
    return ['srun', '--jobid='+job['id'], '--exclusive', '--exact', '-N1', '-n1', '-c8',
            '--mem=96G', f'--gres=gpu:{job["gpu_type"]}:2',
            sys.executable, '-u', str(REPO/'training/ray_node.py'),
            '--plan', str(plan), '--role', role, '--allocation', job['id']]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--jobs', nargs=2, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--check-only', action='store_true', help='Validate four-GPU NCCL communication')
    parser.add_argument('--minimum-seconds', type=int, default=None)
    args, training_args = parser.parse_known_args()
    jobs = [allocation(job_id) for job_id in args.jobs]
    if jobs[0]['id'] == jobs[1]['id'] or jobs[0]['gpu_type'] != jobs[1]['gpu_type']:
        raise ValueError('Pair two distinct allocations of the same GPU type')
    required = args.minimum_seconds if args.minimum_seconds is not None else (600 if args.check_only else 19800)
    if min(job['end'] for job in jobs)-time.time() < required:
        raise ValueError(f'Both allocations need at least {required} seconds remaining')
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    plan_path = output/'pair.json'
    if plan_path.exists() or (output/'metrics.jsonl').exists():
        raise FileExistsError(f'Existing pair run: {output}; use a new output directory')
    if args.check_only:
        environment = dict(PYTHONNOUSERSITE='1', PYTHONDONTWRITEBYTECODE='1',
                           OMP_NUM_THREADS='4', RAY_USAGE_STATS_ENABLED='0')
    else:
        command = [sys.executable, str(REPO/'training/launch.py'), *training_args,
                   '--output', str(output), '--nnodes', '2', '--ray-address', 'pending:0', '--dry-run']
        environment = json.loads(subprocess.check_output(command, text=True))['environment']
        environment.pop('RAY_ADDRESS')
    environment['PYTHONPATH'] = str(REPO/'vendor/verl') + (
        os.pathsep+os.environ['PYTHONPATH'] if os.environ.get('PYTHONPATH') else '')
    plan = dict(jobs=jobs, output=str(output), check_only=args.check_only,
                training_args=training_args, environment=environment,
                usable_until=min(job['end'] for job in jobs))
    plan_path.write_text(json.dumps(plan, indent=2)+'\n')
    processes = []
    logs = []
    code = 1

    def terminate(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        for job, role in zip(jobs, ('head', 'worker')):
            log = (output/f'{role}.log').open('w')
            logs.append(log)
            processes.append(subprocess.Popen(step_command(job, plan_path, role),
                                              stdout=log, stderr=subprocess.STDOUT))
        while processes[0].poll() is None:
            if processes[1].poll() is not None and not (output/'driver.exit').exists():
                raise RuntimeError('Worker allocation ended before the training driver')
            time.sleep(2)
        code = processes[0].returncode
        if code == 0:
            worker_code = processes[1].wait(timeout=60)
            code = worker_code if worker_code else 0
    finally:
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
        for log in logs:
            log.close()
        (output/'exit.code').write_text(str(code)+'\n')
    return code


if __name__ == '__main__':
    sys.exit(main())
