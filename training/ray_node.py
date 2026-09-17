"""Start one Ray node inside its allocated Slurm step; run the driver on the head."""
import argparse
import json
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

REPO = Path(__file__).resolve().parents[1]


def free_port():
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


def communication_check(ray, nodes, address):
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=1, num_gpus=1)
    class Rank:
        def check(self, rank, endpoint):
            from datetime import timedelta
            import torch
            import torch.distributed as dist
            torch.cuda.set_device(0)
            dist.init_process_group('nccl', init_method=endpoint, rank=rank,
                                    world_size=4, timeout=timedelta(seconds=120))
            value = torch.tensor([rank+1.0], device='cuda')
            dist.all_reduce(value)
            total = value.item()
            dist.destroy_process_group()
            if total != 10:
                raise RuntimeError(f'Four-rank all-reduce returned {total}')
            return dict(rank=rank, node=socket.gethostname(), gpu=ray.get_gpu_ids(),
                        device=torch.cuda.get_device_name(0), all_reduce_sum=total)

    endpoint = 'tcp://'+address.split(':')[0]+':'+str(free_port())
    actors = []
    try:
        for node in nodes:
            for _ in range(2):
                actors.append(Rank.options(scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node['NodeID'], soft=False)).remote())
        return ray.get([actor.check.remote(rank, endpoint)
                        for rank, actor in enumerate(actors)], timeout=240)
    finally:
        for actor in actors:
            ray.kill(actor)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--role', choices=['head', 'worker'], required=True)
    parser.add_argument('--allocation', required=True)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text())
    output = Path(plan['output'])
    os.environ.update(plan['environment'])
    os.environ.pop('RAY_ADDRESS', None)
    os.environ['RAY_TMPDIR'] = f'/tmp/p{os.getuid()}j{args.allocation}'
    ip = socket.gethostbyname(socket.gethostname())
    ray_command = [str(Path(sys.executable).with_name('ray')), 'start',
                   '--node-ip-address='+ip, '--num-cpus=8', '--num-gpus=2',
                   '--object-store-memory=8589934592', '--memory=68719476736',
                   '--temp-dir='+os.environ['RAY_TMPDIR'], '--dashboard-agent-listen-port=0',
                   '--node-manager-port=0', '--object-manager-port=0',
                   '--min-worker-port=0', '--max-worker-port=0',
                   '--ray-client-server-port=0', '--block']
    address_file = output/'ray_address.json'
    if args.role == 'head':
        port = free_port()
        address = f'{ip}:{port}'
        ray_command += ['--head', '--port='+str(port), '--include-dashboard=false']
    else:
        deadline = time.time()+240
        while not address_file.exists():
            if time.time() >= deadline:
                raise TimeoutError('Ray head address was not published')
            time.sleep(2)
        address = json.loads(address_file.read_text())['address']
        ray_command += ['--address='+address]

    def terminate(signum, frame):
        raise SystemExit(128+signum)

    signal.signal(signal.SIGTERM, terminate)
    process = subprocess.Popen(ray_command, start_new_session=True)
    code = 1
    try:
        if args.role == 'head':
            deadline = time.time()+240
            while True:
                if process.poll() is not None:
                    raise RuntimeError('Ray head exited during startup')
                try:
                    with socket.create_connection((ip, port), timeout=2):
                        break
                except OSError:
                    if time.time() >= deadline:
                        raise TimeoutError('Ray head startup timed out')
                    time.sleep(2)
            address_file.write_text(json.dumps(dict(address=address))+'\n')
            import ray
            ray.init(address=address)
            deadline = time.time()+240
            while True:
                nodes = [node for node in ray.nodes() if node['Alive']]
                if len(nodes) == 2 and all(node['Resources'].get('GPU') == 2 for node in nodes):
                    break
                if process.poll() is not None or time.time() >= deadline:
                    raise RuntimeError('Ray did not register two nodes with two GPUs each')
                time.sleep(2)
            results = communication_check(ray, nodes, address)
            (output/'communication.json').write_text(json.dumps(results, indent=2)+'\n')
            ray.shutdown()
            if plan['check_only']:
                code = 0
            else:
                command = [sys.executable, '-u', str(REPO/'training/launch.py'),
                           *plan['training_args'], '--output', str(output),
                           '--nnodes', '2', '--ray-address', address]
                code = subprocess.call(command)
        else:
            while not (output/'driver.exit').exists():
                if process.poll() is not None:
                    raise RuntimeError('Ray worker exited while the driver was active')
                time.sleep(2)
            code = int((output/'driver.exit').read_text())
    finally:
        if args.role == 'head':
            (output/'driver.exit').write_text(str(code)+'\n')
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=15)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
    return code


if __name__ == '__main__':
    sys.exit(main())
