"""Replay fixed PPO updates on two or four ranks and record gradients and parameters."""
import argparse
import json
import os
from pathlib import Path
import sys
from types import MethodType
from unittest.mock import patch

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO/'vendor/verl'))

import torch
from torch import distributed as dist
import torch.multiprocessing as mp
from tensordict import TensorDict
from omegaconf import OmegaConf

from verl import DataProto
from verl.workers.actor.dp_actor import DataParallelPPOActor
from verl.utils.seqlen_balancing import get_seqlen_balanced_partitions


def fixed_batch():
    count, length, response = 128, 24, 12
    ids = torch.arange(count)
    positions = torch.arange(length).expand(count, -1).clone()
    mask = (positions < (13 + ids % 12).unsqueeze(-1)).long()
    advantages = ((ids % 7).float() - 3).unsqueeze(-1).expand(-1, response).clone() / 3
    batch = DataProto.from_dict(tensors=dict(
        input_ids=ids.unsqueeze(-1).expand(-1, length).clone(), position_ids=positions,
        attention_mask=mask, responses=torch.zeros(count, response, dtype=torch.long),
        advantages=advantages, old_log_probs=torch.zeros(count, response),
        gd2po_query_weight=(ids % 5).float() / 4), meta_info={'temperature': 1.})
    partitions = get_seqlen_balanced_partitions(mask.sum(-1).tolist(), 4, equal_size=True)
    batch.reorder(torch.tensor([index for part in partitions for index in part]))
    return batch


def worker(rank, world, device_kind, rendezvous, output):
    torch.set_num_threads(1)
    device = torch.device('cuda', rank) if device_kind == 'cuda' else torch.device('cpu')
    if device_kind == 'cuda':
        torch.cuda.set_device(device)
    dist.init_process_group('nccl' if device_kind == 'cuda' else 'gloo',
                            init_method=rendezvous, rank=rank, world_size=world)
    torch.manual_seed(123)
    module = torch.nn.Linear(3, 2).to(device)
    with torch.no_grad():
        for param in module.parameters():
            param.mul_(.01)
    if device_kind == 'cuda':
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        module = FSDP(module, device_id=device, use_orig_params=True)
    actor = object.__new__(DataParallelPPOActor)
    actor.config = OmegaConf.create(dict(ppo_mini_batch_size=8, ppo_micro_batch_size=4,
        logical_shards=4//world, use_dynamic_bsz=True, ppo_max_token_len_per_gpu=48,
        use_kl_loss=False, clip_ratio=.2, entropy_coeff=.001, grad_clip=.2))
    actor.ulysses_sequence_parallel_size = 1
    actor.actor_module = module
    actor.actor_optimizer = torch.optim.AdamW(module.parameters(), lr=.002, weight_decay=.01)
    records, losses, microbatches = [], [], []
    original_backward = torch.Tensor.backward

    def backward(tensor, *args, **kwargs):
        losses.append(float(tensor.detach()))
        return original_backward(tensor, *args, **kwargs)

    def forward(self, micro_batch, temperature):
        ids = micro_batch['input_ids'][:, 0]
        microbatches.append(ids.cpu().tolist())
        features = torch.stack((ids.float()/128, (ids % 7).float()/7, (ids % 11).float()/11), -1)
        values = self.actor_module(features)
        n = micro_batch['responses'].shape[-1]
        return values[:, 1:].sigmoid().expand(-1, n), values[:, :1].expand(-1, n)

    def optimizer_step(self):
        params = list(self.actor_module.parameters())
        if device_kind == 'cpu':
            for param in params:
                dist.all_reduce(param.grad)
                param.grad.div_(world)
        loss = torch.tensor(sum(losses), device=device)
        dist.all_reduce(loss)
        loss.div_(world)
        if device_kind == 'cuda':
            with FSDP.summon_full_params(self.actor_module, with_grads=True, writeback=False):
                gradient = torch.cat([p.grad.flatten() for p in self.actor_module.parameters()]).cpu().tolist()
        else:
            gradient = torch.cat([p.grad.flatten() for p in params]).cpu().tolist()
        norm = DataParallelPPOActor._optimizer_step(self)
        if device_kind == 'cuda':
            with FSDP.summon_full_params(self.actor_module, writeback=False):
                updated = torch.cat([p.detach().flatten() for p in self.actor_module.parameters()]).cpu().tolist()
        else:
            updated = torch.cat([p.detach().flatten() for p in params]).cpu().tolist()
        all_micro = [None] * world
        dist.all_gather_object(all_micro, list(microbatches))
        records.append(dict(loss=loss.item(), gradient=gradient, norm=norm.item(), parameters=updated,
                            microbatches=all_micro))
        losses.clear()
        microbatches.clear()
        return norm

    actor._forward_micro_batch = MethodType(forward, actor)
    actor._optimizer_step = MethodType(optimizer_step, actor)
    batch = fixed_batch().chunk(world)[rank].to(device)
    with patch.object(torch.Tensor, 'backward', backward):
        if device_kind == 'cpu':
            with patch.object(TensorDict, 'cuda', lambda self, *a, **kw: self):
                actor.update_policy(batch)
        else:
            actor.update_policy(batch)
    if rank == 0:
        Path(output).write_text(json.dumps(dict(world_size=world, logical_world_size=4,
            device=device_kind, updates=records), indent=2)+'\n')
    dist.destroy_process_group()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--world-size', type=int, choices=[2, 4], required=True)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--reference', type=Path, help='Compare all updates against a four-rank replay')
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    rendezvous = args.output.parent/f'gloo-{os.getpid()}'
    mp.spawn(worker, args=(args.world_size, args.device, rendezvous.resolve().as_uri(), str(args.output)),
             nprocs=args.world_size, join=True)
    if args.reference:
        compare(json.loads(args.reference.read_text()), json.loads(args.output.read_text()))
        print('PASS: logical batches, loss, gradients, clipping and optimizer updates match four ranks.', flush=True)


def compare(reference, candidate):
    if len(reference['updates']) != len(candidate['updates']):
        raise AssertionError('Optimizer update counts differ')
    differences = {}
    for left, right in zip(reference['updates'], candidate['updates']):
        for field in ('loss', 'gradient', 'norm', 'parameters'):
            a, b = torch.tensor(left[field]), torch.tensor(right[field])
            torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-6)
            differences[field] = max(differences.get(field, 0.), (a-b).abs().max().item())
        logical = []
        shards = 4 // candidate['world_size']
        for physical in right['microbatches']:
            stride = len(physical) // shards
            logical.extend(physical[start:start+stride] for start in range(0, len(physical), stride))
        if logical != left['microbatches']:
            raise AssertionError('Logical microbatch membership or order differs')
    print(json.dumps(dict(max_absolute_differences=differences)), flush=True)


if __name__ == '__main__':
    main()
