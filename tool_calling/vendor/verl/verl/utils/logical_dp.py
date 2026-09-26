"""Preserve four data-parallel shards while two physical ranks execute them."""
from contextlib import contextmanager
import copy
import random

import numpy as np
import torch
from torch import distributed as dist

from verl.utils.seqlen_balancing import ceildiv, rearrange_micro_batches


def logical_micro_batches(batch, shards, max_token_len):
    """Use the maximum microbatch count over all logical ranks, as with four GPUs."""
    pieces = batch.chunk(shards)
    count = max(ceildiv(piece['attention_mask'].sum().item(), max_token_len) for piece in pieces)
    if dist.is_initialized():
        value = torch.tensor([count], device=batch['attention_mask'].device)
        dist.all_reduce(value, op=dist.ReduceOp.MAX)
        count = value.item()
    batches, indices = [], []
    offset = 0
    for piece in pieces:
        micro, order = rearrange_micro_batches(piece, max_token_len, num_micro_batches=count)
        batches.extend(micro)
        indices.extend([[offset + index for index in part] for part in order])
        offset += len(piece)
    return batches, indices


def logical_minibatches(batch, shards, minibatch_size):
    """Align optimizer update i from each local logical shard before moving to i+1."""
    chunks = [piece.split(minibatch_size) for piece in batch.chunk(shards)]
    return [torch.cat(parts) for parts in zip(*chunks)]


class RolloutStreams:
    """Time-multiplex the original per-rank RNG streams on one TP1 vLLM engine."""

    def __init__(self, engine, first_logical_rank, shards):
        self.engine = engine
        self.counters = [engine.request_counter, engine.llm_engine.seq_counter]
        self.states = []
        state = torch.cuda.get_rng_state()
        for rank in range(first_logical_rank, first_logical_rank + shards):
            torch.cuda.manual_seed(rank + 1000)
            self.states.append(dict(cuda=torch.cuda.get_rng_state(), cpu=torch.get_rng_state(),
                                    numpy=copy.deepcopy(np.random.get_state()), python=random.getstate(),
                                    counters=[counter.counter for counter in self.counters]))
        torch.cuda.set_rng_state(state)

    @contextmanager
    def use(self, index):
        state = self.states[index]
        outer = (torch.cuda.get_rng_state(), torch.get_rng_state(), np.random.get_state(), random.getstate())
        torch.cuda.set_rng_state(state['cuda'])
        torch.set_rng_state(state['cpu'])
        np.random.set_state(state['numpy'])
        random.setstate(state['python'])
        for counter, value in zip(self.counters, state['counters']):
            counter.counter = value
        try:
            yield
        finally:
            state.update(cuda=torch.cuda.get_rng_state(), cpu=torch.get_rng_state(),
                         numpy=np.random.get_state(), python=random.getstate(),
                         counters=[counter.counter for counter in self.counters])
            torch.cuda.set_rng_state(outer[0])
            torch.set_rng_state(outer[1])
            np.random.set_state(outer[2])
            random.setstate(outer[3])
