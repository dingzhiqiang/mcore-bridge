# Copyright (c) ModelScope Contributors. All rights reserved.
import torch
import torch.distributed as dist
from typing import Iterator, Tuple


def iter_ple_shard_ranges(total_rows: int, num_shards: int) -> Iterator[Tuple[int, int, int]]:
    """HF PLE shards use ceil-divided contiguous vocabulary rows."""
    if total_rows <= 0 or num_shards <= 0:
        raise ValueError('PLE total_rows and num_shards must be positive.')
    rows_per_shard = (total_rows + num_shards - 1) // num_shards
    for index in range(num_shards):
        start = min(index * rows_per_shard, total_rows)
        end = min(start + rows_per_shard, total_rows)
        yield index, start, end


def build_ple_shard_fragment(local_weight: torch.Tensor, vocab_start: int, vocab_end: int, shard_start: int,
                             shard_end: int) -> torch.Tensor:
    """Place this TP rank's owned rows into one disjoint full-shard buffer."""
    if local_weight.ndim != 2 or vocab_end - vocab_start != local_weight.shape[0]:
        raise ValueError('PLE local weight shape does not match its TP vocabulary interval.')
    if not 0 <= vocab_start <= vocab_end or not 0 <= shard_start <= shard_end:
        raise ValueError('PLE vocabulary and shard intervals must be non-negative and ordered.')
    if local_weight.dtype not in (torch.bfloat16, torch.float16, torch.float32, torch.float64):
        raise ValueError('Trainable PLE export requires dequantized floating-point parameters.')
    fragment = local_weight.new_zeros((shard_end - shard_start, local_weight.shape[1]))
    start, end = max(vocab_start, shard_start), min(vocab_end, shard_end)
    if start < end:
        fragment[start - shard_start:end - shard_start].copy_(local_weight[start - vocab_start:end - vocab_start])
    return fragment


@torch.no_grad()
def iter_ple_table_shards(local_weight: torch.Tensor, vocab_start: int, vocab_end: int, total_rows: int,
                          num_shards: int, tp_group: dist.ProcessGroup) -> Iterator[Tuple[int, torch.Tensor]]:
    """Gather current parameter values one HF shard at a time on every TP rank.

    No quantization scale from the source checkpoint is used. Besides the model
    weight, the generator owns at most B bytes, where
    B = ceil(total_rows / num_shards) * head_dim * local_weight.element_size().
    A yielded buffer belongs to the consumer and is never reused or overwritten.
    Call every TP rank in the owning PP stage in the same order.
    """
    for index, start, end in iter_ple_shard_ranges(total_rows, num_shards):
        shard = build_ple_shard_fragment(local_weight, vocab_start, vocab_end, start, end)
        if dist.get_world_size(tp_group) > 1 and shard.numel() > 0:
            # Each row has one owner, so SUM gathers without rounding trained values.
            dist.all_reduce(shard, group=tp_group)
        yield index, shard
        del shard
