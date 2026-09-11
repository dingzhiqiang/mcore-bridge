# Copyright (c) ModelScope Contributors. All rights reserved.
"""NCCL integration for current-value PLE export, including non-head PP owners.

Run separately on two GPUs with --tp 2 --pp 1 and --tp 1 --pp 2; optionally
four GPUs with --tp 2 --pp 2. A writable --output-dir is required for save IO.
The tiny table receives an Adam step. Its real PLE export code is unchanged;
ordinary transformer-weight conversion is disabled to avoid constructing layers.
"""
import argparse
import os
import torch
import torch.distributed as dist
from megatron.core import parallel_state as mpu
from pathlib import Path
from safetensors import safe_open
from safetensors.torch import save_file
from torch import nn
from types import SimpleNamespace

from mcore_bridge.model.gpts.qwen4_exp import Qwen4ExpBridge
from mcore_bridge.model.modules.ple import Qwen4ExpTextNGramEmbedding

ROWS, WIDTH, PARTS = 24, 4, 5


class ToyTable(Qwen4ExpTextNGramEmbedding):

    def __init__(self, layer_number, device):
        nn.Module.__init__(self)
        self.cpu_offload = False
        self.padded_vocab_size = ROWS
        self.split_ngram_parts = PARTS
        start = mpu.get_tensor_model_parallel_rank() * ROWS // mpu.get_tensor_model_parallel_world_size()
        end = start + ROWS // mpu.get_tensor_model_parallel_world_size()
        self.ngram_embedding = nn.Module()
        self.ngram_embedding.vocab_start_index = start
        self.ngram_embedding.vocab_end_index = end
        self.ngram_embedding.weight = nn.Parameter(reference_weight(layer_number, device)[start:end].clone())
        # Deliberately stale. Current-value export must never consult this scale.
        self._ngram_weight_scale = torch.tensor(0.03125, device=device)
        optimizer = torch.optim.Adam([self.ngram_embedding.weight], lr=0.0078125, weight_decay=0)
        self.ngram_embedding.weight.grad = torch.ones_like(self.ngram_embedding.weight)
        optimizer.step()


def reference_weight(layer_number, device):
    rows = torch.arange(ROWS * WIDTH, device=device).reshape(ROWS, WIDTH).remainder(17)
    return rows.to(torch.bfloat16) * 0.03125 + layer_number * 0.5


def make_model(config, device):
    model = nn.Module()
    model.config = config
    model.vp_stage = None
    model.decoder = nn.Module()
    model.decoder.layers = nn.ModuleList()
    pp_rank, pp_size = mpu.get_pipeline_model_parallel_rank(), mpu.get_pipeline_model_parallel_world_size()
    for number in range(1, config.num_layers + 1):
        if (number - 1) % pp_size != pp_rank:
            continue
        layer = nn.Module()
        layer.layer_number = number
        layer.ple = nn.Module()
        layer.ple.ple_embedding = ToyTable(number, device)
        model.decoder.layers.append(layer)
    return model


def make_bridge(config):
    bridge = object.__new__(Qwen4ExpBridge)
    bridge.config = config
    bridge.is_multimodal = False
    bridge.tp_group = mpu.get_tensor_model_parallel_group()
    bridge.pp_group = mpu.get_pipeline_model_parallel_group()
    bridge.tp_size = mpu.get_tensor_model_parallel_world_size()
    bridge.tp_rank = mpu.get_tensor_model_parallel_rank()
    bridge.pp_size = mpu.get_pipeline_model_parallel_world_size()
    bridge.pp_rank = mpu.get_pipeline_model_parallel_rank()
    bridge._convert_pre_process = lambda *args: {}
    bridge._convert_post_process = lambda *args: {}
    bridge._set_layer_state = lambda *args: {}
    return bridge


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tp', type=int, required=True)
    parser.add_argument('--pp', type=int, required=True)
    parser.add_argument('--output-dir', required=True)
    args = parser.parse_args()
    torch.cuda.set_device(int(os.environ['LOCAL_RANK']))
    dist.init_process_group('nccl')
    assert dist.get_world_size() == args.tp * args.pp
    mpu.initialize_model_parallel(tensor_model_parallel_size=args.tp, pipeline_model_parallel_size=args.pp)
    device = torch.device('cuda', torch.cuda.current_device())
    config = SimpleNamespace(num_layers=2, ple_layer_ids=[1, 2], split_ngram_parts=PARTS, mtp_num_layers=None)
    bridge = make_bridge(config)
    model = make_model(config, device)
    for only_master_rank in (False, True):
        exported = {}
        keys = []
        for name, tensor in bridge.export_weights([model], only_master_rank=only_master_rank):
            keys.append(name)
            if only_master_rank and dist.get_rank() != 0:
                assert tensor is None
            else:
                assert tensor.dtype == torch.bfloat16
                assert tensor.device.type == 'cuda'
                exported[name] = tensor.cpu()
        all_keys = [None] * dist.get_world_size()
        dist.all_gather_object(all_keys, keys)
        assert all(item == keys for item in all_keys)
        assert len(keys) == 2 * PARTS and all('weight_scale' not in key for key in keys)
        if exported:
            for layer_number in (1, 2):
                prefix = f'model.layers.{layer_number - 1}.ple.ple_embedding.ngram_embedding.'
                actual = torch.cat([exported[f'{prefix}shard_{i}.weight'] for i in range(PARTS)])
                expected = reference_weight(layer_number, 'cpu') - 0.0078125
                torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    source_dir = Path(args.output_dir) / 'source'
    output_dir = Path(args.output_dir) / 'exported'
    if dist.get_rank() == 0:
        source_dir.mkdir(parents=True, exist_ok=True)
        source = {'model.visual.weight': torch.tensor([2.0])}
        for layer_number in (1, 2):
            prefix = f'model.layers.{layer_number - 1}.ple.ple_embedding.ngram_embedding.'
            source[f'{prefix}weight_scale'] = torch.tensor(0.03125)
            shard_rows = (ROWS + PARTS - 1) // PARTS
            for index in range(PARTS):
                source[f'{prefix}shard_{index}.weight'] = reference_weight(
                    layer_number, 'cpu')[index * shard_rows:(index + 1) * shard_rows].clone()
        save_file(source, str(source_dir / 'model.safetensors'))
    dist.barrier()
    bridge.save_weights([model], str(output_dir), max_shard_size='1GB', save_missing_weights=str(source_dir))
    if dist.get_rank() == 0:
        saved = {}
        for path in output_dir.glob('model*.safetensors'):
            with safe_open(path, framework='pt', device='cpu') as handle:
                saved.update({key: handle.get_tensor(key) for key in handle.keys()})
        assert len(saved) == 2 * PARTS + 1
        assert not any('weight_scale' in key for key in saved)
        torch.testing.assert_close(saved['model.visual.weight'], torch.tensor([2.0]), atol=0, rtol=0)
        for layer_number in (1, 2):
            prefix = f'model.layers.{layer_number - 1}.ple.ple_embedding.ngram_embedding.'
            actual = torch.cat([saved[f'{prefix}shard_{i}.weight'] for i in range(PARTS)])
            torch.testing.assert_close(actual, reference_weight(layer_number, 'cpu') - 0.0078125, atol=0, rtol=0)
        print(f'PLE stream export PASS: TP={args.tp} PP={args.pp}, live/save current BF16, Adam no-WD.')
    dist.barrier()
    mpu.destroy_model_parallel()
    dist.destroy_process_group()


if __name__ == '__main__':
    main()
