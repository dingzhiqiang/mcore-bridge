"""PLE replica gradients must match a full-sequence reference after TP reduction."""
import os
import pytest
import torch
from types import SimpleNamespace


def _make_ple(dtype, tp_group):
    from megatron.core.transformer.transformer_config import TransformerConfig

    from mcore_bridge.model.modules.ple import Qwen4ExpTextPLELayer

    config = TransformerConfig(
        num_layers=1,
        hidden_size=64,
        num_attention_heads=4,
        params_dtype=dtype,
        pipeline_dtype=dtype,
        gradient_accumulation_fusion=False,
        sequence_parallel=True,
        tensor_model_parallel_size=2)
    for name, value in {
            'hc_count': 2,
            'ple_embed_dim': 32,
            'ple_conv_kernel_size': 4,
            'ngram_size': 3,
            'heads_per_ngram': 2,
            'eos_token_id': 0,
            'split_ngram_parts': 1,
            'ple_seed': 1234,
            'padded_vocab_size': 128,
            'ngram_vocab_size_base': 131,
            'make_ngram_vocab_size_divisible_by': 8,
    }.items():
        setattr(config, name, value)
    config.init_method = lambda tensor: torch.nn.init.normal_(tensor, std=0.02)
    model = Qwen4ExpTextPLELayer(config, 0, pg_collection=SimpleNamespace(tp=tp_group)).cuda()
    with torch.no_grad():
        model.conv1d.weight.normal_(0, 0.03)
    model.ple_embedding.ngram_embedding.weight.requires_grad_(False)
    return model


def _compare_tp_gradients(rank, rendezvous):
    from megatron.core import parallel_state
    from megatron.core.distributed.finalize_model_grads import _allreduce_non_tensor_model_parallel_grads
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    torch.cuda.set_device(rank)
    torch.distributed.init_process_group('nccl', init_method=rendezvous, rank=rank, world_size=2)
    parallel_state.initialize_model_parallel(tensor_model_parallel_size=2)
    model_parallel_cuda_manual_seed(1234)
    group = parallel_state.get_tensor_model_parallel_group()
    try:
        for dtype in (torch.float32, torch.bfloat16):
            for fused in (False, True):
                os.environ['PLE_FUSED_KERNEL'] = str(int(fused))
                model = _make_ple(dtype, group)
                reference = _make_ple(dtype, group)
                reference.load_state_dict(model.state_dict())
                for instance in (model, reference):
                    for name, param in instance.named_parameters():
                        if not name.startswith('ple_embedding.'):
                            torch.distributed.broadcast(param.data, src=0, group=group)
                torch.manual_seed(1451)
                input_ids = torch.randint(1, 128, (1, 64), device='cuda')
                input_ids[:, [4, 25, 42, 53]] = 0
                full = torch.randn(64, 1, 128, dtype=dtype, device='cuda')
                upstream = torch.randn_like(full)
                offsets = [0, 20, 36, 64]
                packed = SimpleNamespace(
                    qkv_format='thd',
                    num_samples=3,
                    max_seqlen_q=28,
                    cu_seqlens_q=torch.tensor(offsets, device='cuda', dtype=torch.int32))
                local = full[rank * 32:(rank + 1) * 32].clone().requires_grad_()
                full_reference = full.clone().requires_grad_()
                expected = reference._forward_impl(full_reference, input_ids, packed)
                actual = model(local, input_ids, packed)
                (expected * upstream).float().sum().backward()
                (actual * upstream[rank * 32:(rank + 1) * 32]).float().sum().backward()
                for param in model.parameters():
                    if param.requires_grad:
                        param.main_grad = param.grad.detach().clone()
                model.ddp_config = SimpleNamespace(use_megatron_fsdp=False)
                _allreduce_non_tensor_model_parallel_grads([model], model.config, group)
                torch.testing.assert_close(actual, expected[rank * 32:(rank + 1) * 32], rtol=0, atol=0)
                torch.testing.assert_close(local.grad, full_reference.grad[rank * 32:(rank + 1) * 32], rtol=0, atol=0)
                for (name, param), (reference_name, ref_param) in zip(model.named_parameters(),
                                                                      reference.named_parameters()):
                    assert name == reference_name
                    if param.requires_grad:
                        torch.testing.assert_close(param.main_grad, ref_param.grad, rtol=1e-5, atol=1e-6)
    finally:
        parallel_state.destroy_model_parallel()
        torch.distributed.destroy_process_group()


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason='Real PLE TP regression requires two CUDA GPUs')
def test_ple_tp_finalized_gradients_match_full_sequence(tmp_path):
    torch.multiprocessing.spawn(
        _compare_tp_gradients, args=('file://' + str(tmp_path / 'rendezvous'), ), nprocs=2, join=True)
