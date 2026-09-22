# Copyright (c) ModelScope Contributors. All rights reserved.
import importlib.util
import pytest
import sys
import torch
from pathlib import Path


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason='QSA requires CUDA')
@pytest.mark.parametrize('tokens', [256, 111872])
def test_qsa_bitmap_large_offsets_exact_forward_backward(tokens):
    """Cross signed-int32 addressing with a closed-form attention oracle."""
    pytest.importorskip('triton')
    if torch.cuda.mem_get_info()[0] < 8 * 1024**3:
        pytest.skip('Large-offset regression requires 8 GiB free GPU memory')
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/kernels/qsa_block_sparse_attn.py'
    spec = importlib.util.spec_from_file_location('qsa_large_offsets_kernel', path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    q = torch.zeros((tokens, 1, 32), dtype=torch.bfloat16, device='cuda', requires_grad=True)
    k = torch.zeros_like(q, requires_grad=True)
    v = torch.ones_like(q, requires_grad=True)
    sel = torch.zeros((tokens, (tokens + 3) // 4), dtype=torch.uint8, device='cuda')
    sel[:, 0] = 1
    lo = torch.zeros(tokens, dtype=torch.int32, device='cuda')
    hi = torch.full_like(lo, 3)
    out = module.qsa_block_sparse_attention_triton(q, k, v, sel, lo, hi, lo, lo, 1.0, 4)
    torch.testing.assert_close(out, torch.ones_like(out), rtol=0, atol=0)
    out.sum().backward()
    torch.testing.assert_close(q.grad, torch.zeros_like(q), rtol=0, atol=0)
    torch.testing.assert_close(k.grad, torch.zeros_like(k), rtol=0, atol=0)
    expected = torch.zeros_like(v)
    expected[:4] = tokens / 4
    torch.testing.assert_close(v.grad, expected, rtol=0, atol=0)
