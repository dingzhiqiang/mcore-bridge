# Copyright (c) ModelScope Contributors. All rights reserved.
import pytest
import torch
from test_qwen4_exp_units import _make_config

import mcore_bridge.model.modules.qsa_indexer as qi


@pytest.mark.parametrize('lengths', [[41, 35, 19], [3, 5, 2], [32, 48]])
@pytest.mark.parametrize('chunk', [1, 7, 32])
@pytest.mark.skipif(not torch.cuda.is_available(), reason='TELinear requires CUDA')
def test_packed_chunking_matches_full_scores_and_bounds_workspace(monkeypatch, lengths, chunk):
    torch.manual_seed(42)
    cfg = _make_config(compress_ratio=4, budget=16)
    indexer = qi.QSAIndexer(cfg).cuda()
    with torch.no_grad():
        indexer.index_qk_proj.weight.normal_(0, 0.02)
        indexer.q_layernorm.weight.normal_(0, 0.02)
        indexer.k_layernorm.weight.normal_(0, 0.02)
    total = sum(lengths)
    hidden = torch.randn(total, cfg.hidden_size, device='cuda')
    freqs = torch.randn(total, 1, 1, cfg.indexer_head_dim, device='cuda')
    cu = torch.tensor([0] + lengths, device='cuda').cumsum(0)
    monkeypatch.setattr(qi, '_QSA_INDEX_SCORE_CHUNK_BYTES', 1 << 62)
    monkeypatch.setenv('QWEN_QSA_QUERY_CHUNK_SIZE', str(total))
    expected = indexer.select_token_indices_thd(hidden, freqs, cu, force_materialize=True)
    monkeypatch.setenv('QWEN_QSA_QUERY_CHUNK_SIZE', str(chunk))
    einsum = torch.einsum
    sizes = []

    def measured(equation, *args):
        if equation == 'thd,kd->thk':
            sizes.append(args[0].shape[0])
        return einsum(equation, *args)

    monkeypatch.setattr(torch, 'einsum', measured)
    actual = indexer.select_token_indices_thd(hidden, freqs, cu, force_materialize=True)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert sizes and max(sizes) <= chunk
    assert sum(sizes) == total
    for start, end in zip(cu.tolist()[:-1], cu.tolist()[1:]):
        for row in range(start, end):
            valid = actual[row][actual[row] >= 0]
            assert bool(((valid >= start) & (valid <= row)).all())


def test_query_chunk_size_honors_byte_and_row_limits(monkeypatch):
    monkeypatch.setattr(qi, '_QSA_INDEX_SCORE_CHUNK_BYTES', 128)
    monkeypatch.setenv('QWEN_QSA_QUERY_CHUNK_SIZE', '7')
    assert qi._query_chunk_size(100, 32) == 4
    assert qi._query_chunk_size(100, 8) == 7
    assert qi._query_chunk_size(3, 8) == 3
    assert qi._query_chunk_size(100, 256) == 1


@pytest.mark.parametrize('limit', ['0', '-1', 'bad'])
def test_invalid_query_limit_fails(monkeypatch, limit):
    monkeypatch.setenv('QWEN_QSA_QUERY_CHUNK_SIZE', limit)
    with pytest.raises(ValueError):
        qi._query_chunk_size(10, 32)
