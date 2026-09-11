# Copyright (c) ModelScope Contributors. All rights reserved.
"""CPU-only checks for the PLE shard partitioner; no Megatron/CUDA import."""
import gc
import importlib.util
import pytest
import torch
import weakref
from pathlib import Path


@pytest.fixture
def exporter():
    path = Path(__file__).resolve().parents[1] / 'src/mcore_bridge/model/modules/ple_export.py'
    spec = importlib.util.spec_from_file_location('_ple_export_cpu', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize('tp_size', [1, 2, 3, 4])
@pytest.mark.parametrize('num_shards', [1, 5, 7, 24])
def test_tp_shards_reconstruct_exact_current_parameter(exporter, tp_size, num_shards):
    rows, width = 24, 3
    weight = torch.arange(rows * width).reshape(rows, width).to(torch.bfloat16) / 8 - 5
    ranges = list(exporter.iter_ple_shard_ranges(rows, num_shards))
    restored = []
    for _, start, end in ranges:
        fragments = []
        for rank in range(tp_size):
            tp_start, tp_end = rank * rows // tp_size, (rank + 1) * rows // tp_size
            fragments.append(exporter.build_ple_shard_fragment(weight[tp_start:tp_end], tp_start, tp_end, start, end))
        restored.append(torch.stack(fragments).sum(0))
        assert all(fragment.shape[0] <= (rows + num_shards - 1) // num_shards for fragment in fragments)
    torch.testing.assert_close(torch.cat(restored), weight, atol=0, rtol=0)


def test_stream_reflects_adam_update_without_source_fp8_rounding(exporter, monkeypatch):
    monkeypatch.setattr(exporter.dist, 'get_world_size', lambda group: 1)
    weight = torch.nn.Parameter(torch.full((12, 4), 0.5, dtype=torch.bfloat16))
    optimizer = torch.optim.Adam([weight], lr=0.0078125, weight_decay=0.0)
    weight.grad = torch.arange(48).reshape(12, 4).remainder(3).to(weight) + 1
    optimizer.step()
    expected = weight.detach().clone()
    values = dict(exporter.iter_ple_table_shards(weight, 0, 12, 12, 5, None))
    actual = torch.cat(list(values.values()))
    assert actual.dtype == torch.bfloat16
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert not torch.equal(actual, torch.full_like(actual, 0.5))
    old_scale = 0.03125
    old_fp8_roundtrip = (expected.float() / old_scale).to(torch.float8_e4m3fn).float() * old_scale
    assert not torch.equal(actual.float(), old_fp8_roundtrip)


def test_stream_releases_each_shard_when_consumer_drops_it(exporter, monkeypatch):
    monkeypatch.setattr(exporter.dist, 'get_world_size', lambda group: 1)
    weight = torch.zeros(24, 8, dtype=torch.bfloat16)
    stream = exporter.iter_ple_table_shards(weight, 0, 24, 24, 4, None)
    _, first = next(stream)
    ref = weakref.ref(first)
    assert first.numel() * first.element_size() == 6 * 8 * 2
    del first
    _, second = next(stream)
    gc.collect()
    assert ref() is None
    assert second.numel() == 6 * 8
    stream.close()


def test_stream_does_not_overwrite_retained_previous_shard(exporter, monkeypatch):
    monkeypatch.setattr(exporter.dist, 'get_world_size', lambda group: 1)
    weight = torch.arange(24, dtype=torch.float32).reshape(12, 2)
    stream = exporter.iter_ple_table_shards(weight, 0, 12, 12, 4, None)
    _, first = next(stream)
    before = first.clone()
    _, second = next(stream)
    assert first.data_ptr() != second.data_ptr()
    torch.testing.assert_close(first, before, atol=0, rtol=0)


def test_invalid_partition_fails_before_export(exporter):
    with pytest.raises(ValueError, match='TP vocabulary interval'):
        exporter.build_ple_shard_fragment(torch.ones(4, 2), 0, 3, 0, 2)
    with pytest.raises(ValueError, match='dequantized'):
        exporter.build_ple_shard_fragment(torch.ones(4, 2, dtype=torch.uint8), 0, 4, 0, 2)
