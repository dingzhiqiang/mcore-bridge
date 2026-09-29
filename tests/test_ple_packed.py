# Copyright (c) ModelScope Contributors. All rights reserved.
"""Packed PLE computes each real segment without longest-segment padding."""

import torch
from types import SimpleNamespace

from mcore_bridge.model.modules.ple import Qwen4ExpTextPLELayer


class _CausalPLE(Qwen4ExpTextPLELayer):

    def __init__(self):
        torch.nn.Module.__init__(self)
        self.scale = torch.nn.Parameter(torch.tensor(0.5))
        self.ple_embedding = SimpleNamespace(eos_token_id=0)

    def compute(self, hidden_states, input_ids):
        return (hidden_states * self.scale + hidden_states.cumsum(dim=1) * 0.25
                + input_ids.unsqueeze(-1).to(hidden_states.dtype) * 0.125)


def _padded_reference(model, hidden_states, input_ids, offsets, max_len):
    num_samples = len(offsets) - 1
    hid = hidden_states.new_zeros((num_samples, max_len, hidden_states.shape[-1]))
    toks = input_ids.new_full((num_samples, max_len), model.ple_embedding.eos_token_id)
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        hid[i, :end - start] = hidden_states[start:end, 0]
        toks[i, :end - start] = input_ids[0, start:end]
    res = model.compute(hid, toks)
    out = res.new_zeros(hidden_states.shape)
    for i, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        out[start:end, 0] = res[i, :end - start]
    return out


def test_packed_ple_preserves_outputs_and_gradients_with_uneven_segments_and_padding():
    model = _CausalPLE()
    reference_input = torch.randn(10, 1, 2, requires_grad=True)
    packed_input = reference_input.detach().clone().requires_grad_()
    input_ids = torch.arange(10).unsqueeze(0)
    offsets = [0, 3, 4, 8]
    packed = SimpleNamespace(qkv_format='thd', num_samples=3, max_seqlen_q=4, cu_seqlens_q=torch.tensor([*offsets, 10]))

    reference = _padded_reference(model, reference_input, input_ids, offsets, 4)
    actual = model._forward_impl(packed_input, input_ids, packed)
    reference_grad = torch.autograd.grad(reference.sum(), (reference_input, model.scale))
    actual_grad = torch.autograd.grad(actual.sum(), (packed_input, model.scale))

    torch.testing.assert_close(actual, reference, rtol=0, atol=0)
    for got, expected in zip(actual_grad, reference_grad):
        torch.testing.assert_close(got, expected, rtol=1e-6, atol=1e-6)
