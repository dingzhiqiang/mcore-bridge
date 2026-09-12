# SPDX-License-Identifier: Apache-2.0

import ast
import torch
from pathlib import Path

SOURCE = Path(__file__).parents[1] / 'src/mcore_bridge/model/gpts/qwen4_exp.py'


def load_combine_helper():
    tree = ast.parse(SOURCE.read_text())
    helper = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == '_combine_hyper_connection')
    module = ast.Module(body=[helper], type_ignores=[])
    namespace = {'torch': torch}
    exec(compile(ast.fix_missing_locations(module), SOURCE, 'exec'), namespace)
    return namespace['_combine_hyper_connection'], tree


def old_inline_combine(block_output, hyper_input, injection_weights):
    injection = block_output.unsqueeze(-2) * injection_weights.unsqueeze(-1)
    return hyper_input + injection.flatten(-2)


def test_private_combine_is_bitwise_equal_for_attn_and_mlp_values():
    combine, _ = load_combine_helper()
    generator = torch.Generator().manual_seed(7)

    for _branch in ('attn', 'mlp'):
        block_output = torch.randn(9, 1, 16, generator=generator).to(torch.bfloat16)
        hyper_input = torch.randn(9, 1, 64, generator=generator).to(torch.bfloat16)
        injection_weights = torch.randn(9, 1, 4, generator=generator).to(torch.bfloat16)
        expected = old_inline_combine(block_output, hyper_input, injection_weights)
        actual = combine(object(), block_output, hyper_input, injection_weights)

        assert actual.dtype == torch.bfloat16
        assert torch.equal(actual, expected)


def test_both_layer_combine_sites_call_private_helper():
    _, tree = load_combine_helper()
    layer = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == 'Qwen4ExpLayer')
    forward = next(node for node in layer.body if isinstance(node, ast.FunctionDef) and node.name == 'forward')
    calls = [
        node for node in ast.walk(forward) if isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
        and node.func.id == '_combine_hyper_connection'
    ]

    assert len(calls) == 2
