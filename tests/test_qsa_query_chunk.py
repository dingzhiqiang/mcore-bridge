"""Compare real selection methods against the unchunked reference snapshot."""
import os
import torch
import types
from pathlib import Path


def load(path):
    source = path.read_text().replace('from megatron.core.extensions.transformer_engine import TELinear', '')
    scope = {}
    exec(compile(source, str(path), 'exec'), scope)
    return scope['QSAIndexer']


def make(cls):
    obj = cls.__new__(cls)
    torch.nn.Module.__init__(obj)
    obj.config = types.SimpleNamespace(attention_scaling=1.0)
    obj.index_n_heads = 4
    obj.index_kv_heads = 1
    obj.index_head_dim = 8
    obj.compress_ratio = 4
    obj.block_topk = 3
    obj.q_layernorm = torch.nn.Identity()
    obj.k_layernorm = torch.nn.Identity()
    obj.index_qk_proj = lambda x: (x, None)
    return obj


def main():
    base = Path(os.environ['QSA_REFERENCE_FILE'])
    changed = Path(__file__).parents[1] / 'src/mcore_bridge/model/modules/qsa_indexer.py'
    old, new = make(load(base)), make(load(changed))
    torch.set_default_device(os.environ.get('QSA_TEST_DEVICE', 'cpu'))
    torch.manual_seed(37)
    cases = 0
    for chunk in (1, 7, 32, 1024):
        os.environ['QWEN_QSA_QUERY_CHUNK_SIZE'] = str(chunk)
        for ties in (False, True):
            for lengths in ([37, 26, 17], [3, 33, 2, 29]):
                n = sum(lengths)
                hidden = torch.randn(n, 40) if not ties else torch.zeros(n, 40)
                freqs = torch.randn(n, 1, 1, 8)
                cu = torch.tensor([0] + list(torch.tensor(lengths).cumsum(0).tolist()), dtype=torch.int32)
                a = old.select_token_indices_thd(hidden, freqs, cu, force_materialize=True)
                b = new.select_token_indices_thd(hidden, freqs, cu, force_materialize=True)
                torch.testing.assert_close(a, b, rtol=0, atol=0)
                for start, end in zip(cu[:-1], cu[1:]):
                    for pos in range(int(start), int(end)):
                        valid = b[pos][b[pos] >= 0]
                        assert ((valid >= start) & (valid <= pos)).all()
                cases += 1
            hidden = torch.randn(73, 2, 40) if not ties else torch.zeros(73, 2, 40)
            freqs = torch.randn(73, 1, 1, 8)
            a = old._score_and_topk_blocks(hidden, freqs)
            b = new._score_and_topk_blocks(hidden, freqs)
            for x, y in zip(a, b):
                torch.testing.assert_close(x, y, rtol=0, atol=0)
            cases += 1
    print({'cases': cases, 'packed_and_unpacked_exact': True, 'ties_and_doc_boundaries': True})


if __name__ == '__main__':
    main()
