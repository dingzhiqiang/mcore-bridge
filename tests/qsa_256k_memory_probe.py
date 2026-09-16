import json
import os
import time
import torch
from pathlib import Path
from test_qsa_query_chunk import load, make

os.environ['QWEN_QSA_QUERY_CHUNK_SIZE'] = '1024'
root = Path(__file__).parents[1]
model = make(load(root / 'src/mcore_bridge/model/modules/qsa_indexer.py'))
model.index_head_dim = 128
model.block_topk = 512
length = 262144
hidden = torch.randn(length, 640, device='cuda', dtype=torch.bfloat16)
freqs = torch.zeros(length, 1, 1, 128, device='cuda', dtype=torch.bfloat16)
cu = torch.tensor([0, length], device='cuda', dtype=torch.int32)
torch.cuda.reset_peak_memory_stats()
start = time.time()
indices = model.select_token_indices_thd(hidden, freqs, cu, force_materialize=True)
torch.cuda.synchronize()
assert tuple(indices.shape) == (length, 2052)
for pos in (0, 3, 4, 1023, 1024, length - 1):
    valid = indices[pos][indices[pos] >= 0]
    assert bool(((valid >= 0) & (valid <= pos)).all())
print(
    json.dumps({
        'stage': 'indexer_only_not_full_training',
        'tokens': length,
        'shape': list(indices.shape),
        'peak_GiB': torch.cuda.max_memory_allocated() / 2**30,
        'elapsed_s': time.time() - start
    }),
    flush=True)
