# QSA long-context offset overflow

## Symptom and root cause

A visual RL run completed its first training/weight-transfer cycle, then failed in
QSA sparse attention during the second PPO forward. Synchronous CUDA execution
localized the failure to `_qsa_bs_fwd_kernel`, rather than the secondary NCCL
watchdog error seen with asynchronous execution.

The uint8 selection bitmap has shape `[T, ceil(T / 4)]`. At `T=111872`, it
contains 3,128,836,096 elements. `offs_q * stride_st` was evaluated in signed
int32 before addition to the pointer; rows starting at 76,784 exceed its range.
CP scatters local queries back into full packed order before calling this kernel,
so CP2 does not halve this address range. Chunked logits loss does not affect it.

## Fix

Promote `offs_q` to `tl.int64` **before** multiplication by `stride_st` in all
three selection accesses: forward, dQ backward, and dK/dV backward. Casting the
already-overflowed product is insufficient. This changes address arithmetic only;
attention selection, CP semantics and model weights are unchanged. The kernel was
vendored from Miles; this is a documented local correction, not a claim that an
upstream Miles fix has been merged.

## Evidence and regression

On the actor CUDA image, an isolated original-kernel run with 111,872 queries
reproduced illegal memory access. The corrected kernel passed exact forward and
backward comparisons with zero Q/K, unit V, and each query selecting keys 0..3.
The analytical output is one, dQ/dK zero, and dV for the selected keys T/4.
`tests/test_qsa_large_offsets.py` preserves this regression and a short control.
Run with `python -m pytest tests/test_qsa_large_offsets.py -q`; the large test
requires CUDA, Triton and at least 8 GiB free GPU memory.

The original full replay job was 970877 (2026-09-22): step 1 passed, step 2 failed.
Its experiment artifact directory is
`chucai-swemm-flash-next-replay-cycle-20260922`, containing
`qsa-offset-rootcause-audit.json`, `qsa-offset-original.log` and
`qsa-offset-int64.log`. Full two-step replay after this fix is still pending.
Zero-advantage diagnostic batches do not prove effective RL learning or AWEX
numerical correctness. A prior replay without a crash does not establish correct
values: wrapped addresses may land in another mapped allocation.

## Related reports

- [SageAttention #386](https://github.com/thu-ml/SageAttention/issues/386):
  int32 index/stride overflow can crash or silently corrupt results.
- [comfy-kitchen #172](https://github.com/Comfy-Org/comfy-kitchen/pull/172):
  promotes indices before multiplication and tests offsets beyond 2**31.

These corroborate the mechanism; the local reproducer is the evidence for this
specific QSA implementation. Neither report establishes that all remaining visual
RL or weight-transfer issues are resolved.
