"""Complete attention core: the existing FP64/Kitchen bound and graph-owned scratch."""
import pytest
import torch

from conftest import bits_equal
from test_attn_dpas import _kitchen, _mask, _reference


@pytest.mark.parametrize("parents", [None, [-1, 0, 0, 1, 1, 2, 2, 3]])
def test_attention_mono_state_accuracy_and_graph_replay(C, parents):
    rows, length, capacity = (1, 4000, 4097) if parents is None else (8, 4217, 12416)
    gen = torch.Generator().manual_seed(124)

    def rand(*shape):
        return torch.randn(*shape, generator=gen).to(dtype=torch.bfloat16, device="xpu")

    rawq, rawk, values = rand(1, rows, 16, 512)[..., :256], rand(1, rows, 4, 256), rand(1, rows, 4, 256)
    qw, kw = rand(256) * .3 + 1, rand(256) * .3 + 1
    key, value = rand(1, 4, capacity, 256), rand(1, 4, capacity, 256)
    key[:, :, length:] = float("nan")
    value[:, :, length:] = float("nan")
    ck, cv = key.clone(), value.clone()
    pos = torch.arange(length, length + rows, device="xpu", dtype=torch.int64)
    sl = torch.tensor([length], device="xpu", dtype=torch.int32)
    theta = torch.randn(rows * 2, 32, generator=gen)
    # Dense inner frequency dimensions with a nontrivial row stride.
    freq = torch.stack((theta.cos(), -theta.sin(), theta.sin(), theta.cos()), -1).reshape(-1, 32, 2, 2).xpu()[::2]
    mask = None if parents is None else _mask(parents)
    mt = torch.tensor([] if mask is None else mask, device="xpu", dtype=torch.int32)
    q, k = C.qk_norm_rope_kv(rawq, rawk, values, qw, kw, freq, key, value, pos, 1e-6, 1e-6, 64)
    workspace = C.attention_mono_workspace(rawq, rawk, ck)
    args = (rawq, rawk, values, qw, kw, freq, ck, cv, pos, sl, mt, 1e-6, 1e-6, 64, workspace)
    output, packed_q, actual_k = C.attention_mono(*args)
    torch.xpu.synchronize()
    graph = torch.xpu.XPUGraph()
    with torch.xpu.graph(graph):
        C.attention_mono(*args)
    for _ in range(4):
        graph.replay()
    torch.xpu.synchronize()
    assert bits_equal(q, packed_q[:, :rows * 4].reshape(1, 16, rows, 256))
    assert bits_equal(k, actual_k)
    assert bits_equal(key, ck) and bits_equal(value, cv)
    oracle = _reference(q, key, value, length, k, values.transpose(1, 2), mask).reshape(output.shape)
    error = (output.double().cpu() - oracle).abs()
    kitchen = _kitchen(q, key, value, sl, k, values.transpose(1, 2), None if mask is None else mt)
    bound = (kitchen.double().cpu().reshape(output.shape) - oracle).abs()
    assert torch.isfinite(output).all()
    assert error.max() <= 1.5 * bound.max() + 1e-3
    assert error.mean() <= 1.5 * bound.mean() + 1e-5
