# SPDX-License-Identifier: Apache-2.0
"""Compact regular-Hadamard post against the original full-matrix route."""
import pytest
import torch
import torch.nn.functional as F

from conftest import bits_equal
from test_w4a8_linear import random_w4


def test_compact_hadamard_post_matches_full(C):
    pytest.importorskip("comfy_kitchen")
    from comfy_kitchen.tensor.int8_utils import _build_hadamard

    g = torch.Generator(device="cpu").manual_seed(17)
    h = _build_hadamard(256, device="xpu", dtype=torch.bfloat16).contiguous()
    small = h[:16, :16]
    compact = torch.stack((small, -small)).contiguous()
    bf = torch.arange(65536, dtype=torch.int32).to(torch.int16).view(torch.bfloat16).xpu()
    silu, sig = F.silu(bf).contiguous(), torch.sigmoid(bf).contiguous()
    out = random_w4(4096, 4096, g)
    mats = [random_w4(n, k, g) for n, k in
            [(12288, 4096), (12288, 4096), (4096, 12288)]]
    args = [t for q, s, c, _ in mats for t in (q, s, c, out[3])]
    norm = torch.ones(4096, device="xpu", dtype=torch.bfloat16)
    for rows in (1, 3, 8):
        x = torch.randn(rows, 4096, generator=g).to(torch.bfloat16).xpu()
        y = torch.randn(rows, 4096, generator=g).to(torch.bfloat16).xpu()
        reference, actual = x.clone(), x.clone()
        C.layer_post(reference, y, None, sig, *out, norm, 1e-6, h, silu, *args, 0, 6, False)
        C.layer_post_h16(actual, y, None, sig, *out, norm, 1e-6, compact, silu, *args, 0, 6, False)
        assert bits_equal(actual, reference), rows
