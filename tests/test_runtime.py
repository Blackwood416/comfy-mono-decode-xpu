# SPDX-License-Identifier: Apache-2.0
"""Minimal regressions for stateful error recovery and inference-mode norm caching."""
import sys
import types

import pytest
import torch

from comfy_mono_decode_xpu import guard, layer, mono, tiles


@pytest.mark.parametrize("failure", ["kv", "x", "subpath"])
def test_state_write_is_not_replayed(C, monkeypatch, failure):
    x = torch.zeros(1, 1, 4096, dtype=torch.bfloat16, device="xpu")
    kv = torch.zeros(1, device="xpu")
    replay = []

    class Block:
        layer_type = "full_attention"
        self_attn = None
        forward = lambda *args, **kwargs: replay.append(1)

    # Only the ComfyUI entry types are stand-ins; writes use real XPU tensors.
    mods = {name: types.ModuleType(name) for name in
            ("comfy", "comfy.text_encoders", "comfy.text_encoders.qwen35",
             "comfy.text_encoders.llama")}
    mods["comfy.text_encoders.qwen35"].Qwen35TransformerBlock = Block
    mods["comfy.text_encoders.llama"].SPEC_ROWS = 8
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(guard, "_off", {})
    monkeypatch.setattr(guard, "_is_interrupt", lambda error: False)
    for name in ("_install_attend", "_install_gated_attention"):
        monkeypatch.setattr(layer, name, lambda: False)
    monkeypatch.setattr(tiles, "install", lambda: 0)
    monkeypatch.setenv("MONO_DECODE_XPU_NORM_CACHE", "0")
    monkeypatch.setenv("MONO_DECODE_XPU_LOGITS", "0")
    monkeypatch.setenv("MONO_DECODE_XPU_AUTOTREE", "0")
    monkeypatch.setattr(layer, "_static_plan", lambda block: True)
    monkeypatch.setattr(layer, "_args", lambda *args:
                        dict(tiled=False, norm_in=(None, 0), norm_post=(None, 0), pre=[], out=[], mlp=[]))
    monkeypatch.setattr(layer, "_constants", lambda device: (None, None, None, None))
    monkeypatch.setattr(C, "layer_pre", lambda *args: [x] * 4)

    def attend(*args):
        kv.add_(1)
        if failure == "kv":
            raise RuntimeError("after state write")
        if failure == "subpath":  # a fused RoPE route failed after its KV write and disabled itself
            raise layer.SubpathError("after state write")
        return x

    def post(*args):
        x.add_(1)
        raise RuntimeError("after state write")

    monkeypatch.setattr(layer, "_attn_core", attend)
    monkeypatch.setattr(C, "layer_post_h16", post)
    monkeypatch.setattr(C, "layer_post", post)
    assert layer.install()
    with pytest.raises(RuntimeError, match="after state write"):
        Block().forward(x, past_key_value=kv)
    assert not replay and kv.item() == 1 and x.max().item() == int(failure == "x")
    # A sub-route failure leaves the layer kernels enabled; any other failure disables them.
    assert guard.off("layer") == (failure != "subpath")


def _norm_rope_inputs(monkeypatch, C, kernel):
    S, H, Hk, D, rd = 2, 2, 1, 256, 64
    rawq = torch.randn(1, S, H, 2 * D).to(torch.bfloat16)[..., :D]
    rawk = torch.randn(1, S, Hk, D).to(torch.bfloat16)
    xv4 = torch.randn(1, S, Hk, D).to(torch.bfloat16)
    ref_q, ref_k = torch.randn(1, H, S, D).to(torch.bfloat16), torch.randn(1, Hk, S, D).to(torch.bfloat16)
    writes = []
    kv = types.SimpleNamespace(key=torch.zeros(1, Hk, 8, D, dtype=torch.bfloat16), value=None,
                               position=torch.arange(8), write=lambda k, v: writes.append((k, v)))
    class Norm:
        eps = 1e-6

        def __call__(self, t):
            return t

    mix = types.SimpleNamespace(q_norm=Norm(), k_norm=Norm(), rotary_dim=rd)
    q35 = types.ModuleType("comfy.text_encoders.qwen35")
    q35.apply_partial_rope = lambda q, k, f, r: (ref_q, ref_k)
    mods = {"comfy": types.ModuleType("comfy"), "comfy.text_encoders": types.ModuleType("comfy.text_encoders"),
            "comfy.text_encoders.qwen35": q35}
    for name, module in mods.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.setattr(guard, "_off", {})
    monkeypatch.setattr(guard, "_is_interrupt", lambda error: False)
    monkeypatch.setattr(layer, "_norm_weight", lambda n, device: torch.ones(D, dtype=torch.bfloat16))
    monkeypatch.setattr(C, "qk_norm_rope_kv", kernel)
    return (mix, rawq, rawk, xv4, torch.zeros(S, rd // 2, 2, 2), kv), (ref_q, ref_k), writes


def test_norm_rope_mismatch_rewrites_kv_and_keeps_layer(C, monkeypatch):
    """A different reduction order (another Torch build) must not fail the request."""
    def kernel(rawq, rawk, xv4, wq, wk, freqs, key, value, position, eq, ek, rd):
        key.fill_(7)  # the fused route wrote its own (different) K into the cache
        return torch.zeros(1, rawq.shape[2], rawq.shape[1], 256, dtype=torch.bfloat16), \
            torch.zeros(1, rawk.shape[2], rawk.shape[1], 256, dtype=torch.bfloat16)

    args, (ref_q, ref_k), writes = _norm_rope_inputs(monkeypatch, C, kernel)
    q, k = layer._norm_rope_write(*args)
    assert q is ref_q and k is ref_k
    assert len(writes) == 1 and writes[0][0] is ref_k  # KV slots rewritten from the live chain
    assert guard.off("norm_rope") and not guard.off("layer")


def test_norm_rope_native_error_is_not_replayed(C, monkeypatch):
    def kernel(*a):
        raise RuntimeError("native failure")

    args, _, writes = _norm_rope_inputs(monkeypatch, C, kernel)
    with pytest.raises(layer.SubpathError):
        layer._norm_rope_write(*args)
    assert not writes and guard.off("norm_rope") and not guard.off("layer")


def test_copy_budget_error_reads_row_major(monkeypatch):
    def broken(device):
        raise AttributeError("unexpected parameter layout")

    owner = types.SimpleNamespace(modules=lambda: iter([]))
    monkeypatch.setattr(tiles, "_copy_budget", broken)
    monkeypatch.setattr(tiles, "_tiled_ok", True)
    monkeypatch.delenv("MONO_DECODE_XPU_SESSION_TILES", raising=False)
    try:
        tiles.begin(owner)
        assert tiles.tiled(object(), types.SimpleNamespace(device=torch.device("cpu"))) is None
        assert not tiles.tiles_enabled() and tiles.stats()["budget"]["layout"] == "row"
    finally:
        tiles.end()


def test_inactive_vbar_budget_preserves_active_owners(monkeypatch):
    device = torch.device("xpu:0")
    active, shared = object(), types.SimpleNamespace(loaded_size=lambda: 128)
    control = types.SimpleNamespace(get_xpu_allocator_mode=lambda: "native_hook",
                                    get_xpu_ur_hook_stats=lambda: {"tracked_alloc_calls": 1})
    monkeypatch.setitem(sys.modules, "comfy_aimdo", types.SimpleNamespace(control=control))
    mm = types.ModuleType("comfy.model_management")
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setitem(tiles._session, "model", active)

    def loaded(used=False, owner=None, dev=device, dynamic=True, bar=shared):
        model = types.SimpleNamespace(modules=lambda: iter([owner]), dynamic_vbars={dev: bar})
        return types.SimpleNamespace(device=dev, currently_used=used,
                                     model=types.SimpleNamespace(model=model, is_dynamic=lambda: dynamic))

    # Current owner, in-use model, other device and non-dynamic storage cannot contribute.
    # Shared inactive VBARs contribute their physical backing once.
    protected = types.SimpleNamespace(loaded_size=lambda: 512)
    mm.current_loaded_models = [loaded(owner=active, bar=protected), loaded(bar=protected),
                                loaded(used=True, bar=object()), loaded(dev=torch.device("xpu:1")),
                                loaded(dynamic=False), loaded(), loaded()]
    assert tiles._inactive_vram_budget(device) == 128
    control.get_xpu_allocator_mode = lambda: "global"
    assert tiles._inactive_vram_budget(device) == 0
    control.get_xpu_allocator_mode = lambda: "native_hook"
    control.get_xpu_ur_hook_stats = lambda: {}
    assert tiles._inactive_vram_budget(device) == 0


def test_reclaimable_pages_do_not_override_model_capacity(monkeypatch):
    device = torch.device("xpu:0")
    owner = torch.nn.Linear(16, 16, bias=False)
    owner._v = object()
    owner.weight._qdata = torch.zeros(16, 16, dtype=torch.int8)
    owner.weight._params = types.SimpleNamespace(group_size=16, convrot_groupsize=256,
        scale=torch.ones(32, dtype=torch.uint8), s_channel=torch.ones(16), codebook=torch.ones(16))
    mm = types.ModuleType("comfy.model_management")
    mm.minimum_inference_memory = lambda: 100
    monkeypatch.setitem(sys.modules, "comfy", types.SimpleNamespace(model_management=mm))
    monkeypatch.setitem(sys.modules, "comfy.model_management", mm)
    monkeypatch.setitem(tiles._session, "model", owner)
    monkeypatch.setattr(tiles, "_inactive_vram_budget", lambda device: 10000)
    monkeypatch.setattr(tiles, "_margin", 64)
    monkeypatch.setattr(torch.xpu, "memory_reserved", lambda device: 100)
    monkeypatch.setattr(torch.xpu, "mem_get_info", lambda device: (0, 1024))
    props = types.SimpleNamespace(total_memory=1024)
    monkeypatch.setattr(torch.xpu, "get_device_properties", lambda device: props)
    assert tiles._copy_budget(device)["layout"] == "tile"
    props.total_memory = 800
    assert tiles._copy_budget(device)["layout"] == "row"


def test_inference_norm_cache_has_generation_lifetime(monkeypatch):
    with torch.inference_mode():
        w = torch.ones(16, dtype=torch.bfloat16)
    calls = []
    norm = types.SimpleNamespace(weight=w, scale=lambda: calls.append(1) or w + 1)
    owner = types.SimpleNamespace(modules=lambda: iter([norm]))
    monkeypatch.setattr(layer, "comfy_cast", lambda value, device: value)
    try:
        tiles.begin(owner)
        a = layer._norm_weight(norm, torch.device("cpu"))
        assert layer._norm_weight(norm, torch.device("cpu")) is a and calls == [1]
        tiles.end()
        tiles.begin(owner)
        assert torch.equal(layer._norm_weight(norm, torch.device("cpu")), a)
        assert calls == [1, 1]
    finally:
        tiles.end()
