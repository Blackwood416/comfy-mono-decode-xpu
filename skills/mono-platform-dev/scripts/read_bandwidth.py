"""DRAM read bandwidth of the XPU, the roof for a memory-bound decode step.

usage: python read_bandwidth.py [--gib 2] [--iters 20] [--device N]

Streams a buffer much larger than the last-level cache with a few torch kernels and reports
the best achieved GB/s per kernel. Device time comes from XPU events. Treat the best read number as
the practical roof: a decode kernel at 90-100% of it has no bandwidth headroom left.
"""
import argparse
import json

import torch


def timed(fn, iters):
    for _ in range(3):
        fn()
    torch.xpu.synchronize()
    best = float("inf")
    for _ in range(iters):
        s, e = torch.xpu.Event(enable_timing=True), torch.xpu.Event(enable_timing=True)
        s.record()
        fn()
        e.record()
        e.synchronize()
        best = min(best, s.elapsed_time(e) / 1e3)
    return best


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--gib", type=float, default=2.0)
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--device", type=int, default=0)
    a = ap.parse_args()
    torch.xpu.set_device(a.device)
    props = torch.xpu.get_device_properties(a.device)
    n = int(a.gib * 2**30) // 4
    x = torch.empty(n, dtype=torch.float32, device="xpu").uniform_()
    y = torch.empty_like(x)
    nbytes = x.numel() * 4
    res = {"device": props.name, "buffer_gib": round(nbytes / 2**30, 2), "llc_mib": props.last_level_cache_size / 2**20}
    for name, fn, moved in (
        ("read: sum fp32", lambda: x.sum(), nbytes),
        ("read: amax fp32", lambda: x.amax(), nbytes),
        ("read: sum bf16 view", lambda: x.view(torch.bfloat16).sum(dtype=torch.float32), nbytes),
        ("copy (read + write)", lambda: y.copy_(x), 2 * nbytes),
    ):
        s = timed(fn, a.iters)
        res[name] = round(moved / s / 1e9, 1)
    res["best_read_gbps"] = max(v for k, v in res.items() if k.startswith("read:"))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
