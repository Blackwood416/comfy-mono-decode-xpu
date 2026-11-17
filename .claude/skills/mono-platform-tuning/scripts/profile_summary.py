"""Per-step device time by kernel family from an XPU-only decode profile (observer MONO_TUNE_PROFILE=N).

usage: python profile_summary.py PROFILE.json [--gdn-per-step 24] [--skip 1]

Steps are delimited by every Nth Gated-DeltaNet layer kernel (Qwen3.5-9B: 24 per verify). Prints
ms per step for each kernel family, the device idle per step and where the gaps fall. Compare the
families with their bytes floor (reference/measurement.md). oneDNN GEMMs over 1 ms are the int8
lm_head; the MTP block GEMMs and the int4 draft lm_head land in "other gemm".
"""
import argparse
import collections
import json

FAMILIES = (  # (substring of the kernel name, family); first match wins
    ("LayerPost", "layer_post"), ("LayerGdn", "layer_gdn"), ("LayerPre", "layer_pre"),
    ("AttnDpasPart", "attn_dpas part"), ("AttnDpasMerge", "attn_dpas merge"), ("AttnCore", "attn_core"),
    ("NormRopeKv", "rope / norm_rope"), ("RopeKv", "rope / norm_rope"), ("TopkRows", "verify top-k"),
    ("FastTopk", "fast_topk / pick"), ("DraftPick", "draft_pick"), ("MonoMlp", "mono_mlp"),
    ("W4A8", "w4a8 linear"), ("RotQuant", "rot_quant"),
)


def family(e):
    name = e["name"]
    for key, fam in FAMILIES:
        if key in name:
            return fam
    if name.startswith("gemm_kernel") or "gemm" in name.lower():
        return "int8 lm_head (gemm > 1 ms)" if e["dur"] > 1000 else "other gemm (oneDNN)"
    if e.get("cat") != "kernel":
        return e.get("cat", "?")
    return "other small kernels"


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("profile")
    ap.add_argument("--gdn-per-step", type=int, default=24)
    ap.add_argument("--skip", type=int, default=1, help="steps to drop at the start of the window")
    a = ap.parse_args()
    ev = json.load(open(a.profile))["traceEvents"]
    ks = sorted((e for e in ev if e.get("ph") == "X" and e.get("cat") in ("kernel", "gpu_memcpy", "gpu_memset")),
                key=lambda e: e["ts"])
    gdn = [e for e in ks if "LayerGdn" in e["name"]]
    starts = gdn[::a.gdn_per_step]
    if len(starts) < a.skip + 2:
        raise SystemExit(f"only {len(starts)} steps found (LayerGdn kernels: {len(gdn)}); is the layer path on?")
    t0, t1 = starts[a.skip]["ts"], starts[-1]["ts"]
    n = len(starts) - 1 - a.skip
    win = [e for e in ks if t0 <= e["ts"] < t1]
    fam, cnt = collections.Counter(), collections.Counter()
    for e in win:
        f = family(e)
        fam[f] += e["dur"]
        cnt[f] += 1
    busy = sum(fam.values())
    print(f"window: {n} steps, {(t1 - t0) / n / 1e3:.3f} ms/step, device busy {busy / n / 1e3:.3f}, "
          f"idle {((t1 - t0) - busy) / n / 1e3:.3f} ms/step, {len(win) / n:.0f} kernels/step")
    for f, d in fam.most_common():
        print(f"  {f:24s} {d / n / 1e3:7.3f} ms/step  x{cnt[f] / n:5.1f}  {d / cnt[f]:9.1f} us each")
    gaps = []
    for x, y in zip(win, win[1:]):
        g = y["ts"] - (x["ts"] + x["dur"])
        if g > 2:
            gaps.append((g, family(x), family(y)))
    pairs = collections.Counter()
    for g, fx, fy in gaps:
        pairs[(fx, fy)] += g
    print("idle by (previous, next) family, ms/step:")
    for (fx, fy), g in pairs.most_common(8):
        print(f"  {g / n / 1e3:7.3f}  {fx} -> {fy}")


if __name__ == "__main__":
    main()
