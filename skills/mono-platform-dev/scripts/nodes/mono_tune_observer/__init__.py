"""Decode observer for tuning runs (a ComfyUI custom node that adds no nodes).

Wraps Qwen35.generate and appends one JSON line per MTP generate to $MONO_TUNE_OUT:
tokens, verify steps, drafts, prefill and decode time, the effective MONO_DECODE_XPU* / MONO_TUNE*
variables, every node module's stats() after the generate, and `fallbacks`: the fallback counters
that grew during this generate (stats() counters are cumulative). Off unless MONO_TUNE_OUT is set.

- MONO_TUNE_EVENTS=1: XPU events around each verify forward and the step's first draft give
  device-timed draft / verify / tail segments per step (the first quarter of the steps is skipped).
- MONO_TUNE_PROFILE=N: an XPU-only torch profile of N verify steps, after the first
  MONO_TUNE_PROFILE_SKIP (default 16) steps, written next to $MONO_TUNE_OUT as profile-<n>.json.
  Summarise it with profile_summary.py. Kernels replayed from an XPU graph do not appear in it.

Load it through an extra model paths file whose `custom_nodes:` entry is the `nodes` directory;
bench_workflow.py does that.
It wraps Qwen35.generate, not _generate_mtp: the node's device MTP loop replaces _generate_mtp's
body and bypasses wrappers beneath generate.
"""
import hashlib
import inspect
import json
import os
import sys
import time

import torch

NODE_CLASS_MAPPINGS = {}
OUT = os.environ.get("MONO_TUNE_OUT")


def _node_modules():
    """Every loaded module of the node's package, by its last name component."""
    out = {}
    for name, module in list(sys.modules.items()):
        head, _, leaf = name.rpartition(".")
        if module is not None and (head == "comfy_mono_decode_xpu" or head.endswith(".comfy_mono_decode_xpu")):
            out.setdefault(leaf, module)
    return out


def _stats():
    out = {}
    for leaf, m in sorted(_node_modules().items()):
        if callable(getattr(m, "stats", None)):
            try:
                out[leaf] = m.stats()
            except Exception as e:  # noqa: BLE001 - diagnostics only
                out[leaf] = f"stats() failed: {e}"
    dl = _node_modules().get("draft_lmhead")
    if dl is not None and hasattr(dl, "MODE"):
        out["draft_int4_mode"] = dl.MODE
    return out


def _fallback_counts(stats):
    counts = {}
    for leaf, st in stats.items():
        fb = st.get("fallback") if isinstance(st, dict) else None
        if isinstance(fb, dict):
            for reason, n in fb.items():
                if isinstance(n, (int, float)):
                    counts[f"{leaf}.{reason}"] = n
    return counts


if OUT:
    from comfy.text_encoders import qwen35

    _orig_generate = qwen35.Qwen35.generate
    _signature = inspect.signature(_orig_generate)
    _count = [0]

    def _argument(self, a, k, name):  # callers pass max_length positionally
        try:
            return _signature.bind(self, *a, **k).arguments.get(name, k.get(name))
        except TypeError:
            return k.get(name)

    def _generate(self, *a, **k):
        if self.mtp is None or not k.get("mtp", True) or k.get("initial_input_ids") is not None:
            return _orig_generate(self, *a, **k)
        return _observed(self, *a, **k)

    def _observed(self, *a, **k):
        events = os.environ.get("MONO_TUNE_EVENTS") == "1"
        prof_steps = int(os.environ.get("MONO_TUNE_PROFILE", "0"))
        prof_skip = int(os.environ.get("MONO_TUNE_PROFILE_SKIP", "16"))
        ctr = {"verify": 0, "draft": 0, "t": [], "ev": [], "first_draft": True, "prof": None, "prof_file": None}

        def mark(kind):
            e = torch.xpu.Event(enable_timing=True)
            e.record()
            ctr["ev"].append((kind, e))

        mf, mtp_f = self.model.forward, self.mtp.forward

        def model_forward(*fa, **fk):
            if fk.get("decode_buffers") is None:  # prefill
                return mf(*fa, **fk)
            ctr["verify"] += 1
            ctr["t"].append(time.perf_counter())
            if prof_steps and ctr["verify"] == prof_skip + 1:
                ctr["prof"] = torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.XPU])
                ctr["prof"].__enter__()
            if events:
                mark("v0")
            out = mf(*fa, **fk)
            if events:
                mark("v1")
            ctr["first_draft"] = True
            if ctr["prof"] is not None and ctr["verify"] == prof_skip + prof_steps:
                _stop_profile(ctr)
            return out

        def note_draft():  # the fused draft head bypasses forward and calls this per draft
            ctr["draft"] += 1
            if events and ctr["first_draft"]:
                mark("d0")
            ctr["first_draft"] = False

        def mtp_forward(*fa, **fk):
            note_draft()
            return mtp_f(*fa, **fk)

        before = _fallback_counts(_stats())
        mtp_forward._mono_observer = True  # lets the fused draft head run (it skips unknown wrappers)
        mtp_forward._mono_on_draft = note_draft
        saved = [(obj, vars(obj).get("forward")) for obj in (self.model, self.mtp)]
        self.model.forward, self.mtp.forward = model_forward, mtp_forward
        t0 = time.perf_counter()
        try:
            ids = _orig_generate(self, *a, **k)
            torch.xpu.synchronize()
        finally:
            for obj, previous in saved:  # restore what was there (another observer may wrap too)
                if previous is None:
                    vars(obj).pop("forward", None)
                else:
                    obj.forward = previous
            if ctr["prof"] is not None:
                _stop_profile(ctr)
        t1 = time.perf_counter()
        _count[0] += 1
        st, steps = ctr["t"], ctr["verify"]
        seg = None
        if events and ctr["ev"]:
            d = {"draft": [], "verify": [], "tail": []}
            ev = ctr["ev"]
            for (ka, ea), (kb, eb) in zip(ev, ev[1:]):
                key = {("d0", "v0"): "draft", ("v0", "v1"): "verify", ("v1", "d0"): "tail"}.get((ka, kb))
                if key:
                    d[key].append(ea.elapsed_time(eb))
            seg = {k: round(sum(v[len(v) // 4:]) / (len(v) - len(v) // 4), 3) if v else None for k, v in d.items()}
        gaps = sorted(b - a for a, b in zip(st, st[1:]))
        n = len(ids)
        stats = _stats()
        max_length = _argument(self, a, k, "max_length")
        rec = dict(
            generate=_count[0], mtp_arg=_argument(self, a, k, "mtp"), max_length=max_length, tokens=n,
            hit_max_length=isinstance(max_length, int) and n >= max_length,
            ids_sha=hashlib.sha256(json.dumps(list(ids)).encode()).hexdigest()[:16],
            generate_s=round(t1 - t0, 3), prefill_s=round(st[0] - t0, 3) if st else None,
            decode_tps=round((n - 1) / (t1 - st[0]), 2) if st else None,
            steps=steps, drafts=ctr["draft"],
            tokens_per_step=round((n - 1) / steps, 3) if steps else None,
            drafts_per_step=round(ctr["draft"] / steps, 3) if steps else None,
            step_ms_mean=round(1000 * (st[-1] - st[0]) / (len(st) - 1), 3) if len(st) > 1 else None,
            step_ms_median=round(1000 * gaps[len(gaps) // 2], 3) if gaps else None,
            device_segments_ms=seg, profile=ctr["prof_file"],
            env={key: v for key, v in os.environ.items() if key.startswith(("MONO_DECODE_XPU", "MONO_TUNE"))},
            stats=stats, fallbacks={key: v - before.get(key, 0) for key, v in _fallback_counts(stats).items()
                                    if v - before.get(key, 0) > 0})
        with open(OUT, "a") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        return ids

    def _stop_profile(ctr):
        torch.xpu.synchronize()
        prof, ctr["prof"] = ctr["prof"], None
        prof.__exit__(None, None, None)
        path = os.path.join(os.path.dirname(os.path.abspath(OUT)), f"profile-{_count[0] + 1}.json")
        prof.export_chrome_trace(path)
        ctr["prof_file"] = path

    qwen35.Qwen35.generate = _generate
