---
name: mono-platform-tuning
description: Bring up, benchmark and tune comfy-mono-decode-xpu on a specific Intel GPU (Arc B580, B60, B70 or another Xe2 device). Checks the device against the kernels' assumptions, builds for it, confirms which decode paths actually run, measures prompt-enhance decode against the device's bandwidth roof, and guides device-gated policy or kernel changes that keep the verify path bit-exact. Use when the user wants to run, port, profile, benchmark or speed up this node on their own hardware, or asks why it is slower on their card.
---

# Tuning mono decode on a new platform

The node replaces ComfyUI prompt enhance's decode (Qwen3.5-9B W4A8, MTP speculative decoding) with
persistent SYCL kernels. A decode step is memory-bound: the goal on any device is to stream the
weights near its DRAM read bandwidth with the fast paths active, without changing the verify path's
numerics. Read `docs/DESIGN.md` (what the kernels do, the exactness boundary) and `docs/ENV.md` (every
switch) before changing anything.

Work through the steps in order and report what you found at each one. Do not skip to kernel work:
on the ported devices the large wins were the fast paths running at all (B580) and the weight layout,
not kernel edits.

## 1. Probe the platform

```bash
python .claude/skills/mono-platform-tuning/scripts/platform_probe.py --comfyui /path/to/ComfyUI --json probe.json
```

Run it with ComfyUI's Python. It prints the device, the stack versions and checks:
- **FAIL co-residency / work-group / isa:** the persistent kernels cannot run safely on this
  device (for example DG2: 16 EUs per Xe core, no 2D block IO). Stop and tell the user; the node
  must stay on the stock path there (`MONO_DECODE_XPU=0`), and a port is a separate project
  (reference/devices.md).
- **WARN memory / slm / integrated:** expect row-major weights or rejected launches; look for them in
  step 3.
- **WARN device:** not a development device; every later step applies in full.

Also measure the roof once: `python .claude/skills/mono-platform-tuning/scripts/read_bandwidth.py`.

## 2. Build and run the unit tests

```bash
python install.py --force            # default AOT target bmg (BMG-G21 and G31); --target for others
PYTHONPATH=/path/to/ComfyUI python -m pytest -q tests
```

Use the same environment ComfyUI starts with (oneAPI, launcher variables, allocator preload). The
tests compare each kernel with the operation it replaces, bit for bit, on this device; all must pass.
A failure here is a correctness bug on this device: do not benchmark around it. The build takes
several minutes; `install.py` never fails the node install, so check its output for "built".

## 3. Confirm the fast paths run

Run the official template once through the frontend, or with the bench driver below, and read:
- the server log lines `[mono_decode_xpu] ... installed=True` and `weight layout=...`;
- `stats.layer` in the records: `layer`, `gdn1`, `attn_dpas`, `norm_rope` counting, `fallback`
  explained, `disabled` empty, `weight_layout` per generate;
- `stats.mtp_loop` (device MTP loop active) and `stats.autotree`.

reference/tuning.md section 1–2 explains each field and the usual causes.

## 4. Baseline

```bash
python .claude/skills/mono-platform-tuning/scripts/bench_workflow.py \
  --comfyui /path/to/ComfyUI --workflow template-api.json --out bench/base \
  --config stock:MONO_DECODE_XPU=0 --config default: --config stock2:MONO_DECODE_XPU=0 \
  --warmup 2 --runs 3 [--pe-only] [--launcher "/path/to/start.sh"] [--extra-model-paths paths.yaml]
```

- The workflow is the official template in API format (frontend: Export (API)), with the user's
  models and settings unchanged.
- Each configuration gets a fresh server. Run 0 (the first image) is reported apart from the
  measured runs; only the prompt-enhance seed varies.
- `summary.md` gives decode tok/s, tokens, tokens and drafts per step, step time, the weight layout,
  fallbacks and runs that hit `max_length`.
- Add `--config int4:MONO_DECODE_XPU_DRAFT_INT4=1` for the experimental draft lm_head.
- Add `--mtp tree4s1m8` for identical work across configurations.
- `--pe-only` runs prompt enhance alone: faster, but its memory state matches the first image only.

Rules (reference/measurement.md): real server and workflow, one variable at a time, baseline again
at the end, first image versus later images, a fixed tree for kernel A/B, and read every prompt and
look at every image.

## 5. Find where the time goes

- `MONO_TUNE_EVENTS=1` in a configuration adds device-timed draft / verify / tail per step.
- `MONO_TUNE_PROFILE=24` records an XPU-only profile of 24 verify steps. Summarize it:
  `python .claude/skills/mono-platform-tuning/scripts/profile_summary.py bench/<cfg>/profile-<n>.json`.
- Compare with the floor: bytes per step over the measured read bandwidth, and each kernel family
  against its share of the floor on B70 (reference/measurement.md).
- A family far below its B70 share is the lead. lm_head and the oneDNN GEMMs show what the memory
  system can reach on this device.

## 6. Tune

Follow reference/tuning.md in order:
1. fast paths running;
2. weight layout and memory budget;
3. the draft lm_head switch;
4. the MTP tree;
5. attention chunks;
6. work-groups per launch;
7. runtime and memory modes;
8. kernel work last.

For any code change follow reference/kernel-changes.md:
- device-gated policy in the launch planning;
- other devices unchanged unless measured;
- verify path bitwise;
- one process per build in an A/B;
- tests after every change.

## 7. Validate and report

For a change to keep:
- `tests/` pass on this device;
- the official workflow completes in the user's memory mode for the first image and later images;
- the prompts read well and the images look right;
- decode improves at a fixed tree and under `auto`;
- the baseline drift is small.

Report:
- device: probe output, read bandwidth;
- stack versions and node commit;
- each configuration's switches;
- the summary table for the first image and the measured runs;
- weight layout, fallbacks, runs that hit `max_length`;
- floor and ceiling estimates;
- what was tried and did not help, with numbers.

If a change is device-specific, say which devices were measured and which were not.

## References

- reference/devices.md: every device assumption in the code, and the B70 / B580 / B60 ports
  (numbers, device-specific code, what did not help), plus DG2 and other devices.
- reference/measurement.md: protocol, record fields, the bytes floor, profiling traps, host effects.
- reference/tuning.md: the levers in order, with what each changed on the ported devices.
- reference/kernel-changes.md: exactness contract, gating, persistent-kernel rules, build and A/B
  traps.
- scripts/: `platform_probe.py`, `read_bandwidth.py`, `bench_workflow.py` (with the `nodes/mono_tune_observer/`
  custom node it loads), `profile_summary.py`.
