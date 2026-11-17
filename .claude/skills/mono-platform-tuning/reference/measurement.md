# Measuring decode

## Protocol

- **Real server, real workflow.** Start ComfyUI the way the user does (same launcher, flags and
  environment) and submit the official template. Standalone scripts that load the text encoder
  themselves miss ComfyUI-OmniXPU's prestartup and other startup patches, and measure a different
  stack. `scripts/bench_workflow.py` replays an API-format workflow on fresh servers.
- **One fresh server per configuration.** Configurations differ only in `MONO_DECODE_XPU*`
  variables. Check `env` in the records: launch scripts sometimes set their own defaults.
- **Only the nodes the workflow needs.** Other custom nodes that wrap `Qwen35.generate` or the
  model's `forward` (timers, debug observers) change what is measured and can hide the fused draft
  head's draft count. Pass `--server-args "--disable-all-custom-nodes --whitelist-custom-nodes
  comfy-mono-decode-xpu ComfyUI-OmniXPU mono_tune_observer ..."` (add what the template needs).
- **Change only the prompt-enhance sampling seed between runs.** Prompt enhance and everything
  after it re-execute; the loaders stay cached.
- **First image apart from later images.** After the first image the diffusion model stays
  resident while the prompt is enhanced again, so the free memory, and with it the weight layout,
  can differ. Report both.
- **Warmups, then at least three measured runs.** Run the baseline configuration again at the end
  (`--config stock:... --config default: --config stock2:...`) and report its drift. If the
  baseline drifts by more than about 1%, the comparison is inconclusive.
- **Identical work needs a fixed tree.** Under `auto` the MTP tree follows each configuration's
  measured speed, so tokens per step, drafts per step and the generated text all differ. For a
  kernel A/B use `--mtp tree4s1m8` (or another fixed mode) and compare step time; report the `auto`
  numbers as the user-visible result.
- **Text length moves tok/s.** Attention cost grows with the context, and H3 prompts think for
  several thousand tokens. A generate that reaches `max_length` (`hit_max_length`) usually has an
  empty caption: that is a prompt-enhance thinking-length issue, not a decode error. Use another
  seed for functional checks, and keep such runs out of speed means, or report them separately.
- **No end-to-end speedup claims across different outputs.** Conditioning, sampling and VAE do
  not depend on the decode kernels; attribute only prompt-enhance time to them.
- **Look at the outputs.** A saved image is not a pass: read the enhanced prompt (complete,
  coherent, consistent with the request) and look at the image.

The records (`gens.jsonl`, one per MTP generate):
- `decode_tps`: `(tokens - 1) / (end of generate - first verify)`, so prefill is excluded
  (`prefill_s`).
- `tokens_per_step`, `drafts_per_step`, `step_ms_mean`: verify steps are counted on the host.
- `device_segments_ms` (with `MONO_TUNE_EVENTS=1`): device time per step for the draft, verify and
  tail (lm_head and sampling after the verify) segments.
- `stats`: every module's `stats()` after the generate. The counters accumulate over the server's
  lifetime.

## The roof

A decode step streams the weights once per verify and once per draft; at 1–8 rows nothing else
comes close. Measure the read bandwidth with `scripts/read_bandwidth.py`, then:

```
bytes per step ~ 5.0 GB (verify: layer_post 3.0, layer_gdn 0.8, layer_pre 0.2, lm_head 1.0)
               + 1.5 GB per draft (lm_head 1.0, MTP block 0.5; about 1.0 GB with MONO_DECODE_XPU_DRAFT_INT4=1)
               + KV cache reads (grows with the context; about 0.15 GB per step at a few thousand tokens)
floor ms       = bytes per step / read GB/s
ceiling tok/s  = tokens per step / floor
```

These are Qwen3.5-9B W4A8 figures derived from a B70 profile. At 2 drafts and 2.7 tokens per step,
the floor on B70 is about 13.6 ms (598 GB/s), against about 18.7 ms measured.

Per family on B70 (share of its own bytes floor), as a guide to where a new device should land:

| family | per step on B70 | share of floor |
|---|---:|---:|
| `layer_post` x32 | 7.6 ms | about 67% |
| lm_head (int8), each call | 1.7 ms | about 100% |
| `layer_gdn` x24 | 2.4 ms | about 55% |
| MTP block GEMMs (oneDNN), per draft | 0.85 ms | about 92% |
| attention, 10 calls | 0.75 ms | about 33% |
| `layer_pre` x8 | 0.5 ms | about 64% |
| other small kernels | 0.26 ms | - |
| device idle | 0.3 ms | - |

A family much further below its B70 share on a new device is the place to look first. lm_head and
the oneDNN GEMMs show what the memory system can do with a good access pattern on that device.

## Profiles

- `MONO_TUNE_PROFILE=N` records an XPU-only profile of N verify steps, then
  `scripts/profile_summary.py profile-<n>.json` prints ms per step per kernel family, the device idle
  and where the gaps fall.
- Profiles with CPU activity inflate host-bound gaps (by 1–1.5 ms per step) and misplace them. Use the
  device-only profile for kernel time, and the observer's timings for host effects.
- Kernels replayed from an XPU graph do not appear in torch profiles; they show up as idle time.
- XPU events have a floor of about 45 us; time small kernels with the profiler's device time instead.
- Many small launches look worse in a profile than they are in the loop. Confirm with the
  observer's step time before acting on them.
- Instrumentation changes the persistent kernels' code generation (one register allocation for all
  phases). Compare clean builds; use in-kernel clocks only to locate time, not to compare variants.

## Host side

- The device MTP loop (`MONO_DECODE_XPU_DEVICE_MTP=1`, default) queues step n + 1 before it reads
  step n, which hides most host time. With it off, each step's sync idles the device for the
  Python path that follows (about 2 ms on the development host). A slower CPU shows up as device
  idle in the profile.
- Watch for per-token host readbacks (`.cpu()`, `.item()`, `tolist()`) appearing in a new
  configuration, for example in a memory mode that serves embedding rows from host memory: they
  serialize the loop.
- Do not remove or move the syncs that `auto` tree timing relies on. Removing the sampling wait
  while keeping host clocks made `auto` pick about one draft per step (fewer tokens per step,
  slower overall).
