# Tuning levers, in the order to try them

Each lever lists what it changes, how to see it, and what development found. Change one at a time
and measure with the protocol in measurement.md.

## 1. Make sure the fast paths run

Nothing else matters until they do. In the records' `stats.layer`:
- `layer`, `gdn1`, `attn_dpas`, `norm_rope` count up with every generate;
- `fallback` is empty or explains itself: the `weights` entry counts the BF16 MTP block, which never
  takes the layer path;
- `disabled` is empty.

A persistent launch that does not fit the device (SLM, co-residency) disables that path; the
reason is in `fallback` and in the server log. On a new device this is the most likely first
finding (B580's first port ran with the layer path off).

## 2. Weight layout and memory

- `stats.layer.weight_layout` shows the per-generate decision: tile-major copies (fast) or
  row-major checkpoint weights. The log line `[mono_decode_xpu] weight layout=...` prints copies and
  available bytes.
- The copy needs about 3.9 GB for Qwen3.5-9B W4A8, decided after prefill from the actual weight
  storage, the allocator pool, the inference reserve and `MONO_DECODE_XPU_TILE_MARGIN_GB` (default
  3). Under the AIMDO allocator hook it also counts reclaimable pages of inactive resident models.
- Typical outcome: the first image fits and later images do not, or a 12 GB card never fits in
  the full workflow. Prompt enhance alone (`--pe-only`) usually fits.
- Levers: a smaller margin (verify no OOM in the full workflow, all images); the memory mode the
  user actually runs (DynamicVRAM default, `--disable-dynamic-vram`, `--reserve-vram`). Do not add a
  partial tile cache: on B580 it was slower than none.
- On row-major weights `layer_post` dominates the step; measure it there, not on tile-major numbers.

## 3. Draft lm_head (`MONO_DECODE_XPU_DRAFT_INT4=1`)

Experimental, off by default. An int4 copy (about 0.55 GB more per generate, when free memory
allows) scores the vocabulary and the exact int8 logits of about 1000 candidates pick the drafts.
It halves the draft lm_head bytes: on B70 about +10% decode. Only drafts change; the verify still
decides every token. `=check` runs both and counts mismatches in `mtp_loop.stats()`; validate with it
on a new device before reporting numbers.

## 4. MTP tree

`auto` scores width chains by accepted tokens per millisecond from device timings
(`autotree.stats()` shows the draft and verify cost it used). A device with a different
draft/verify cost ratio settles on a different tree. Look at `tokens_per_step` and
`drafts_per_step` before attributing a change to a kernel; compare kernels at a fixed tree.

## 5. Attention chunks

`attention_chunks` in `csrc/mono_launch.hpp` splits the context into at most 48 chunks of at least
32 tokens, one work-group per (kv head, chunk). Long contexts (H3 thinking, 4k–8k tokens) make the
verify attention visible in the profile.
- B580 caps the 8-row verify pack at one chunk per Xe core (-33% at an 8k prefix); on B70 the same
  cap was slower. The rule is gated on the device id.
- To try a cap on a new device, add a device-gated branch next to the B580 one, rebuild, and time
  verify and draft attention at 1k / 4k / 8k contexts. Keep the draft row's policy unless it is
  measured too.

## 6. Work-groups per launch

The persistent kernels use one work-group per Xe core (`resident_wgs`). Fewer work-groups give each
a larger share of the tiles and more SLM; more than the core count can deadlock. The tests run
16 workers in both layouts to cover growth. A different count is a code change in the launch
planning, gated per device, with the SLM check left in place.

## 7. Runtime and memory modes

- Level Zero v2 is required for graphs; the probe shows the platform name.
- ComfyUI's `--async-offload 2 --force-non-blocking` lane uses prepared CombinedLinear weights.
  Record it apart from the default lane, and check `stats.mono["combined"]` when groups are
  prepared.
- Experimental switches in docs/ENV.md (`MONO_DECODE_XPU_GRAPH`, `_DEVICE_MTP_GRAPH`) gave no gain on
  B70 with the layer kernels; try them last.

## 8. Kernel work

Only when a family sits well below its share of the floor and the levers above are exhausted.
Follow kernel-changes.md. On B70 the remaining gaps are known and hard: `layer_post` GEMV decode
(the codebook lookups per weight), `layer_gdn` serial phases and long-context attention.
