# What depends on the device

The kernels were developed on an Arc Pro B70 and ported to an Arc B580 and an Arc Pro B60. This
page lists every device assumption the code makes, then what each port found. Re-measure on
your own card: driver, stack and workflow all move the numbers.

## Assumptions in the code

| assumption | where | what breaks otherwise |
|---|---|---|
| 8 vector engines per Xe core: resident work-groups = `max_compute_units / 8` | `csrc/mono_launch.hpp` `resident_wgs`, `csrc/attn_port.sycl` | Persistent kernels spin on a grid barrier, so every work-group must be resident at once. On a device with 16 EUs per core (DG2), `/8` launches twice as many work-groups as cores and can deadlock. |
| One 1024-thread work-group per Xe core, SIMD16 sub-groups | persistent kernels | Launch fails or does not co-reside. |
| DPAS with N = 16 and 2D block IO (Xe2) | GEMV engine, `attn_dpas`, rotations | DG2 has DPAS with N = 8 and no 2D block IO: the persistent kernels and `attn_dpas` do not apply. |
| SLM of 128 KiB per work-group | persistent kernels | They need roughly 100–125 KB per work-group, more when fewer cores share the tiles. `plan_streams` rejects a launch whose SLM exceeds `local_mem_size` before it starts (the layer path then falls back and reports it). |
| Enough free VRAM for a per-generate tile-major copy of the post-block weights (about 3.9 GB for Qwen3.5-9B W4A8) | `comfy_mono_decode_xpu/tiles.py` | Without room the kernels read row-major weights (`layer.stats()["weight_layout"]`), which is slower. |
| Attention chunk count: up to 48 chunks, at least 32 tokens each | `mono_launch.hpp` `attention_chunks` | Only a scheduling choice. B580 caps the largest verify pack at one chunk per Xe core (gated on device id 0xE20B). |

`scripts/platform_probe.py` checks the first four and the memory size.

## Development devices

Reference points, not targets. "H3 PE decode" is the official MiniMax H3 Image template's prompt
enhance (thinking, sampling, `auto` MTP) in DynamicVRAM. The mono-off column has `MONO_DECODE_XPU=0`.

| | Arc Pro B70 | Arc B580 | Arc Pro B60 |
|---|---|---|---|
| device id | 0xE223 (BMG-G31) | 0xE20B (BMG-G21) | 0xE211 (BMG-G21) |
| Xe cores / SLM / VRAM | 32 / 128 KiB / 32 GB | 20 / 128 KiB / 12 GB | 20 / 128 KiB / 24 GB |
| read bandwidth (`read_bandwidth.py` class of test) | about 595 GB/s | about 453 GB/s | not measured |
| AOT target | `bmg` (default) | `bmg` | `bmg` |
| device-specific code | none (development device) | attention chunk cap for the largest verify pack | none |
| weight layout in the full workflow | tile-major on every image | row-major (no room for the copy) | tile-major |
| H3 PE decode, mono on / off | about 140–149 / 24–28 tok/s | about 91–98 / 32 tok/s | about 94–99 / 27 tok/s |
| share of the bytes floor reached | about 73% of a 598 GB/s floor per step | about 63–67% | not measured |

Port notes:
- **B580.** With 20 work-groups the busiest work-group of the gate / up stream owns 78 tiles and
  needs 9984 accumulator elements, more than the original fixed 8192. The launch failed its
  capacity check and the layer path silently turned off. Accumulators now grow to the stream share
  on every device, checked against `local_mem_size` before launch. The chunk cap cut the 8-row
  verify attention at an 8k prefix from 0.311 to 0.209 ms; on B70 the same cap (32 or 24 chunks)
  made verify and draft attention slower, which is why it is gated on the device id. A partial tile
  cache for the row-major case was slower than none and was removed.
- **B60.** No code change. The accumulator SLM at 20 / 18 / 16 work-groups is 110 / 114 / 119 KB
  for `layer_post` and 119 / 123 / 128 KB for `layer_gdn`, all inside 128 KiB. Neighbouring worker
  counts, row versus tile and chunk caps were not measured.

## Devices outside the validated set

- **Other Xe2 (BMG, Lunar Lake).** Expect the code to run; run the whole bring-up before trusting
  numbers. Integrated GPUs share system memory: the bandwidth roof is far lower and DynamicVRAM
  behaves differently. AOT targets: `ocloc compile --help` lists the device names (`bmg`, `bmg-g21`,
  `bmg-g31`, `lnl-m`, `ptl-h`, ...).
- **Xe3 (Panther Lake) and later.** Check the probe's co-residency and ISA lines first; untested.
- **DG2 / Alchemist (Arc A770, 0x56A0).** Not supported by these kernels: 16 EUs per Xe core,
  64 KiB SLM, DPAS with N = 8 and no 2D block IO. A port needs SG8 kernels, `/16` residency and
  smaller SLM footprints. Lessons from a prototype port:
  - Use the Level Zero v2 adapter (`SYCL_UR_USE_LEVEL_ZERO_V2=1 UR_LOADER_USE_LEVEL_ZERO_V2=1`);
    v1 has no graph support. Check the platform name in the probe.
  - `torch._int_mm` returned wrong results for N = 8, K = 4096.
  - Single-API ratios (71–85% of a roof) said nothing about whole-decode time, which stayed
    near 27–40% of the floor.
- **Multiple XPUs.** Pin the card under test (`ZE_AFFINITY_MASK`) and map cards by UUID, not by
  index. An idle card still shows some memory in use, not 0.
