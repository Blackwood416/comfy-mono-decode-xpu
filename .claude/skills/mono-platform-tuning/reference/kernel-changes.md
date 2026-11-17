# Changing kernels or launch policy for a device

## Contract

- **Verify path: bitwise.** Every kernel gives the same bits as the operator chain it replaces,
  for the same inputs, weights, dtype and layout (docs/DESIGN.md, Exactness). A device-specific
  change may only change which threads do what and when, never the arithmetic or its order.
- **Decode attention:** checked against an fp64 reference, with Kitchen's own error on the same
  inputs as the bound.
- **Drafts:** may differ; judge by acceptance rate (`tokens_per_step`) and the outputs. The verify
  decides every token.
- **Scope:** kernel and scheduling work only. Do not prune the draft vocabulary or change what
  upstream drafts.

## Gating

- Put device-specific policy in host launch planning (`csrc/mono_launch.hpp`), keyed on the device
  id (`at::xpu::getDeviceProperties(...)->device_id`) or on a measured capability, like the B580
  chunk cap. Leave every other device's behaviour unchanged unless you measured it there: the
  B580 cap made B70 slower.
- A capacity or correctness fix (something that would fail or hang elsewhere too) belongs in the
  generic path: the accumulator growth started as a B580-only patch, had to be re-ported on every
  rebase, and was finally made generic with an explicit SLM check.
- Keep the before-launch checks: co-residency (work-groups <= Xe cores) and the total SLM footprint
  against `local_mem_size`. A failing check must disable only that path, with a reason in
  `stats()`.

## Persistent kernel rules (Xe2)

- Every work-group must be co-resident: at most one 1024-thread work-group per Xe core. Count the
  cores from `gpu_subslice_count` (or `max_compute_units / EUs per core`), not from an assumed ratio.
- The grid barrier polls with an atomic read-modify-write; a plain atomic load can hang.
- Fences: `grid_sync` writes its fences per role. If unchanged phases slow by several us after an
  edit, dump the assembly and count `fence.ugm.invalidate.gpu` before blaming SLM or registers.
- The whole kernel shares one register allocation and schedule: an edit in one phase moves the
  others by a few us. `noinline` helpers become stack calls and spill. 256 GRF removes spills but
  halves occupancy: `attn_dpas` got slower, and it did not help the GDN prototype.
- A prefetch is not free: the issuing thread's next device-scope fence (every barrier) waits for it.
  Prefetch traffic during latency-bound row phases delays those phases' own memory operations.
- Elementwise writes into BF16 joint_matrix A/B fragments (`joint_matrix_apply`) hang the device
  with the current compiler; stage through SLM or write the float accumulator.

## Build and compare

- Build with `build_icpx.sh` (one icpx invocation, compile and device link together). Torch's
  per-file SyclExtension build produced different device code for the same source, which breaks the
  bit-exact ports. `install.py --force --target <aot>` rebuilds; the default target `bmg` covers
  BMG-G21 and G31.
- One process per build in an A/B. Two extension modules in one process that define the same
  anonymous-namespace kernel class collide (the runtime finds kernels by mangled name) and both
  calls run one implementation; an identical-code control passes anyway, so it hides the bug.
  Rename the kernel classes (`-DLayerGdnKernel=...`) if both must load together.
- Assembly: `IGC_ShaderDumpEnable=1 IGC_DumpToCustomDir=DIR` at build time; look at spills, SWSB
  tokens and fences of the kernel you changed.
- Re-run `tests/` after every kernel change (`PYTHONPATH=/path/to/ComfyUI python -m pytest -q tests`,
  in the same environment ComfyUI starts with). Add a case for the device-specific branch (for
  example a worker count or chunk count that forces it on any device).
- Do not edit the node's Python while a benchmark server runs: a new process would load the new
  Python against the old extension. Do not `pkill -f` a pattern that also matches your own shell.
- Microbenchmarks: hold the inputs real (captured from a generate) and confirm the speedup in the
  workflow. A microbenchmark and the workflow disagreed by 3x once; the workflow wins.
