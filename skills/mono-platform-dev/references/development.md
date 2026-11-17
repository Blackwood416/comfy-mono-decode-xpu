# Developing for a platform

## Contract

- **Verify path, except decode attention:** bitwise identical to the operator chain it replaces
  on the target platform, for the same inputs, weights, dtype and layout (docs/DESIGN.md,
  Exactness). A platform change may move work between threads and in time, but must not change
  the arithmetic or its order.
- **Decode attention:** within the fp64-referenced error bound, with the replaced kernel's own
  error on the same inputs as the limit; `MONO_DECODE_XPU_ATTN_DPAS=0` keeps the bit-identical
  port available.
- **Drafts:** may differ, judged by acceptance and the outputs; the verify decides every token.
- **New exceptions:** a maintainer decision, not a platform-local one. It needs a stated acceptance
  criterion, an entry in docs/DESIGN.md and a switch back to the exact path.
- **Errors:** stateful dispatch (after KV, the recurrent state or x may have been written)
  propagates errors. Pure operations may disable their own route and fall back. Cancellation
  never disables a path.

## Scope and gating

- Prefer configuration, then host launch policy, then kernel changes, in that order.
- Gate platform-specific behaviour on the device (an id, a family or a measured capability). A
  validated platform's behaviour stays unchanged unless it is measured again.
- A fix for a failure that could happen on any device belongs in the generic path, with an explicit
  before-launch check.
- Keep target-specific code in its own modules and sources, so the validated kernels build
  unchanged.

## Porting

- First confirm what the stock path does on the platform (`platform.md`).
- Bring the node up piece by piece. The pieces in `__init__.py` and the switches in docs/ENV.md are
  independent:
  1. stateless kernels;
  2. decode-loop pieces;
  3. the persistent layer kernels, which depend most on the hardware.

  Each piece must be correct and not slower than stock before the next.
- For a target with a different ISA:
  - give it its own build target;
  - compile only the sources ported to it;
  - record the target in the build, so a binary for another target is never reused;
  - export which pieces the build supports, and install only those.

## Engineering hygiene

- Build with `build_icpx.sh` (a single compiler invocation); other build paths can produce different
  device code.
- One process per build in an A/B: two extension modules with the same kernel names can silently
  share one implementation.
- Run new persistent kernels in a separate process under a timeout; a barrier that cannot complete
  hangs inside the driver.
- Compare clean builds. Inspect the assembly (spills, fences, memory tokens) when a change moves time
  in code it did not touch.
- Do not edit the node's Python while a benchmark server runs.
- Re-run `tests/` after every change, and the validated reference device's tests when shared code
  changed.
