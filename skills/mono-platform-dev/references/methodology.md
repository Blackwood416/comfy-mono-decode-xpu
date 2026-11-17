# Measuring and validating

## Benchmark protocol

- Measure in the real server with the real workflow, started the way the user starts it.
  Standalone scripts skip startup patches and measure a different stack.
- Load only the custom nodes the workflow needs. Other nodes that wrap generate or forward change
  what is measured.
- Give each configuration a fresh server. Configurations differ only in the variables under test;
  confirm the effective environment from the records.
- Vary only the prompt-enhance seed between runs.
- Report the first image apart from later images: the memory state, and with it the code path, can
  differ.
- Run warmups, then several measured runs, then the baseline again to see drift. If the baseline
  drifts, the result is inconclusive.
- Speculative decoding adapts to speed: for a kernel A/B, pin the MTP tree so every configuration
  does the same work per step. Report the adaptive result separately as the user-visible number.
- tok/s depends on the generated text (its length and acceptance). Do not claim end-to-end
  speedups when the outputs differ. Treat generations that hit `max_length` separately.

## Roof and attribution

- The roof is bytes moved per step divided by the measured read bandwidth. The bytes are the
  weights read once per verify and once per draft, plus the KV cache.
- Work out the bytes from the model and the code, then compare each kernel family with its own
  floor. Kernels already at the roof (for example large library GEMMs) show what the memory system
  can do on this device.
- Device-only profiles give kernel time. Profiles with CPU activity distort host gaps. Kernels
  replayed from graphs may not appear at all. Timing events have a floor, and instrumentation
  changes the code generated for persistent kernels.
- A microbenchmark result counts only after the workflow confirms it.

## Acceptance

- Unit tests pass on the platform, including new cases for any platform-specific branch.
- The official workflow completes in the memory mode the user runs, for the first image and for
  later images.
- Read every enhanced prompt and look at every image. A saved file is not a pass.
- Decode is faster at a pinned tree and under the adaptive tree, and the baseline drift is small.
- The report lists the platforms measured, and those not measured, for every change.
