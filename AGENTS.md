# Notes for coding agents

- What the kernels do and the exactness boundary: `docs/DESIGN.md`. Every switch: `docs/ENV.md`.
  Tests and model validation: `tests/README.md`.
- Task guides live in `skills/`. Each has a `SKILL.md` with its steps, plus references and scripts
  next to it:
  - `skills/mono-platform-tuning/SKILL.md`: bring up, benchmark and tune this node on an Intel
    GPU (another card, driver or stack), including device-specific kernel or launch-policy changes.
