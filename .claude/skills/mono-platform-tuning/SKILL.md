---
name: mono-platform-tuning
description: Bring up, benchmark and tune comfy-mono-decode-xpu on a specific Intel GPU (Arc B580, B60, B70 or another Xe2 device). Checks the device against the kernels' assumptions, builds for it, confirms which decode paths actually run, measures prompt-enhance decode against the device's bandwidth roof, and guides device-gated policy or kernel changes that keep the verify path bit-exact. Use when the user wants to run, port, profile, benchmark or speed up this node on their own hardware, or asks why it is slower on their card.
---

This skill is kept agent-neutral in `skills/mono-platform-tuning/` at the repository root. Read
`skills/mono-platform-tuning/SKILL.md` and follow it; its `reference/` and `scripts/` sit next to it.
