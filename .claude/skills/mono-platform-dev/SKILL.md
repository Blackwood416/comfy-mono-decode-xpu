---
name: mono-platform-dev
description: Develop, port, bring up and tune comfy-mono-decode-xpu on an Intel GPU platform other than the one it was validated on (another card, architecture, driver, OS or ComfyUI / Torch stack). Covers checking the platform against the kernels' assumptions, building and testing, confirming which decode paths run, measuring decode against the bandwidth roof, and making platform-specific policy or kernel changes that keep the verify path exact. Use for any work on this node for new hardware or a new stack, or when it underperforms there.
---

This skill is kept agent-neutral in `skills/mono-platform-dev/` at the repository root. Read
`skills/mono-platform-dev/SKILL.md` and follow it; its `reference/` and `scripts/` sit next to it.
