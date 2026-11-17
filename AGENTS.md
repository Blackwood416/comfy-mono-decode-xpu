# Notes for coding agents

- What the kernels do and the exactness boundary: `docs/DESIGN.md`. Every switch: `docs/ENV.md`.
  Tests and model validation: `tests/README.md`.
- Task guides live in `skills/`, each a `SKILL.md` with references and scripts next to it
  (`.claude/skills/` and `.codex/skills/` hold stubs that point there, for agents that scan those;
  one stub per folder, so agents that scan several of them do not see the skill twice):
  - `skills/mono-platform-dev/SKILL.md`: developing, porting, bringing up and tuning this node on
    an Intel GPU platform other than the validated one (hardware, driver, OS or stack).
