# NL Arm Controller — workshop demo

Voice/text-driven SO-101 controller. Speak or type a natural-language command,
an LLM translates it into a structured joint-motion plan, and the Cyberwave
SDK executes it on the digital twin (and optionally the physical SO-101).

## Tech stack

- **STT** — Mistral Voxtral (`voxtral-mini-latest`), spacebar push-to-talk.
- **Planner** — Anthropic Claude (text-only) → constrained JSON action plan.
- **Robot control** — Cyberwave Python SDK → MQTT → SO-101 follower.
- **TTS** — macOS `say`.

## Quick start

```bash
cp .env.example .env
# edit .env with your three API keys

pip install -r requirements.txt

python nl_arm_controller.py
```

The script in **Phase 1** just verifies your environment. Each subsequent
phase adds a layer:

| Phase | Adds |
|------:|------|
| 2 | Independent smoke tests for SDK + arm, Anthropic, Mistral STT, spacebar |
| 3 | Deterministic motion executor (mock JSON plans, no LLM) |
| 4 | Claude motion planner with constrained JSON output |
| 5 | Agent loop with typed input |
| 6 | Voice input via Mistral Voxtral |
| 7 | Hardening: clamps, fallbacks, sim/live toggle, kill switch |
| 8 | Rehearsal pack + day-of pre-flight |

## Mode

`CW_MODE=simulation` (default) drives only the 3D twin in the browser viewer.

`CW_MODE=live` also drives the physical SO-101 — requires `so101-remoteoperate`
running on the edge device wired to the arm.
