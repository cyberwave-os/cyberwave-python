# Cyberwave Python SDK: custom Claude review policy

Review only defects introduced by the supplied changes. Prioritize correctness
and the contributor's ability to act on the feedback. Be respectful and concise.

## SDK compatibility

- Preserve public imports, signatures, return types, and documented behavior.
- Keep handwritten code compatible with Python 3.10 through 3.14.
- Optional camera, ML, Zenoh, and hardware packages must remain optional. Watch
  for new eager imports that break `import cyberwave` in a minimal installation.
- Generated REST client changes usually belong in the generator or source schema;
  do not recommend editing generated files blindly. They are excluded from the
  supplied patch set and need separate human review.

## Connections and async behavior

- Check cleanup of MQTT, REST, streaming, and data-bus resources on errors and exit.
- Look for event-loop blocking, orphan tasks, duplicate callbacks, and reconnect
  paths that lose subscriptions or leak resources.
- Check timeout/error handling and preservation of useful exception context.

## Robotics behavior

- Changes must preserve the distinction between simulation and live hardware.
- Look for unintended live motion, incorrect units, wrong twin/environment
  routing, and regressions in stop, cancellation, or readiness handling.
- Never suggest testing a patch by commanding real hardware from CI.

## Tests and documentation

- When a concrete defect is visible, describe a focused regression test alongside
  the fix. Do not emit generic 'add tests' findings without explaining a risk.
- Flag an example or documented call only when the supplied changes demonstrably
  make it incorrect. Missing context alone is not a bug.
- Skip stylistic preferences, wholesale refactors, and unrelated preexisting bugs.

## Findings

Return at most three high-confidence findings. Each must name an exact supplied
file path and explain the trigger, user-visible consequence, and a practical fix.
Use `high` for serious behavioral breakage and `medium` for a narrower concrete
defect. Return no findings when the evidence is insufficient. Acknowledge that
this is a review of patch excerpts; do not claim to have run tests or inspected
the complete repository.
