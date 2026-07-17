"""LLM motion planner — Claude turns natural language into a validated `MotionPlan`.

Architecture:
  utterance ──► Claude (constrained JSON) ──► parse_plan_json ──► MotionPlan
                                                       │
                                                       ▼
                                              validate_plan (Phase 3)
                                                       │
                                                       ▼
                                              MotionExecutor.execute

The system prompt:
  * pins the JSON schema and forbids markdown/prose
  * lists the 6 joints with directional semantics (so "turn right" maps to a
    positive joint-1 angle)
  * shows three few-shot examples covering single-joint, multi-joint, and
    "stop" semantics
  * tells Claude to always return *some* valid plan (preferring a small
    conservative gesture) rather than refusing — refusals are useless to the
    executor

Anything Claude returns that doesn't pass `validate_plan` is rejected with an
error message; the agent loop in Phase 5 will turn that into a spoken
"sorry, I couldn't plan that" response, never an arm motion.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass

from motion import MotionPlan, validate_plan
from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.any_llm_model import AnyLLMModel

set_tracing_disabled(disabled=True)


SYSTEM_PROMPT = """You are the motion planner for an SO-101 6-axis robot arm.

You translate the user's natural-language request into a JSON motion plan that
will drive the arm. You do nothing else — no chat, no apologies, no markdown.

The arm has 6 revolute joints, named "1" through "6":
  "1" — base rotation         (±90°)   positive = turn RIGHT (clockwise from above)
  "2" — shoulder pitch        (±60°)   positive = lift arm UP
  "3" — elbow                 (±60°)   positive = extend FORWARD
  "4" — wrist pitch           (±60°)   positive = wrist UP
  "5" — wrist roll            (±60°)   positive = roll CW
  "6" — gripper / wrist yaw   (±60°)

Output format — return EXACTLY one JSON object, no code fences, no commentary:

{
  "say": "<one short sentence describing the motion>",
  "actions": [
    { "type": "set_joint", "joint": "1", "angle": 30, "duration": 1.5 },
    { "type": "set_pose",  "pose": {"1": 30, "2": -20}, "duration": 2.0 },
    { "type": "wait",      "duration": 0.5 },
    { "type": "home",      "duration": 1.5 }
  ]
}

Rules:
- "actions" must contain 1–8 entries.
- Every "duration" is seconds, range 0.0–5.0 per action.
- "angle" is degrees. Stay within the per-joint range above. ±20°–45° looks expressive.
- Use "set_joint" for single-joint moves, "set_pose" for coordinated multi-joint moves,
  "wait" for pauses, "home" to return every joint to 0°.
- ALWAYS finish with a return-toward-zero (either "home" or a final move to 0°)
  unless the user explicitly asks the arm to hold a pose.
- If the request is unsafe, ambiguous, or impossible, still return a valid plan —
  pick a small conservative gesture and explain it in "say".
- NEVER output prose outside the JSON object.

Few-shot examples:

User: "wave at the audience"
{"say":"Waving from my base.","actions":[{"type":"set_joint","joint":"1","angle":30,"duration":0.7},{"type":"set_joint","joint":"1","angle":-30,"duration":1.0},{"type":"set_joint","joint":"1","angle":30,"duration":1.0},{"type":"set_joint","joint":"1","angle":0,"duration":0.7}]}

User: "look up and to the right"
{"say":"Tilting up and turning right.","actions":[{"type":"set_pose","pose":{"1":25,"2":-25},"duration":1.5},{"type":"wait","duration":0.5},{"type":"home","duration":1.5}]}

User: "stop"
{"say":"Stopping and going home.","actions":[{"type":"home","duration":1.0}]}
"""


@dataclass
class PlanResult:
    """Outcome of a single planner call."""

    plan: MotionPlan | None
    raw_response: str
    error: str | None
    model: str

    @property
    def ok(self) -> bool:
        return self.plan is not None and self.error is None


_FENCE_HEAD = re.compile(r"^```(?:json)?\s*", re.IGNORECASE)
_FENCE_TAIL = re.compile(r"\s*```$")


def parse_plan_json(raw: str) -> tuple[MotionPlan | None, str | None]:
    """Best-effort extract a JSON object from `raw` and convert to MotionPlan.

    Returns (plan, None) on success, (None, error_message) on failure.
    """
    text = (raw or "").strip()
    if not text:
        return None, "empty response"

    text = _FENCE_HEAD.sub("", text)
    text = _FENCE_TAIL.sub("", text)

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None, f"no JSON object found in response: {raw[:200]!r}"

    blob = text[start : end + 1]

    try:
        data = json.loads(blob)
    except json.JSONDecodeError as exc:
        return None, f"JSON decode error: {exc}  (blob: {blob[:200]!r})"

    if not isinstance(data, dict):
        return None, f"top-level JSON must be an object, got {type(data).__name__}"

    try:
        plan = MotionPlan.from_dict(data)
    except (KeyError, TypeError, ValueError) as exc:
        return None, f"plan shape error: {exc}  (data: {data!r})"

    errors = validate_plan(plan)
    if errors:
        return None, "validation failed:\n  - " + "\n  - ".join(errors)

    return plan, None


def plan_from_utterance(
    utterance: str,
    *,
    model: str | None = None,
    max_tokens: int = 400,
    temperature: float = 0.2,
) -> PlanResult:
    """Plan from text.

    Prefers OpenRouter (via the OpenAI Agents SDK's `AnyLLMModel`) when
    `OPENROUTER_API_KEY` is set; otherwise falls back to calling Claude
    directly with `ANTHROPIC_API_KEY`.
    """
    if os.environ.get("OPENROUTER_API_KEY"):
        chosen_model = _openrouter_model_name(model)
        raw = _run_any_llm_agent(
            utterance,
            system_prompt=SYSTEM_PROMPT,
            model=chosen_model,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        plan, err = parse_plan_json(raw)
        return PlanResult(plan=plan, raw_response=raw, error=err, model=chosen_model)

    raw, chosen_model = _call_anthropic(
        utterance,
        system_prompt=SYSTEM_PROMPT,
        model=model,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    plan, err = parse_plan_json(raw)
    return PlanResult(plan=plan, raw_response=raw, error=err, model=chosen_model)


def _openrouter_model_name(model: str | None) -> str:
    chosen = model or os.environ.get("OPENROUTER_MODEL", "openai/gpt-5.4-mini")
    return chosen if chosen.startswith("openrouter/") else f"openrouter/{chosen}"


def _run_any_llm_agent(
    utterance: str,
    *,
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
) -> str:
    """Run a single-turn agent via `AnyLLMModel` (OpenAI Agents SDK) and return its raw text."""
    agent = Agent(
        name="MotionPlanner",
        instructions=system_prompt,
        model=AnyLLMModel(model=model, api_key=os.environ["OPENROUTER_API_KEY"]),
        model_settings=ModelSettings(max_tokens=max_tokens, temperature=temperature),
    )
    result = asyncio.run(Runner.run(agent, utterance))
    return str(result.final_output or "")


def _call_anthropic(
    utterance: str,
    *,
    system_prompt: str,
    model: str | None,
    max_tokens: int,
    temperature: float,
) -> tuple[str, str]:
    """Call Claude directly with `utterance`, picking up `ANTHROPIC_API_KEY` via the SDK default."""
    import anthropic

    client = anthropic.Anthropic()
    chosen_model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    response = client.messages.create(
        model=chosen_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": utterance}],
    )

    raw = "".join(
        getattr(block, "text", "") for block in response.content
        if getattr(block, "type", None) == "text"
    )
    return raw, chosen_model


# ---------------------------------------------------------------------------
# Vision-aware planning
# ---------------------------------------------------------------------------

VISION_SYSTEM_PROMPT = """You are the motion planner AND scene narrator for an SO-101 6-axis robot arm.
You receive an image from the arm's workspace camera AND a natural-language
request from the operator. Your job is to return a single JSON object that
either describes what you see, plans an arm motion, or does both.

The arm has 6 revolute joints, named "1" through "6":
  "1" — base rotation         (±90°)   positive = turn RIGHT (clockwise from above)
  "2" — shoulder pitch        (±60°)   positive = lift arm UP
  "3" — elbow                 (±60°)   positive = extend FORWARD
  "4" — wrist pitch           (±60°)   positive = wrist UP
  "5" — wrist roll            (±60°)   positive = roll CW
  "6" — gripper / wrist yaw   (±60°)

The camera is mounted near the operator looking at the arm's workspace.
"left" / "right" in your descriptions refer to the audience's view, which
matches the camera frame.

Output format — return EXACTLY one JSON object, no code fences, no markdown,
no commentary outside the JSON. Schema:

{
  "say":     "<the spoken response — describe the scene, answer the question,\
 or narrate the motion>",
  "actions": [ ... 0 to 8 motion actions, same shape as the text-only planner ... ]
}

Action shapes (any combination, in order):
  { "type": "set_joint", "joint": "1"-"6", "angle": -90..90, "duration": 0.1..5.0 }
  { "type": "set_pose",  "pose": {"1": deg, "2": deg, ...},  "duration": 0.1..5.0 }
  { "type": "wait",                                         "duration": 0.1..5.0 }
  { "type": "home",                                         "duration": 0.1..5.0 }

Decision rules:

1. If the operator asks ABOUT the scene ("what do you see?", "is there a red
   cup?", "where's my notebook?"), respond ONLY with description in `say` and
   an empty `actions` array. NO MOTION for purely informational questions.

2. If the operator asks for MOTION ("wave at the audience", "do a small bow"),
   plan motion in `actions`. The `say` should briefly narrate what you'll do.

3. If the operator asks for VISUALLY-GROUNDED MOTION ("wave at the red cup",
   "look at the laptop"), describe what you see in `say`, then plan a small
   gesture toward the relevant area. Without precise camera-arm calibration
   you cannot point exactly — aim with joint "1" (base rotation) toward the
   approximate horizontal direction (left = negative, right = positive,
   typically ±20° to ±40°).

4. If the operator references something you DON'T see, say so honestly and
   leave `actions` empty. Do not pretend or hallucinate motion.

5. ALWAYS return to a near-zero pose at the end of any motion (either via a
   final "home" action or a final move to 0°) unless the operator explicitly
   says to hold a pose.

6. NEVER output prose outside the JSON object. NEVER use code fences.

Few-shot examples:

User says: "what's on the table?"
{"say":"I see a red cup on the right side, a laptop in the middle, and a blue notebook to the left.","actions":[]}

User says: "wave at the audience"
{"say":"Waving from my base.","actions":[{"type":"set_joint","joint":"1","angle":30,"duration":0.7},{"type":"set_joint","joint":"1","angle":-30,"duration":1.0},{"type":"set_joint","joint":"1","angle":30,"duration":1.0},{"type":"set_joint","joint":"1","angle":0,"duration":0.7}]}

User says: "look at the red cup"
{"say":"I see a red cup on the right. Turning toward it.","actions":[{"type":"set_pose","pose":{"1":35,"2":-15},"duration":1.5},{"type":"wait","duration":0.5},{"type":"home","duration":1.5}]}

User says: "do you see a banana?"
{"say":"No, I don't see a banana — I see a red cup, a laptop, and a notebook.","actions":[]}
"""


def plan_from_utterance_with_image(
    utterance: str,
    frame_b64_jpeg: str | None,
    *,
    model: str | None = None,
    max_tokens: int = 500,
    temperature: float = 0.2,
) -> PlanResult:
    """Call Claude Vision with `utterance` + the image, return a `PlanResult`.

    Vision uses Claude's multimodal API directly (`ANTHROPIC_API_KEY`), since
    that's the format wired up here. If no fresh frame is available, or only
    `OPENROUTER_API_KEY` is configured (no vision path for it yet), this
    falls back to the text-only planner via `plan_from_utterance` so the
    agent stays usable.
    """
    if frame_b64_jpeg is None or not os.environ.get("ANTHROPIC_API_KEY"):
        return plan_from_utterance(utterance, model=model, max_tokens=max_tokens, temperature=temperature)

    import anthropic

    client = anthropic.Anthropic()
    chosen_model = model or os.environ.get("ANTHROPIC_MODEL", "claude-sonnet-4-5")

    response = client.messages.create(
        model=chosen_model,
        max_tokens=max_tokens,
        temperature=temperature,
        system=VISION_SYSTEM_PROMPT,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": frame_b64_jpeg,
                        },
                    },
                    {"type": "text", "text": utterance},
                ],
            }
        ],
    )

    raw = "".join(
        getattr(block, "text", "") for block in response.content
        if getattr(block, "type", None) == "text"
    )

    plan, err = parse_plan_json(raw)
    return PlanResult(plan=plan, raw_response=raw, error=err, model=chosen_model)
