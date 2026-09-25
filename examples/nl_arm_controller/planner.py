"""LLM motion planner — Claude turns natural language into a validated `MotionPlan`.

Architecture:
  utterance ──► Claude (constrained JSON) ──► parse_plan_json ──► MotionPlan
                                                       │
                                                       ▼
                                              validate_plan (motion.py)
                                                       │
                                                       ▼
                                              MotionExecutor.execute

The system prompt:
  * pins the JSON schema and forbids markdown/prose
  * lists the PiPER's joints with directional semantics, generated from
    `motion.joint_table_for_prompt()` so the prompt can never drift out of
    sync with the limits the executor enforces
  * shows few-shot examples covering single-joint, multi-joint, gripper, and
    "stop" semantics
  * tells Claude to always return *some* valid plan (preferring a small
    conservative gesture) rather than refusing — refusals are useless to the
    executor

Anything Claude returns that doesn't pass `validate_plan` is rejected with an
error message; the agent loop turns that into a spoken "sorry, I couldn't plan
that" response, never an arm motion.
"""

from __future__ import annotations

import asyncio
import json
import os
import re
from dataclasses import dataclass

from motion import MotionPlan, validate_plan
from planner_config import get_openrouter_model
from agents import Agent, ModelSettings, Runner, function_tool, set_tracing_disabled
from agents.extensions.models.any_llm_model import AnyLLMModel

set_tracing_disabled(disabled=True)


_JOINT_TABLE = joint_table_for_prompt()

_ACTION_SHAPES = f"""Action shapes (any combination, in order):
  {{ "type": "set_joint",   "joint": "joint1".."joint6", "angle": <deg>, "duration": 0.1..{MAX_DURATION_S} }}
  {{ "type": "set_pose",    "pose": {{"joint1": <deg>, "joint2": <deg>, ...}}, "duration": 0.1..{MAX_DURATION_S} }}
  {{ "type": "set_gripper", "opening": 0..100, "duration": 0.1..{MAX_DURATION_S} }}
  {{ "type": "wait",                                     "duration": 0.1..{MAX_DURATION_S} }}
  {{ "type": "home",                                     "duration": 0.1..{MAX_DURATION_S} }}"""


SYSTEM_PROMPT = f"""You are the motion planner for an AgileX PiPER 6-axis robot arm with a parallel gripper.

You translate the user's natural-language request into a JSON motion plan that
will drive the arm. You do nothing else — no chat, no apologies, no markdown.

The arm has 6 revolute joints plus a gripper:
{_JOINT_TABLE}

Note the asymmetric joints: "joint2" only moves 0° → positive (shoulder sweeps
forward from vertical) and "joint3" only moves 0° → negative (elbow folds
back). A "reach forward" is joint2 positive together with joint3 negative.

Output format — return EXACTLY one JSON object, no code fences, no commentary:

{{
  "say": "<one short sentence describing the motion>",
  "actions": [
    {{ "type": "set_joint",   "joint": "joint1", "angle": 30, "duration": 1.5 }},
    {{ "type": "set_pose",    "pose": {{"joint2": 30, "joint3": -40}}, "duration": 2.0 }},
    {{ "type": "set_gripper", "opening": 100, "duration": 0.6 }},
    {{ "type": "wait",        "duration": 0.5 }},
    {{ "type": "home",        "duration": 1.5 }}
  ]
}}

{_ACTION_SHAPES}

Rules:
- "actions" must contain 1–{MAX_ACTIONS_PER_PLAN} entries.
- Every "duration" is seconds, range 0.0–{MAX_DURATION_S} per action.
- "angle" is degrees. Stay within the per-joint range above. 20°–45° of travel
  looks expressive on this arm.
- Use "set_joint" for single-joint moves, "set_pose" for coordinated multi-joint
  moves, "set_gripper" for grasp/release, "wait" for pauses, "home" to return
  every arm joint to 0°.
- NEVER put "{GRIPPER.name}" in a "set_joint" or "set_pose" action — the gripper
  is only reachable through "set_gripper", in percent open.
- ALWAYS finish with a return-toward-zero (either "home" or a final move to 0°)
  unless the user explicitly asks the arm to hold a pose.
- If the request is unsafe, ambiguous, or impossible, still return a valid plan —
  pick a small conservative gesture and explain it in "say".
- NEVER output prose outside the JSON object.

Few-shot examples:

User: "wave at the audience"
{{"say":"Waving from my base.","actions":[{{"type":"set_joint","joint":"joint1","angle":30,"duration":0.7}},{{"type":"set_joint","joint":"joint1","angle":-30,"duration":1.0}},{{"type":"set_joint","joint":"joint1","angle":30,"duration":1.0}},{{"type":"set_joint","joint":"joint1","angle":0,"duration":0.7}}]}}

User: "reach forward and to the left"
{{"say":"Reaching forward and turning left.","actions":[{{"type":"set_pose","pose":{{"joint1":25,"joint2":40,"joint3":-50}},"duration":2.0}},{{"type":"wait","duration":0.5}},{{"type":"home","duration":2.0}}]}}

User: "open the gripper"
{{"say":"Opening the gripper.","actions":[{{"type":"set_gripper","opening":100,"duration":0.8}}]}}

User: "grab it"
{{"say":"Closing the gripper on it.","actions":[{{"type":"set_gripper","opening":100,"duration":0.6}},{{"type":"set_pose","pose":{{"joint2":35,"joint3":-45}},"duration":1.8}},{{"type":"set_gripper","opening":0,"duration":0.8}},{{"type":"home","duration":2.0}}]}}

User: "stop"
{{"say":"Stopping and going home.","actions":[{{"type":"home","duration":1.0}}]}}
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
    chosen = model or get_openrouter_model()
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

VISION_SYSTEM_PROMPT = f"""You are the motion planner AND scene narrator for an AgileX PiPER 6-axis robot
arm with a parallel gripper. You receive an image from the arm's workspace
camera AND a natural-language request from the operator. Your job is to return
a single JSON object that either describes what you see, plans an arm motion,
or does both.

The arm has 6 revolute joints plus a gripper:
{_JOINT_TABLE}

Note the asymmetric joints: "joint2" only moves 0° → positive (shoulder sweeps
forward from vertical) and "joint3" only moves 0° → negative (elbow folds
back). A "reach forward" is joint2 positive together with joint3 negative.

The camera is mounted near the operator looking at the arm's workspace.
"left" / "right" in your descriptions refer to the audience's view, which
matches the camera frame.

Output format — return EXACTLY one JSON object, no code fences, no markdown,
no commentary outside the JSON. Schema:

{{
  "say":     "<the spoken response — describe the scene, answer the question, or narrate the motion>",
  "actions": [ ... 0 to {MAX_ACTIONS_PER_PLAN} motion actions, same shape as the text-only planner ... ]
}}

{_ACTION_SHAPES}

Decision rules:

1. If the operator asks ABOUT the scene ("what do you see?", "is there a red
   cup?", "where's my notebook?"), respond ONLY with description in `say` and
   an empty `actions` array. NO MOTION for purely informational questions.

2. If the operator asks for MOTION ("wave at the audience", "do a small bow"),
   plan motion in `actions`. The `say` should briefly narrate what you'll do.

3. If the operator asks for VISUALLY-GROUNDED MOTION ("wave at the red cup",
   "look at the laptop"), describe what you see in `say`, then plan a small
   gesture toward the relevant area. Without precise camera-arm calibration
   you cannot point exactly — aim with "joint1" (base rotation) toward the
   approximate horizontal direction (operator's left = positive, right =
   negative, typically 20° to 40°).

4. If the operator references something you DON'T see, say so honestly and
   leave `actions` empty. Do not pretend or hallucinate motion.

5. NEVER put "{GRIPPER.name}" in a "set_joint" or "set_pose" action — the gripper
   is only reachable through "set_gripper", in percent open.

6. ALWAYS return to a near-zero pose at the end of any motion (either via a
   final "home" action or a final move to 0°) unless the operator explicitly
   says to hold a pose.

7. NEVER output prose outside the JSON object. NEVER use code fences.

Few-shot examples:

User says: "what's on the table?"
{{"say":"I see a red cup on the right side, a laptop in the middle, and a blue notebook to the left.","actions":[]}}

User says: "wave at the audience"
{{"say":"Waving from my base.","actions":[{{"type":"set_joint","joint":"joint1","angle":30,"duration":0.7}},{{"type":"set_joint","joint":"joint1","angle":-30,"duration":1.0}},{{"type":"set_joint","joint":"joint1","angle":30,"duration":1.0}},{{"type":"set_joint","joint":"joint1","angle":0,"duration":0.7}}]}}

User says: "look at the red cup"
{{"say":"I see a red cup on the right. Turning toward it.","actions":[{{"type":"set_pose","pose":{{"joint1":-35,"joint2":20,"joint3":-25}},"duration":2.0}},{{"type":"wait","duration":0.5}},{{"type":"home","duration":2.0}}]}}

User says: "pick up the cube in front of you"
{{"say":"I see a small cube directly ahead. Reaching for it.","actions":[{{"type":"set_gripper","opening":100,"duration":0.6}},{{"type":"set_pose","pose":{{"joint2":40,"joint3":-50}},"duration":2.0}},{{"type":"set_gripper","opening":0,"duration":0.8}},{{"type":"home","duration":2.0}}]}}

User says: "do you see a banana?"
{{"say":"No, I don't see a banana — I see a red cup, a laptop, and a notebook.","actions":[]}}
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
