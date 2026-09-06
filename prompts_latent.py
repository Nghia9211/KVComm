"""
prompts_latent.py — Role-based prompt templates for KVComm + LatentMAS

Intentionally SEPARATE from eval.py (original KVComm prompts).
No original KVComm template is modified here.

Prompt design based on LatentMAS paper (arXiv:2511.20639, Appendix K):
  - Sequential MAS: Planner → Critic → Refiner → Solver
  - Our 2-agent simplification: Planner (A) → Solver (B)

Two agent roles:
  Sender A   — "Planner Agent": design a step-by-step plan to solve the question.
               Does NOT produce the final answer.
  Receiver B — "Solver Agent": receives latent/text reasoning from A, produces
               the final answer.

KVComm original tasks (HotpotQA, TMath, RepoBench, SAMSum) keep their existing
asymmetric prompt design (prompt_A ≠ prompt_B) since they naturally split
context vs. question.
"""
from __future__ import annotations


# ══════════════════════════════════════════════════════════════════════════════
# Sender A: Planner Agent Instructions
# ══════════════════════════════════════════════════════════════════════════════
# From LatentMAS paper (Appendix K, Section 20):
#   "You are a Planner Agent. Given an input question, design a clear,
#    step-by-step plan for how to solve the question."
#
# The paper uses the SAME Planner prompt for ALL task types (numeric, MCQ, code).
# We keep this unified design for LatentMAS tasks and preserve KVComm-specific
# prompts for KVComm original tasks.

# ── Think-model variant (Qwen3, DeepSeek-R1, etc.) ──────────────────────────
# Triggers <think> mode for latent thought generation.
PLANNER_INSTRUCTION = (
    "You are a Planner Agent. Given an input question, design a clear, "
    "step-by-step plan for how to solve the question.\n\n"
    "Your outlined plan should be concise with a few bulletpoints for each step. "
    "Do not produce the final answer.\n"
    "Now output your plan to solve the question below:"
)

# ── Non-think variant (Llama-3, Mistral, etc.) ──────────────────────────────
# These models don't have <think> mode, so we use the same Planner framing
# but with explicit "reader" language to guide KV cache encoding.
PLANNER_INSTRUCTION_NOTHINK = (
    "You are a Planner Agent. Given an input question, read it carefully "
    "and design a clear analysis plan.\n\n"
    "Your outlined plan should be concise with a few bulletpoints for each step. "
    "Do not produce the final answer.\n"
    "Now analyze the question below:"
)


# ── KVComm original tasks: keep asymmetric sender instructions ──────────────
# These tasks have prompt_A ≠ prompt_B (context vs. question split).
# Agent A acts as "context encoder", not "planner".

_KVCOMM_SENDER_MATH_THINK = (
    "Think deeply about the following mathematical hint. "
    "Your internal reasoning will be used by another agent to solve a related problem."
)
_KVCOMM_SENDER_QA_THINK = (
    "Think deeply about the following context passage. "
    "Your internal reasoning will be used by another agent to answer a question about it."
)
_KVCOMM_SENDER_CODE_THINK = (
    "Think deeply about the following code context. "
    "Your internal reasoning will be used by another agent to complete the code."
)
_KVCOMM_SENDER_SUMMARIZE_THINK = (
    "Think deeply about the following content. "
    "Your internal reasoning will be used by another agent to summarize related content."
)

_KVCOMM_SENDER_MATH_NOTHINK = (
    "You are a reader agent. Read the following mathematical hint carefully "
    "and reason about the key facts, numbers, and relationships it contains."
)
_KVCOMM_SENDER_QA_NOTHINK = (
    "You are a reader agent. Read the following context passage carefully "
    "and reason about the key information it contains."
)
_KVCOMM_SENDER_CODE_NOTHINK = (
    "You are a reader agent. Read the following code context carefully "
    "and reason about the key functions, variables, and logic it contains."
)
_KVCOMM_SENDER_SUMMARIZE_NOTHINK = (
    "You are a reader agent. Read the following content carefully "
    "and reason about the main ideas and key information it contains."
)


# ── Sender A: message templates ──────────────────────────────────────────────

_SENDER_MATH_TMPL = "Instruction: {instruction} Hint: {hint}"
_SENDER_QA_TMPL = "Instruction: {instruction} Context: {context}"
_SENDER_CODE_TMPL = "Instruction: {instruction} Context: {context}"
_SENDER_SUMMARIZE_TMPL = "Instruction: {instruction} Content part 1: {content_part_1}"


def build_latent_sender_msg(evaluator, item: dict, is_think: bool = False) -> str:
    """
    Build the user-role message string for sender A.

    LatentMAS tasks: uses unified Planner prompt from LatentMAS paper.
    KVComm tasks:    uses task-specific context-encoder prompts.

    Args:
        evaluator: Task evaluator (used to detect task type via hasattr flags).
        item:      Dataset item dict with "prompt_A" key.
        is_think:  Whether model_A is a thinking model.

    Returns:
        str: Formatted message string for model_A.
    """
    # ── KVComm original tasks (asymmetric: prompt_A ≠ prompt_B) ──────────────
    if hasattr(evaluator, "tmath"):
        inst = _KVCOMM_SENDER_MATH_THINK if is_think else _KVCOMM_SENDER_MATH_NOTHINK
        return _SENDER_MATH_TMPL.format(instruction=inst, hint=item["prompt_A"])

    elif hasattr(evaluator, "repobench"):
        inst = _KVCOMM_SENDER_CODE_THINK if is_think else _KVCOMM_SENDER_CODE_NOTHINK
        return _SENDER_CODE_TMPL.format(instruction=inst, context=item["prompt_A"])

    elif hasattr(evaluator, "sasum"):
        inst = _KVCOMM_SENDER_SUMMARIZE_THINK if is_think else _KVCOMM_SENDER_SUMMARIZE_NOTHINK
        return _SENDER_SUMMARIZE_TMPL.format(instruction=inst, content_part_1=item["prompt_A"])

    # ── LatentMAS tasks (symmetric: prompt_A ≈ prompt_B) ─────────────────────
    # All LatentMAS tasks use the unified Planner prompt from the paper.
    # Agent A = Planner: "design a step-by-step plan, do not produce the final answer"
    else:
        planner_inst = PLANNER_INSTRUCTION if is_think else PLANNER_INSTRUCTION_NOTHINK
        return f"{planner_inst}\nQuestion: {item['prompt_A']}"


# ══════════════════════════════════════════════════════════════════════════════
# Receiver B: Solver Agent
# ══════════════════════════════════════════════════════════════════════════════
# From LatentMAS paper (Appendix K):
#   "You are a helpful assistant. You are provided with latent information
#    for reference and a target question to solve."
#   "The latent information might contain irrelevant contents. Ignore it if
#    it is not helpful for solving the target question."

# ── Latent receiver prefix (for LatentMAS — B receives KV cache) ─────────────
LATENT_RECEIVER_PREFIX = (
    "You are a helpful assistant. You are provided with latent information "
    "for reference and a target question to solve.\n"
    "The latent information might contain irrelevant contents. "
    "Ignore it if it is not helpful for solving the target question.\n"
)

# ── Text receiver prefix (for TextMAS — B receives Agent A's text reasoning) ─
TEXT_RECEIVER_PREFIX = (
    "You are a helpful assistant. You are provided with reasoning from another "
    "agent for reference and a target question to solve.\n"
    "The reasoning might not be fully relevant. "
    "Ignore it if it is not helpful for solving the target question.\n"
)


def _get_receiver_core_msg(evaluator, item: dict, allow_b_think: bool = False) -> str:
    """
    Build the core task instruction and question for Receiver B (Solver).

    Uses Solver prompt templates from LatentMAS paper (Appendix K).
    Three task groups: numeric, MCQ, code.

    For KVComm original tasks, falls back to eval.py templates.
    """
    from eval import (
        COMMUNICATION_MATH_MSG_TEMPLATE_B,
        COMMUNICATION_QA_MSG_TEMPLATE_B,
        COMMUNICATION_CODE_MSG_TEMPLATE_B,
        COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B,
        MATH_INSTRUCTION,
        QA_INSTRUCTION,
        CODE_INSTRUCTION,
        SUMMARIZE_INSTRUCTION,
    )

    # ── KVComm original tasks (keep existing B prompts) ──────────────────────
    if hasattr(evaluator, "tmath"):
        instruction = MATH_INSTRUCTION if allow_b_think else "Directly answer the math problem with the final result."
        return COMMUNICATION_MATH_MSG_TEMPLATE_B.format(
            instruction=instruction,
            question=item["prompt_B"],
        )
    elif hasattr(evaluator, "repobench"):
        return COMMUNICATION_CODE_MSG_TEMPLATE_B.format(
            instruction=CODE_INSTRUCTION,
            code_snippet=item["prompt_B"],
        )
    elif hasattr(evaluator, "sasum"):
        return COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B.format(
            instruction=SUMMARIZE_INSTRUCTION,
            content_part_2=item["prompt_B"],
        )

    # ── LatentMAS tasks: Solver prompts from paper (Appendix K) ──────────────

    # --- Numeric tasks (GSM8K, AIME) ---
    # Paper Solver: "Now, reason step by step and output the final answer
    #                inside \boxed{YOUR_FINAL_ANSWER}:"
    elif hasattr(evaluator, "gsm8k") or hasattr(evaluator, "aime"):
        if allow_b_think:
            return (
                f"Target Question: {item['prompt_B']}\n\n"
                f"Now, reason step by step and output the final answer inside "
                f"\\boxed{{YOUR_FINAL_ANSWER}}:"
            )
        else:
            return (
                f"Target Question: {item['prompt_B']}\n\n"
                f"Output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}:"
            )

    # --- MCQ tasks (MedQA, ARC-E/C, GPQA) ---
    # Paper Solver: "Your final answer must be selected from A,B,C,D.
    #                For example \boxed{A}. Do not add any other contents inside the box."
    elif (hasattr(evaluator, "medqa") or hasattr(evaluator, "arc_easy")
          or hasattr(evaluator, "arc_challenge") or hasattr(evaluator, "gpqa")):
        if allow_b_think:
            return (
                f"Target Question: {item['prompt_B']}\n\n"
                f"Your final answer must be selected from A, B, C, D. "
                f"For example \\boxed{{A}}. Do not add any other contents inside the box.\n"
                f"Now, reason step by step and output the final answer inside "
                f"\\boxed{{YOUR_FINAL_ANSWER}}:"
            )
        else:
            return (
                f"Target Question: {item['prompt_B']}\n\n"
                f"Your final answer must be selected from A, B, C, D. "
                f"For example \\boxed{{A}}. Do not add any other contents inside the box."
            )

    # --- Code tasks (MBPP+, HumanEval+) ---
    # Paper Solver: "You must put all python code as self-contained Python function
    #                in markdown code blocks."
    elif hasattr(evaluator, "mbppplus") or hasattr(evaluator, "humanevalplus"):
        if allow_b_think:
            return (
                f"Target Question: {item['prompt_B']}\n\n"
                f"You must put all python code as self-contained Python function "
                f"in markdown code blocks.\n"
                f"Do not add any other contents inside the markdown code block.\n"
                f"Now, reason step by step and output the final answer inside "
                f"```python\nYOUR_PYTHON_CODE\n```:"
            )
        else:
            return (
                f"Target Question: {item['prompt_B']}\n\n"
                f"Put all your Python code inside a markdown code block:\n"
                f"```python\nYOUR_CODE_HERE\n```\n"
                f"Do not add any other contents inside the code block."
            )

    else:
        # Default: general QA (KVComm fallback)
        return COMMUNICATION_QA_MSG_TEMPLATE_B.format(
            instruction=QA_INSTRUCTION,
            question=item["prompt_B"],
        )


def build_latent_receiver_msg(evaluator, item: dict, allow_b_think: bool = False) -> str:
    """
    Build the user-role message string for receiver B in LatentMAS.
    Uses paper Solver prefix: "You are provided with latent information for reference..."
    """
    return LATENT_RECEIVER_PREFIX + _get_receiver_core_msg(evaluator, item, allow_b_think=allow_b_think)


def build_text_receiver_msg(evaluator, item: dict, response_A: str = "", allow_b_think: bool = False) -> str:
    """
    Build the user-role message string for receiver B in TextMAS.
    Uses clean text prefix: "You are provided with reasoning from another agent..."
    followed by Agent A's reasoning, then the target question and task instructions.
    """
    core_msg = _get_receiver_core_msg(evaluator, item, allow_b_think=allow_b_think)
    if response_A:
        return (
            f"{TEXT_RECEIVER_PREFIX}\n"
            f"Agent A's reasoning:\n{response_A}\n\n"
            f"{core_msg}"
        )
    return TEXT_RECEIVER_PREFIX + core_msg
