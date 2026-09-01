"""
prompts_latent.py — Latent-aware prompt templates for KVComm + LatentMAS

Intentionally SEPARATE from eval.py (original KVComm prompts).
No original KVComm template is modified here.

Two roles:
  Sender A  — "Latent Thinker": think deeply, your hidden states will be sent to B.
  Receiver B — latent-aware: you receive latent encoded context from another agent.

Two sender instruction sets:
  LATENT_SENDER_*_INSTRUCTION        — for think-models (Qwen3, DeepSeek-R1, etc.)
                                        triggers <think> mode in A
  LATENT_SENDER_*_INSTRUCTION_NOTHINK — for non-think models (Llama-3, etc.)
                                        uses explicit encode/summarize framing
"""
from __future__ import annotations


# ── Sender A: Latent Thinker instructions (think-models) ───────────────────────
# A is told its internal representations will be communicated to B.
# This primes A to "think for transfer", not just answer for itself.

LATENT_SENDER_MATH_INSTRUCTION = (
    "Think deeply about the following mathematical hint. "
    "Your internal reasoning will be used by another agent to solve a related problem."
)
LATENT_SENDER_QA_INSTRUCTION = (
    "Think deeply about the following context passage. "
    "Your internal reasoning will be used by another agent to answer a question about it."
)
LATENT_SENDER_CODE_INSTRUCTION = (
    "Think deeply about the following code context. "
    "Your internal reasoning will be used by another agent to complete the code."
)
LATENT_SENDER_SUMMARIZE_INSTRUCTION = (
    "Think deeply about the following content. "
    "Your internal reasoning will be used by another agent to summarize related content."
)
# LatentMAS tasks
LATENT_SENDER_MEDQA_INSTRUCTION = (
    "Think deeply about the following medical question. "
    "Analyze the clinical scenario thoroughly and consider each answer option. "
    "Your internal reasoning will be used by another agent to select the correct answer."
)
LATENT_SENDER_CODE_GEN_INSTRUCTION = (
    "Think deeply about the following programming problem. "
    "Reason about the algorithm, edge cases, and data structures needed. "
    "Your internal reasoning will be used by another agent to generate the solution code."
)
LATENT_SENDER_AIME_INSTRUCTION = (
    "Think deeply about the following competition mathematics problem. "
    "Work through all algebraic, geometric, or combinatorial reasoning carefully. "
    "Your internal reasoning will be used by another agent to compute the final answer."
)
LATENT_SENDER_GSM8K_INSTRUCTION = (
    "Think deeply about the following math word problem. "
    "Work through the arithmetic and logical reasoning step by step. "
    "Your internal reasoning will be used by another agent to compute the final answer."
)
LATENT_SENDER_MCQ_INSTRUCTION = (
    "Think deeply about the following question and each answer option. "
    "Analyze the key information and relationships between options carefully. "
    "Your internal reasoning will be used by another agent to select the correct answer."
)


# ── Sender A: Non-think instructions (Llama-3 and other non-thinking models) ───
# These models don't have a dedicated thinking mode, so we ask them to explicitly
# encode key information rather than "think deeply" (which has no effect).

# NOTHINK variants (Llama-3 và các non-think models)
# Dùng goal-directed style thay vì encoding-directive style.
# Bỏ câu "Your internal representation..." vì nó khiến B echo lại thành "I have internalized..."
LATENT_SENDER_MATH_INSTRUCTION_NOTHINK = (
    "You are a reader agent. Read the following mathematical hint carefully "
    "and reason about the key facts, numbers, and relationships it contains."
)
LATENT_SENDER_QA_INSTRUCTION_NOTHINK = (
    "You are a reader agent. Read the following context passage carefully "
    "and reason about the key information it contains."
)
LATENT_SENDER_CODE_INSTRUCTION_NOTHINK = (
    "You are a reader agent. Read the following code context carefully "
    "and reason about the key functions, variables, and logic it contains."
)
LATENT_SENDER_SUMMARIZE_INSTRUCTION_NOTHINK = (
    "You are a reader agent. Read the following content carefully "
    "and reason about the main ideas and key information it contains."
)
# LatentMAS tasks (non-think variants)
LATENT_SENDER_MEDQA_INSTRUCTION_NOTHINK = (
    "You are a medical reader agent. Read the following medical question and answer choices carefully "
    "and reason about the clinical scenario, key facts, and relationships between options."
)
LATENT_SENDER_CODE_GEN_INSTRUCTION_NOTHINK = (
    "You are a reader agent. Read the following programming problem and test cases carefully "
    "and reason about the required function behavior, input/output constraints, and algorithm."
)
LATENT_SENDER_AIME_INSTRUCTION_NOTHINK = (
    "You are a reader agent. Read the following competition mathematics problem carefully "
    "and reason about the numerical constraints, relationships, and mathematical concepts."
)
LATENT_SENDER_GSM8K_INSTRUCTION_NOTHINK = (
    "You are a reader agent. Read the following math word problem carefully "
    "and reason about the numbers, operations, and logical steps needed to solve it."
)
LATENT_SENDER_MCQ_INSTRUCTION_NOTHINK = (
    "You are a reader agent. Read the following question and answer choices carefully "
    "and reason about the key information and relationships between options."
)


# ── Receiver B: Latent-awareness prefix ───────────────────────────────────────
# Prepended to B's regular instruction so B knows latent context exists.
# Mirrors the LatentMAS judger prompt: "latent information might contain irrelevant content."

LATENT_RECEIVER_PREFIX = (
    "You are provided with latent encoded context from another agent. "
    "The latent information might contain irrelevant content — "
    "ignore it if not helpful for the task. "
)


# ── Sender A: message templates ───────────────────────────────────────────────

_SENDER_MATH_TMPL = "Instruction: {instruction} Hint: {hint}"
_SENDER_QA_TMPL = "Instruction: {instruction} Context: {context}"
_SENDER_CODE_TMPL = "Instruction: {instruction} Context: {context}"
_SENDER_SUMMARIZE_TMPL = "Instruction: {instruction} Content part 1: {content_part_1}"


def build_latent_sender_msg(evaluator, item: dict, is_think: bool = False) -> str:
    """
    Build the user-role message string for sender A as a Latent Thinker.

    Automatically selects the right instruction set:
      - is_think=True  (think-models: Qwen3, DeepSeek-R1, ...): uses LATENT_SENDER_*_INSTRUCTION
                        which primes model to enter <think> mode.
      - is_think=False (non-think models: Llama-3, Mistral, ...): uses NOTHINK variant
                        which explicitly asks model to encode key information.

    Supported evaluator task flags:
      evaluator.tmath       → math hint (KVComm original)
      evaluator.repobench   → code context (KVComm original)
      evaluator.sasum       → summarization (KVComm original)
      evaluator.medqa       → medical MCQ (LatentMAS task)
      evaluator.mbppplus    → code generation (LatentMAS task)
      evaluator.aime        → competition math (LatentMAS task)
      evaluator.gsm8k       → math word problem (LatentMAS task)
      evaluator.arc_easy    → MCQ science easy (LatentMAS task)
      evaluator.arc_challenge → MCQ science hard (LatentMAS task)
      evaluator.gpqa        → MCQ graduate science (LatentMAS task)
      evaluator.humanevalplus → code generation HumanEval+ (LatentMAS task)
      (default)             → QA / general context (KVComm original)

    Args:
        evaluator: Task evaluator (used to detect task type via hasattr flags).
        item:      Dataset item dict with "prompt_A" key.
        is_think:  Whether model_A is a thinking model.

    Returns:
        str: Formatted message string for model_A.
    """
    if is_think:
        math_inst      = LATENT_SENDER_MATH_INSTRUCTION
        qa_inst        = LATENT_SENDER_QA_INSTRUCTION
        code_inst      = LATENT_SENDER_CODE_INSTRUCTION
        sum_inst       = LATENT_SENDER_SUMMARIZE_INSTRUCTION
        medqa_inst     = LATENT_SENDER_MEDQA_INSTRUCTION
        code_gen_inst  = LATENT_SENDER_CODE_GEN_INSTRUCTION
        aime_inst      = LATENT_SENDER_AIME_INSTRUCTION
        gsm8k_inst     = LATENT_SENDER_GSM8K_INSTRUCTION
        mcq_inst       = LATENT_SENDER_MCQ_INSTRUCTION
    else:
        math_inst      = LATENT_SENDER_MATH_INSTRUCTION_NOTHINK
        qa_inst        = LATENT_SENDER_QA_INSTRUCTION_NOTHINK
        code_inst      = LATENT_SENDER_CODE_INSTRUCTION_NOTHINK
        sum_inst       = LATENT_SENDER_SUMMARIZE_INSTRUCTION_NOTHINK
        medqa_inst     = LATENT_SENDER_MEDQA_INSTRUCTION_NOTHINK
        code_gen_inst  = LATENT_SENDER_CODE_GEN_INSTRUCTION_NOTHINK
        aime_inst      = LATENT_SENDER_AIME_INSTRUCTION_NOTHINK
        gsm8k_inst     = LATENT_SENDER_GSM8K_INSTRUCTION_NOTHINK
        mcq_inst       = LATENT_SENDER_MCQ_INSTRUCTION_NOTHINK

    # ── KVComm original tasks ──────────────────────────────────────────────────
    if hasattr(evaluator, "tmath"):
        return _SENDER_MATH_TMPL.format(
            instruction=math_inst,
            hint=item["prompt_A"],
        )
    elif hasattr(evaluator, "repobench"):
        return _SENDER_CODE_TMPL.format(
            instruction=code_inst,
            context=item["prompt_A"],
        )
    elif hasattr(evaluator, "sasum"):
        return _SENDER_SUMMARIZE_TMPL.format(
            instruction=sum_inst,
            content_part_1=item["prompt_A"],
        )
    # ── LatentMAS tasks ───────────────────────────────────────────────────
    elif hasattr(evaluator, "medqa"):
        # A processes the medical question (same content as B, to build KV cache)
        return f"Instruction: {medqa_inst} Question: {item['prompt_A']}"
    elif hasattr(evaluator, "mbppplus"):
        # A thinks about the programming problem
        return f"Instruction: {code_gen_inst} Problem: {item['prompt_A']}"
    elif hasattr(evaluator, "aime"):
        # A thinks about the competition problem
        return f"Instruction: {aime_inst} Problem: {item['prompt_A']}"
    elif hasattr(evaluator, "gsm8k"):
        # A thinks about the math word problem
        return f"Instruction: {gsm8k_inst} Problem: {item['prompt_A']}"
    elif hasattr(evaluator, "arc_easy") or hasattr(evaluator, "arc_challenge") or hasattr(evaluator, "gpqa"):
        # A thinks about the MCQ question and options
        return f"Instruction: {mcq_inst} Question: {item['prompt_A']}"
    elif hasattr(evaluator, "humanevalplus"):
        # A thinks about the HumanEval+ programming problem (reuse code_gen_inst)
        return f"Instruction: {code_gen_inst} Problem: {item['prompt_A']}"
    else:
        # Default: QA / general context
        return _SENDER_QA_TMPL.format(
            instruction=qa_inst,
            context=item["prompt_A"],
        )


def build_latent_receiver_msg(evaluator, item: dict, allow_b_think: bool = False) -> str:
    """
    Build the user-role message string for receiver B with latent-awareness prefix.

    Imports and reuses core B templates from eval.py to stay in sync with
    the original KVComm receiver format (only the prefix changes).

    For LatentMAS tasks (medqa, mbppplus, aime2024, etc.) that don’t have a split
    prompt_A / prompt_B, B receives the full question (prompt_B) with an
    instruction dynamically adapted based on allow_b_think:
      - allow_b_think=True: Instructs B to reason step-by-step using the latent context.
      - allow_b_think=False: Instructs B to directly output the answer without step-by-step
                             reasoning to avoid conflict with suppressed CoT / assistant prefixes.

    Args:
        evaluator: Task evaluator (used to detect task type via hasattr flags).
        item:      Dataset item dict with "prompt_B" key.
        allow_b_think: Whether receiver B is allowed to generate thinking tokens.

    Returns:
        str: LATENT_RECEIVER_PREFIX + task-specific receiver message.
    """
    # Import only from eval.py, never modify it
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

    # ── KVComm original tasks ──────────────────────────────────────────────────
    if hasattr(evaluator, "tmath"):
        instruction = MATH_INSTRUCTION if allow_b_think else "Directly answer the math problem with the final result."
        core = COMMUNICATION_MATH_MSG_TEMPLATE_B.format(
            instruction=instruction,
            question=item["prompt_B"],
        )
    elif hasattr(evaluator, "repobench"):
        core = COMMUNICATION_CODE_MSG_TEMPLATE_B.format(
            instruction=CODE_INSTRUCTION,
            code_snippet=item["prompt_B"],
        )
    elif hasattr(evaluator, "sasum"):
        core = COMMUNICATION_SUMMARIZE_MSG_TEMPLATE_B.format(
            instruction=SUMMARIZE_INSTRUCTION,
            content_part_2=item["prompt_B"],
        )
    # ── LatentMAS tasks ───────────────────────────────────────────────────
    elif hasattr(evaluator, "medqa"):
        # B answers a medical MCQ: must output \boxed{A/B/C/D}
        if allow_b_think:
            inst = "Use it to reason step-by-step and select the correct answer."
        else:
            inst = "Based on the provided context, directly select the correct answer without any step-by-step reasoning."
        core = (
            f"You are a medical question answering assistant. "
            f"You are provided with latent context from another reasoning agent. "
            f"{inst}\n\n"
            f"{item['prompt_B']}\n\n"
            f"Your final answer must be selected from A, B, C, D. "
            f"For example \\boxed{{A}}. Do not add any other content inside the box."
        )
        return LATENT_RECEIVER_PREFIX + core
    elif hasattr(evaluator, "mbppplus"):
        # B generates python code: must output ```python ... ```
        if allow_b_think:
            inst = "Use it to write a correct, self-contained solution."
        else:
            inst = "Directly provide the Python code solution without any conversational text or step-by-step explanation."
        core = (
            f"You are a Python programming assistant. "
            f"You are provided with latent context from another reasoning agent. "
            f"{inst}\n\n"
            f"{item['prompt_B']}\n\n"
            f"Put all your Python code inside a markdown code block:\n"
            f"```python\nYOUR_CODE_HERE\n```\n"
            f"Do not add any other content inside the code block."
        )
        return LATENT_RECEIVER_PREFIX + core
    elif hasattr(evaluator, "aime"):
        # B solves competition math: must output \boxed{N} where N is an integer
        if allow_b_think:
            core = (
                f"You are a competition mathematics solver. "
                f"You are provided with latent reasoning context from another agent. "
                f"Use it to reason step-by-step and find the exact integer answer.\n\n"
                f"{item['prompt_B']}\n\n"
                f"Reason step by step and output the final integer answer inside "
                f"\\boxed{{YOUR_FINAL_ANSWER}}.\n"
                f"Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."
            )
        else:
            core = (
                f"You are a competition mathematics solver. "
                f"You are provided with latent reasoning context from another agent. "
                f"Based on the provided context, directly output the final integer answer without any step-by-step reasoning.\n\n"
                f"{item['prompt_B']}\n\n"
                f"Output only the final integer answer inside \\boxed{{YOUR_FINAL_ANSWER}}."
            )
        return LATENT_RECEIVER_PREFIX + core
    elif hasattr(evaluator, "gsm8k"):
        # B solves a math word problem: must output \boxed{N}
        if allow_b_think:
            core = (
                f"You are a math problem solver. "
                f"You are provided with latent reasoning context from another agent. "
                f"Use it to reason step-by-step and find the correct numerical answer.\n\n"
                f"{item['prompt_B']}\n\n"
                f"Your final answer must be a number. "
                f"Reason step by step and output the final answer inside "
                f"\\boxed{{YOUR_FINAL_ANSWER}}.\n"
                f"Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."
            )
        else:
            core = (
                f"You are a math problem solver. "
                f"You are provided with latent reasoning context from another agent. "
                f"Based on the provided context, directly output the final numerical answer without any step-by-step reasoning.\n\n"
                f"{item['prompt_B']}\n\n"
                f"Your final answer must be a number. Output only the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."
            )
        return LATENT_RECEIVER_PREFIX + core
    elif hasattr(evaluator, "arc_easy") or hasattr(evaluator, "arc_challenge") or hasattr(evaluator, "gpqa"):
        # B answers a science/graduate MCQ: must output \boxed{A/B/C/D}
        if allow_b_think:
            core = (
                f"You are a question answering assistant. "
                f"You are provided with latent context from another reasoning agent. "
                f"Use it to reason step-by-step and select the correct answer.\n\n"
                f"{item['prompt_B']}\n\n"
                f"Your final answer must be selected from A, B, C, D. "
                f"For example \\boxed{{A}}. Do not add any other content inside the box.\n"
                f"Now, reason step by step and output the final answer inside \\boxed{{YOUR_FINAL_ANSWER}}."
            )
        else:
            core = (
                f"You are a question answering assistant. "
                f"You are provided with latent context from another reasoning agent. "
                f"Based on the provided context, directly select the correct answer without any step-by-step reasoning.\n\n"
                f"{item['prompt_B']}\n\n"
                f"Your final answer must be selected from A, B, C, D. "
                f"For example \\boxed{{A}}. Do not add any other content inside the box."
            )
        return LATENT_RECEIVER_PREFIX + core
    elif hasattr(evaluator, "humanevalplus"):
        # B generates Python code: must output ```python ... ```
        if allow_b_think:
            core = (
                f"You are a Python programming assistant. "
                f"You are provided with latent context from another reasoning agent. "
                f"Use it to write a correct, self-contained solution.\n\n"
                f"{item['prompt_B']}\n\n"
                f"Put all your Python code inside a markdown code block:\n"
                f"```python\nYOUR_CODE_HERE\n```\n"
                f"Do not add any other content inside the code block.\n"
                f"Now, reason step by step and output the final answer inside ```python\nYOUR_PYTHON_CODE\n```."
            )
        else:
            core = (
                f"You are a Python programming assistant. "
                f"You are provided with latent context from another reasoning agent. "
                f"Directly provide the Python code solution without any conversational text or step-by-step explanation.\n\n"
                f"{item['prompt_B']}\n\n"
                f"Put all your Python code inside a markdown code block:\n"
                f"```python\nYOUR_CODE_HERE\n```\n"
                f"Do not add any other content inside the code block."
            )
        return LATENT_RECEIVER_PREFIX + core
    else:
        # Default: general QA
        core = COMMUNICATION_QA_MSG_TEMPLATE_B.format(
            instruction=QA_INSTRUCTION,
            question=item["prompt_B"],
        )

    return LATENT_RECEIVER_PREFIX + core
