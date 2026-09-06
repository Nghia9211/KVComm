"""Deterministic, modality-independent prompt routing for two-agent evaluation."""

from __future__ import annotations

PLANNER_INSTRUCTION = (
    "You are a Planner Agent. Given an input question, design a clear, step-by-step plan.\n\n"
    "Keep the plan concise. Do not produce the final answer.\n"
    "Now output your plan to solve the question below:"
)
PLANNER_INSTRUCTION_NOTHINK = (
    "You are a Planner Agent. Read the input question carefully and design a clear analysis plan.\n\n"
    "Keep the plan concise. Do not produce the final answer.\n"
    "Now analyze the question below:"
)
LATENT_RECEIVER_PREFIX = (
    "Information from the previous agent is available through transferred internal state. "
    "Use it when relevant.\n\n"
)
TEXT_RECEIVER_PREFIX = "Information communicated by the previous agent:\n"


def _validate(evaluator) -> None:
    if not hasattr(evaluator, "validate_task_profile"):
        raise ValueError("Evaluator does not expose the required task profile")
    evaluator.validate_task_profile()


def build_sender_core(evaluator, item: dict, is_think: bool = False) -> str:
    """Build Agent A's semantic prompt; shared by text/full-KV/selective-KV."""
    _validate(evaluator)
    family = evaluator.prompt_family
    input_mode = evaluator.sender_input_mode
    task_type = evaluator.task_type

    if family == "kvcomm" and input_mode == "query_aware_context":
        return (
            "You are an Evidence Extraction Agent.\n\n"
            "Read the context and target question carefully. Identify the exact evidence\n"
            "needed by another agent to answer the target question.\n\n"
            "Preserve entity names, aliases, dates, quantities, locations, definitions,\n"
            "and relationships exactly as stated in the context. Do not invent information.\n"
            "Do not produce the final answer.\n\n"
            f"Context:\n{item['prompt_A']}\n\n"
            f"Target Question:\n{item['prompt_B']}\n\n"
            "Prepare concise evidence for the next agent:"
        )

    if family == "kvcomm" and input_mode == "native_split":
        instructions = {
            "math": "Read the mathematical hint carefully and communicate its key facts, numbers, and relationships.",
            "code": "Read the code context carefully and communicate the key functions, variables, and logic.",
            "summarization": "Read content part 1 carefully and communicate its main ideas and key information.",
        }
        labels = {"math": "Hint", "code": "Context", "summarization": "Content part 1"}
        if task_type not in instructions:
            raise ValueError(f"Unsupported native-split task_type={task_type!r}")
        return f"Instruction: {instructions[task_type]} {labels[task_type]}: {item['prompt_A']}"

    if family == "latentmas" and input_mode == "shared_problem":
        instruction = PLANNER_INSTRUCTION if is_think else PLANNER_INSTRUCTION_NOTHINK
        return f"{instruction}\nQuestion: {item['prompt_A']}"

    raise ValueError(
        f"Unsupported sender profile: {family}/{task_type}/{input_mode} "
        f"for {getattr(evaluator, 'name', type(evaluator).__name__)}"
    )


def build_receiver_core(evaluator, item: dict, allow_b_think: bool = False) -> str:
    """Build Agent B's task prompt without a communication-modality wrapper."""
    _validate(evaluator)
    family = evaluator.prompt_family
    input_mode = evaluator.sender_input_mode
    task_type = evaluator.task_type
    answer_format = evaluator.answer_format

    if family == "kvcomm" and input_mode == "query_aware_context":
        return (
            "You are the final Answering Agent.\n\n"
            "Use the information provided by the previous agent to answer the target\n"
            "question. Do not invent unsupported information.\n\n"
            f"Target Question:\n{item['prompt_B']}\n\n"
            "Return only the shortest answer that fully answers the question.\n"
            "Do not provide explanations."
        )

    if family == "kvcomm" and input_mode == "native_split":
        if task_type == "math":
            instruction = "Answer the math problem step by step." if allow_b_think else "Directly answer the math problem with the final result."
            return f"Instruction: {instruction} Question: {item['prompt_B']}"
        if task_type == "code":
            return f"Instruction: Complete ONLY THE NEXT LINE of the code snippet based on the context. Code Snippet: {item['prompt_B']}"
        if task_type == "summarization":
            return f"Instruction: Summarize the following content concisely with one sentence. Content part 2: {item['prompt_B']}"

    if family == "latentmas" and input_mode == "shared_problem":
        target = f"Target Question: {item['prompt_B']}\n\n"
        if answer_format == "boxed_integer":
            reasoning = "Now, reason step by step and " if allow_b_think else ""
            return target + reasoning + "output the final answer inside \\boxed{YOUR_FINAL_ANSWER}:"
        if answer_format == "boxed_choice":
            reasoning = "Reason step by step. " if allow_b_think else ""
            return target + reasoning + "Return only option A, B, C, or D inside \\boxed{YOUR_FINAL_ANSWER}."
        if answer_format == "python":
            reasoning = "Reason step by step, then " if allow_b_think else ""
            return target + reasoning + "put all self-contained Python code in one ```python``` markdown code block."

    raise ValueError(
        f"Unsupported receiver profile: {family}/{task_type}/{input_mode}/{answer_format} "
        f"for {getattr(evaluator, 'name', type(evaluator).__name__)}"
    )


def build_latent_sender_msg(evaluator, item: dict, is_think: bool = False) -> str:
    return build_sender_core(evaluator, item, is_think=is_think)


def build_latent_receiver_msg(evaluator, item: dict, allow_b_think: bool = False) -> str:
    return LATENT_RECEIVER_PREFIX + build_receiver_core(evaluator, item, allow_b_think)


def build_text_receiver_msg(evaluator, item: dict, response_A: str = "", allow_b_think: bool = False) -> str:
    core = build_receiver_core(evaluator, item, allow_b_think)
    if not response_A:
        return core
    return (
        f"{TEXT_RECEIVER_PREFIX}<previous_agent_information>\n{response_A}\n"
        f"</previous_agent_information>\n\n{core}"
    )
