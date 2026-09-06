"""Small dependency-free helpers for evaluation configuration."""


def resolve_textmas_budgets(evaluator, max_tokens_a: int = 0, max_tokens_b: int = 0) -> tuple[int, int]:
    budget_a = max_tokens_a if max_tokens_a > 0 else evaluator.sender_max_tokens
    budget_b = max_tokens_b if max_tokens_b > 0 else evaluator.max_tokens
    if budget_a <= 0 or budget_b <= 0:
        raise ValueError("Resolved TextMAS token budgets must both be positive")
    return budget_a, budget_b
