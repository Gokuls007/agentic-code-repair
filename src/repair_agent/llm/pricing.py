"""Cost estimation from token usage."""

from __future__ import annotations

from repair_agent.config import ModelPrice
from repair_agent.llm.base import Usage

# Anthropic bills prompt-cache reads at 0.1x and 5-minute cache writes at 1.25x the
# base input price. Providers without prompt caching report zero for both.
CACHE_READ_MULTIPLIER = 0.1
CACHE_WRITE_MULTIPLIER = 1.25


def estimate_cost(usage: Usage, model: str, pricing: dict[str, ModelPrice]) -> float | None:
    """Estimated USD cost of ``usage`` on ``model``, or None if the model has no price entry.

    Returning None (rather than 0.0) keeps unknown costs distinguishable from free ones
    in reports.
    """
    price = pricing.get(model)
    if price is None:
        return None
    per_token_in = price.input_per_mtok / 1_000_000
    per_token_out = price.output_per_mtok / 1_000_000
    return (
        usage.input_tokens * per_token_in
        + usage.cache_read_input_tokens * per_token_in * CACHE_READ_MULTIPLIER
        + usage.cache_creation_input_tokens * per_token_in * CACHE_WRITE_MULTIPLIER
        + usage.output_tokens * per_token_out
    )
