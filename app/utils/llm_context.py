"""Token-budget helpers for LLM prompt assembly."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

from app.utils.text import normalize_text
from app.utils.tokens import count_tokens


@dataclass(frozen=True, slots=True)
class BudgetedContext[T]:
    """Selected items plus token accounting for an LLM call."""

    items: list[T]
    fixed_tokens: int
    context_tokens: int
    input_tokens: int
    available_tokens: int
    truncated_item: bool


def select_items_within_budget[T](
    items: Sequence[T],
    *,
    render_item: Callable[[T], str],
    fixed_prefix: str,
    max_input_tokens: int,
    item_separator: str = "\n",
) -> BudgetedContext[T]:
    """Keep highest-priority items that fit under an input-token budget.

    Items are assumed to be in priority order (highest first). Whole items are
    dropped from the end first. An individual item is truncated only when it
    alone exceeds the remaining budget.
    """
    fixed_tokens = count_tokens(fixed_prefix)
    available = max(0, max_input_tokens - fixed_tokens)
    selected: list[T] = []
    rendered_parts: list[str] = []
    used = 0
    truncated_item = False

    for item in items:
        rendered = render_item(item)
        if not rendered:
            continue
        candidate_parts = [*rendered_parts, rendered]
        candidate_text = item_separator.join(candidate_parts)
        candidate_tokens = count_tokens(candidate_text)
        if candidate_tokens <= available:
            selected.append(item)
            rendered_parts = candidate_parts
            used = candidate_tokens
            continue

        if selected:
            # Drop lower-priority remaining items; do not truncate earlier ones.
            break

        # Final safety: truncate the sole highest-priority item to fit.
        room = available
        if room <= 0:
            break
        truncated = _truncate_to_tokens(rendered, room)
        if not truncated:
            break
        selected.append(item)
        rendered_parts = [truncated]
        used = count_tokens(truncated)
        truncated_item = True
        break

    return BudgetedContext(
        items=selected,
        fixed_tokens=fixed_tokens,
        context_tokens=used,
        input_tokens=fixed_tokens + used,
        available_tokens=available,
        truncated_item=truncated_item,
    )


def dedupe_by_id_and_text[T](
    items: Sequence[T],
    *,
    get_id: Callable[[T], str],
    get_text: Callable[[T], str],
) -> list[T]:
    """Drop duplicate IDs and exact-duplicate text, preserving first wins."""
    unique: list[T] = []
    seen_ids: set[str] = set()
    seen_text: set[str] = set()
    for item in items:
        item_id = get_id(item)
        if item_id in seen_ids:
            continue
        text_key = normalize_text(get_text(item))
        if text_key and text_key in seen_text:
            continue
        seen_ids.add(item_id)
        if text_key:
            seen_text.add(text_key)
        unique.append(item)
    return unique


def _truncate_to_tokens(text: str, max_tokens: int) -> str:
    if max_tokens <= 0 or not text:
        return ""
    from app.utils.tokens import decode_tokens, encode_text

    tokens = encode_text(text)
    if len(tokens) <= max_tokens:
        return text
    return decode_tokens(tokens[:max_tokens])
