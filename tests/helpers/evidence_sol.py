"""Shared helpers for shared-corpus Sol evidence mocks."""

from __future__ import annotations

import ast
import re
from unittest.mock import AsyncMock

from app.schemas.evaluation import (
    CriterionEvaluation,
    EvidenceCitation,
    EvidenceEvaluationBatch,
    ExclusionEvaluation,
    SolCriterionEvaluation,
    SolEvidenceEvaluationBatch,
    SolExclusionEvaluation,
)

_ITEM_ID_RE = re.compile(r"^- (?:criterion_id|exclusion_id):\s*(\S+)\s*$")
_ALLOWED_IDS_RE = re.compile(r"^\s*allowed_candidate_ids:\s*(\[.*\])\s*$")
_CORPUS_CANDIDATE_RE = re.compile(r"^- candidate_id:\s*(E\d+)\s*$")
_CORPUS_TEXT_MARKER = "  text:"


def parse_prompt_corpus_texts(prompt: str) -> dict[str, str]:
    """Map shared-corpus candidate_id → rendered text."""
    corpus_text_by_id: dict[str, str] = {}
    lines = prompt.splitlines()
    index = 0
    in_corpus = False
    while index < len(lines):
        line = lines[index]
        stripped = line.strip()
        if stripped == "evidence_corpus:":
            in_corpus = True
            index += 1
            continue
        if stripped == "items:":
            break
        if in_corpus:
            match = _CORPUS_CANDIDATE_RE.match(line)
            if match:
                candidate_id = match.group(1)
                text_lines: list[str] = []
                index += 1
                collecting_text = False
                while index < len(lines):
                    nxt = lines[index]
                    if nxt.startswith("- candidate_id:") or nxt.strip() == "items:":
                        break
                    if nxt == _CORPUS_TEXT_MARKER or nxt.strip() == "text:":
                        collecting_text = True
                        index += 1
                        continue
                    if collecting_text:
                        # Production prompt emits chunk text unindented after "  text:".
                        text_lines.append(nxt)
                    index += 1
                corpus_text_by_id[candidate_id] = "\n".join(text_lines).strip()
                continue
        index += 1
    return corpus_text_by_id


def parse_prompt_candidate_lookup(prompt: str) -> dict[tuple[str, str], str]:
    """Map (item_id, chunk_text_fingerprint) → candidate_id from shared corpus.

    The new prompt format places chunk text once in ``evidence_corpus`` and
    lists ``allowed_candidate_ids`` per item. For tests we map via the
    candidate_id itself when chunk_id is unavailable, falling back to matching
    evidence text against the corpus.
    """
    corpus_text_by_id = parse_prompt_corpus_texts(prompt)
    lines = prompt.splitlines()

    lookup: dict[tuple[str, str], str] = {}
    current_item: str | None = None
    for raw_line in lines:
        item_match = _ITEM_ID_RE.match(raw_line)
        if item_match:
            current_item = item_match.group(1)
            continue
        allowed_match = _ALLOWED_IDS_RE.match(raw_line)
        if allowed_match and current_item is not None:
            try:
                allowed = ast.literal_eval(allowed_match.group(1))
            except (SyntaxError, ValueError):
                allowed = []
            for candidate_id in allowed:
                text = corpus_text_by_id.get(str(candidate_id), "")
                # Key by (item, candidate_id) and also (item, text) for adapters.
                lookup[(current_item, str(candidate_id))] = str(candidate_id)
                if text:
                    lookup[(current_item, text)] = str(candidate_id)
    return lookup


def public_batch_to_scoped_sol(
    public: EvidenceEvaluationBatch,
    *,
    prompt: str,
) -> SolEvidenceEvaluationBatch:
    """Translate public chunk citations into shared-corpus candidate IDs."""
    lookup = parse_prompt_candidate_lookup(prompt)
    corpus_text_by_id = parse_prompt_corpus_texts(prompt)

    def resolve(item_id: str, evidence: list[EvidenceCitation]) -> list[str]:
        resolved: list[str] = []
        for citation in evidence:
            candidate_id = lookup.get((item_id, citation.text))
            if candidate_id is None:
                candidate_id = lookup.get((item_id, citation.chunk_id))
            if candidate_id is None and citation.text:
                # Match substring/superset against this item's allowed corpus texts.
                for key, value in lookup.items():
                    if key[0] != item_id or value in resolved:
                        continue
                    corpus_text = corpus_text_by_id.get(value, "")
                    if not corpus_text:
                        continue
                    if citation.text in corpus_text or corpus_text in citation.text:
                        candidate_id = value
                        break
            if candidate_id is None:
                # Fall back: any unused allowed ID for this item.
                for key, value in lookup.items():
                    if key[0] == item_id and value not in resolved:
                        candidate_id = value
                        break
            if candidate_id is None:
                candidate_id = f"E{len(resolved) + 1}"
            resolved.append(candidate_id)
        return resolved

    return SolEvidenceEvaluationBatch(
        criteria=[
            SolCriterionEvaluation(
                criterion_id=item.criterion_id,
                status=item.status,
                evidence_candidate_ids=resolve(item.criterion_id, item.evidence),
                explanation=item.explanation,
            )
            for item in public.criteria
        ],
        exclusions=[
            SolExclusionEvaluation(
                exclusion_id=item.exclusion_id,
                triggered=item.triggered,
                status=item.status,
                evidence_candidate_ids=resolve(item.exclusion_id, item.evidence),
                explanation=item.explanation,
            )
            for item in public.exclusions
        ],
    )


def as_sol_response(
    llm_response: EvidenceEvaluationBatch | SolEvidenceEvaluationBatch,
) -> AsyncMock:
    if isinstance(llm_response, SolEvidenceEvaluationBatch):
        return AsyncMock(return_value=llm_response)

    async def _adapt(**kwargs: object) -> SolEvidenceEvaluationBatch:
        prompt = str(kwargs.get("user_prompt") or "")
        return public_batch_to_scoped_sol(llm_response, prompt=prompt)

    return AsyncMock(side_effect=_adapt)


# Re-export for type checkers / convenience.
__all__ = [
    "CriterionEvaluation",
    "ExclusionEvaluation",
    "as_sol_response",
    "parse_prompt_candidate_lookup",
    "parse_prompt_corpus_texts",
    "public_batch_to_scoped_sol",
]
