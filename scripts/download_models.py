#!/usr/bin/env python3
"""Prepare local Hugging Face embedding and reranker models for runtime use.

Cached models are verified against the runtime-required artifact set only.
A partial snapshot with all required files counts as cached; excluded
repository assets (ONNX, images, docs, sparse/ColBERT weights) are ignored.

SentenceTransformer / FlagReranker are never constructed here; runtime
services load them lazily on first inference.

Embedding and reranker preparation run concurrently via ThreadPoolExecutor.
"""

from __future__ import annotations

import sys
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

from huggingface_hub import constants, snapshot_download
from huggingface_hub.errors import LocalEntryNotFoundError

from app.core.config import Settings, get_settings

# Dense SentenceTransformer path for BAAI/bge-m3. Verified against
# sentence_transformers loading of modules.json (Transformer + Pooling +
# Normalize). Normalize needs no files; Pooling needs 1_Pooling/config.json.
EMBEDDING_ALLOW_PATTERNS: tuple[str, ...] = (
    "1_Pooling/config.json",
    "config.json",
    "config_sentence_transformers.json",
    "modules.json",
    "pytorch_model.bin",
    "sentence_bert_config.json",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)

# FlagReranker (AutoTokenizer + AutoModelForSequenceClassification) path for
# BAAI/bge-reranker-v2-m3.
RERANKER_ALLOW_PATTERNS: tuple[str, ...] = (
    "config.json",
    "model.safetensors",
    "sentencepiece.bpe.model",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
)


def _hf_token(settings: Settings) -> str | None:
    token = (settings.HF_TOKEN or "").strip()
    return token or None


def _is_cached(
    model_name: str,
    token: str | None,
    allow_patterns: Sequence[str],
) -> bool:
    """Return True when all runtime-required files exist locally (no network)."""
    try:
        snapshot_download(
            repo_id=model_name,
            local_files_only=True,
            allow_patterns=list(allow_patterns),
            token=token,
        )
        return True
    except LocalEntryNotFoundError:
        return False


def _download_model(
    model_name: str,
    token: str | None,
    allow_patterns: Sequence[str],
    max_workers: int,
) -> None:
    """Download only runtime-required weights into the Hugging Face cache."""
    snapshot_download(
        repo_id=model_name,
        allow_patterns=list(allow_patterns),
        token=token,
        max_workers=max_workers,
    )


def _prepare_embedding(
    model_name: str,
    token: str | None,
    max_workers: int,
) -> None:
    _download_model(model_name, token, EMBEDDING_ALLOW_PATTERNS, max_workers)


def _prepare_reranker(
    model_name: str,
    token: str | None,
    max_workers: int,
) -> None:
    _download_model(model_name, token, RERANKER_ALLOW_PATTERNS, max_workers)


def _ensure_model(
    *,
    label: str,
    model_name: str,
    token: str | None,
    allow_patterns: Sequence[str],
    prepare: Callable[[str, str | None, int], None],
    max_workers: int,
) -> None:
    print(f"[{label}] checking {model_name}", flush=True)
    if _is_cached(model_name, token, allow_patterns):
        print(f"[{label}] ready", flush=True)
        return

    print(f"[{label}] downloading required files", flush=True)
    prepare(model_name, token, max_workers)
    print(f"[{label}] ready", flush=True)


def main() -> int:
    settings = get_settings()
    token = _hf_token(settings)
    max_workers = settings.HF_DOWNLOAD_MAX_WORKERS

    print(f"Hugging Face cache: {constants.HF_HUB_CACHE}", flush=True)
    print(flush=True)

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = {
            executor.submit(
                _ensure_model,
                label="embedding",
                model_name=settings.EMBEDDING_MODEL,
                token=token,
                allow_patterns=EMBEDDING_ALLOW_PATTERNS,
                prepare=_prepare_embedding,
                max_workers=max_workers,
            ): ("embedding", settings.EMBEDDING_MODEL),
            executor.submit(
                _ensure_model,
                label="reranker",
                model_name=settings.RERANKER_MODEL,
                token=token,
                allow_patterns=RERANKER_ALLOW_PATTERNS,
                prepare=_prepare_reranker,
                max_workers=max_workers,
            ): ("reranker", settings.RERANKER_MODEL),
        }

        for future in as_completed(futures):
            label, model_name = futures[future]
            try:
                future.result()
            except Exception as exc:
                for other in futures:
                    if other is not future and not other.done():
                        other.cancel()
                print(
                    f"Model preparation failed: {label} / {model_name}\n{exc}",
                    file=sys.stderr,
                )
                return 1

    print(flush=True)
    print("Models ready.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
