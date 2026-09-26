"""Local embedding generation and similarity."""

from __future__ import annotations

import time
from typing import Any

import numpy as np

from app.core.config import Settings, get_settings
from app.core.exceptions import EmbeddingError
from app.core.logging import get_logger
from app.core.metrics import get_embedding_operation, get_request_metrics

logger = get_logger(__name__)


class EmbeddingService:
    """Generate normalized embeddings with a configured SentenceTransformer model."""

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._model: Any | None = None
        self.init_count: int = 0

    def embed(
        self,
        texts: list[str],
        *,
        operation: str | None = None,
    ) -> np.ndarray:
        """Embed texts into an (n, d) float32 matrix of normalized vectors."""
        if not texts:
            return np.zeros((0, 0), dtype=np.float32)

        started = time.perf_counter()
        try:
            model = self._get_model()
            batch_size = self._settings.EMBEDDING_BATCH_SIZE
            embeddings = model.encode(
                texts,
                batch_size=batch_size,
                normalize_embeddings=True,
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            array = np.asarray(embeddings, dtype=np.float32)
            if array.ndim == 1:
                array = array.reshape(1, -1)
            duration_ms = int((time.perf_counter() - started) * 1000)
            self._record_usage(
                text_count=len(texts),
                batch_size=batch_size,
                texts=texts,
                duration_ms=duration_ms,
                operation=operation,
            )
            return array
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.error("embedding_failed", error=str(exc), text_count=len(texts))
            raise EmbeddingError("Failed to generate embeddings") from exc

    def similarity(
        self,
        query_embedding: np.ndarray,
        document_embeddings: np.ndarray,
    ) -> np.ndarray:
        """Return cosine similarities via dot product of normalized vectors."""
        try:
            query = np.asarray(query_embedding, dtype=np.float32)
            docs = np.asarray(document_embeddings, dtype=np.float32)

            if docs.size == 0:
                return np.zeros((0,), dtype=np.float32)
            if query.ndim != 1:
                query = query.reshape(-1)
            if docs.ndim != 2:
                raise EmbeddingError("Document embeddings must be a 2D array")
            if query.shape[0] != docs.shape[1]:
                raise EmbeddingError("Query and document embedding dimensions differ")

            return docs @ query
        except EmbeddingError:
            raise
        except Exception as exc:
            logger.error("similarity_failed", error=str(exc))
            raise EmbeddingError("Failed to compute embedding similarity") from exc

    def _record_usage(
        self,
        *,
        text_count: int,
        batch_size: int,
        texts: list[str],
        duration_ms: int,
        operation: str | None,
    ) -> None:
        metrics = get_request_metrics()
        if metrics is None or text_count <= 0:
            return
        batches = max(1, (text_count + max(batch_size, 1) - 1) // max(batch_size, 1))
        metrics.record_local_embedding_batch(
            self._settings.EMBEDDING_MODEL,
            batches=batches,
            texts=text_count,
        )
        resolved_operation = (
            operation
            or get_embedding_operation()
            or "embed"
        )
        metrics.record_embedding_call(
            operation=resolved_operation,
            texts=texts,
            duration_ms=duration_ms,
        )

    def _get_model(self) -> Any:
        if self._model is not None:
            return self._model

        model_name = self._settings.EMBEDDING_MODEL
        try:
            from sentence_transformers import SentenceTransformer

            logger.info("embedding_model_loading", model=model_name)
            token = (self._settings.HF_TOKEN or "").strip() or None
            # Explicit CPU device; no CUDA dependency path.
            # Models are prepared by scripts/download_models.py before runtime.
            self._model = SentenceTransformer(
                model_name,
                device="cpu",
                token=token,
                local_files_only=True,
            )
            self.init_count += 1
            logger.info("embedding_model_loaded", model=model_name)
            return self._model
        except Exception as exc:
            logger.error(
                "embedding_model_load_failed",
                model=model_name,
                error=str(exc),
            )
            raise EmbeddingError(
                f"Failed to load embedding model: {model_name}"
            ) from exc
