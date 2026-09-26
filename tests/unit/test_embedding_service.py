"""Unit tests for EmbeddingService."""

from __future__ import annotations

import os
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from app.core.config import Settings
from app.core.exceptions import EmbeddingError
from app.services.embedding_service import EmbeddingService

_BGE_M3_DIM = 1024


class _FakeModel:
    def __init__(self, dim: int = _BGE_M3_DIM) -> None:
        self.dim = dim
        self.encode_calls: list[list[str]] = []

    def encode(
        self,
        texts: list[str],
        normalize_embeddings: bool = False,
        convert_to_numpy: bool = True,
        show_progress_bar: bool = False,
        batch_size: int = 16,
    ) -> np.ndarray:
        self.encode_calls.append(list(texts))
        _ = batch_size
        matrix = np.random.default_rng(42).normal(size=(len(texts), self.dim))
        if normalize_embeddings:
            norms = np.linalg.norm(matrix, axis=1, keepdims=True)
            matrix = matrix / np.clip(norms, 1e-12, None)
        return matrix.astype(np.float32)


def test_embeddings_have_expected_dimensions() -> None:
    service = EmbeddingService(settings=Settings(EMBEDDING_MODEL="fake-model"))
    service._model = _FakeModel(dim=_BGE_M3_DIM)

    embeddings = service.embed(["alpha", "beta"])

    assert embeddings.shape == (2, _BGE_M3_DIM)


def test_embeddings_are_normalized() -> None:
    service = EmbeddingService(settings=Settings(EMBEDDING_MODEL="fake-model"))
    service._model = _FakeModel()

    embeddings = service.embed(["normalized text"])
    norms = np.linalg.norm(embeddings, axis=1)

    assert norms.shape == (1,)
    assert norms[0] == pytest.approx(1.0, abs=1e-5)


def test_batch_embedding_works() -> None:
    fake = _FakeModel()
    service = EmbeddingService(settings=Settings(EMBEDDING_MODEL="fake-model"))
    service._model = fake

    texts = ["one", "two", "three"]
    embeddings = service.embed(texts)

    assert embeddings.shape[0] == 3
    assert fake.encode_calls == [texts]


def test_similarity_uses_dot_product_for_normalized_vectors() -> None:
    service = EmbeddingService(settings=Settings(EMBEDDING_MODEL="fake-model"))
    query = np.array([1.0, 0.0], dtype=np.float32)
    docs = np.array(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [0.6, 0.8],
        ],
        dtype=np.float32,
    )

    scores = service.similarity(query, docs)

    assert scores.tolist() == pytest.approx([1.0, 0.0, 0.6])


def test_model_loads_lazily_once() -> None:
    service = EmbeddingService(
        settings=Settings(EMBEDDING_MODEL="BAAI/bge-m3", HF_TOKEN=None)
    )
    fake_model = _FakeModel()

    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=fake_model,
    ) as ctor:
        assert os.environ.get("DISABLE_SAFETENSORS_CONVERSION") == "1"
        first = service.embed(["a"])
        second = service.embed(["b"])

    assert first.shape[1] == _BGE_M3_DIM
    assert second.shape[0] == 1
    ctor.assert_called_once_with(
        "BAAI/bge-m3",
        device="cpu",
        token=None,
        local_files_only=True,
    )


def test_model_passes_optional_hf_token() -> None:
    service = EmbeddingService(
        settings=Settings(EMBEDDING_MODEL="BAAI/bge-m3", HF_TOKEN="hf_unit_token")
    )
    fake_model = _FakeModel()

    with patch(
        "sentence_transformers.SentenceTransformer",
        return_value=fake_model,
    ) as ctor:
        service.embed(["a"])

    ctor.assert_called_once_with(
        "BAAI/bge-m3",
        device="cpu",
        token="hf_unit_token",
        local_files_only=True,
    )


def test_embed_wraps_failures() -> None:
    service = EmbeddingService(settings=Settings(EMBEDDING_MODEL="fake-model"))

    class _Broken:
        def encode(self, *args: Any, **kwargs: Any) -> np.ndarray:
            raise RuntimeError("boom")

    service._model = _Broken()

    with pytest.raises(EmbeddingError):
        service.embed(["text"])


def test_empty_input_returns_empty_matrix() -> None:
    service = EmbeddingService(settings=Settings(EMBEDDING_MODEL="fake-model"))
    service._model = MagicMock()

    result = service.embed([])

    assert result.shape == (0, 0)
    service._model.encode.assert_not_called()
