"""Unit tests for lightweight Hugging Face model preparation.

All Hugging Face calls are mocked — these tests download ZERO bytes.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock, call, patch

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError
from pydantic import ValidationError

from app.core.config import Settings, get_settings
from scripts import download_models


@pytest.fixture(autouse=True)
def _clear_settings_cache() -> None:
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_embedding_allow_patterns_exclude_unused_artifacts() -> None:
    patterns = download_models.EMBEDDING_ALLOW_PATTERNS
    joined = "\n".join(patterns).lower()

    assert "onnx" not in joined
    assert "imgs" not in joined
    assert "readme" not in joined
    assert "long.jpg" not in joined
    assert "colbert_linear.pt" not in patterns
    assert "sparse_linear.pt" not in patterns
    assert ".gitattributes" not in patterns


def test_embedding_allow_patterns_include_required_files() -> None:
    required = {
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
    }
    assert set(download_models.EMBEDDING_ALLOW_PATTERNS) == required


def test_reranker_allow_patterns_include_required_files() -> None:
    required = {
        "config.json",
        "model.safetensors",
        "sentencepiece.bpe.model",
        "special_tokens_map.json",
        "tokenizer.json",
        "tokenizer_config.json",
    }
    assert set(download_models.RERANKER_ALLOW_PATTERNS) == required
    joined = "\n".join(download_models.RERANKER_ALLOW_PATTERNS).lower()
    assert "readme" not in joined
    assert "assets" not in joined


def test_hf_download_max_workers_defaults_to_eight() -> None:
    assert Settings().HF_DOWNLOAD_MAX_WORKERS == 8


def test_hf_download_max_workers_rejects_zero() -> None:
    with pytest.raises(ValidationError):
        Settings(HF_DOWNLOAD_MAX_WORKERS=0)


def test_both_preparations_are_submitted_to_executor(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Both embedding and reranker jobs are independently scheduled."""
    settings = Settings(
        EMBEDDING_MODEL="org/embed-model",
        RERANKER_MODEL="org/rerank-model",
        HF_TOKEN=None,
        HF_DOWNLOAD_MAX_WORKERS=8,
    )
    submitted_labels: list[str] = []
    real_submit = ThreadPoolExecutor.submit

    def tracking_submit(
        self: ThreadPoolExecutor,
        fn: object,
        /,
        *args: object,
        **kwargs: object,
    ) -> object:
        label = kwargs.get("label")
        if isinstance(label, str):
            submitted_labels.append(label)
        return real_submit(self, fn, *args, **kwargs)  # type: ignore[arg-type]

    with (
        patch.object(download_models, "get_settings", return_value=settings),
        patch.object(download_models, "_is_cached", return_value=True),
        patch.object(
            ThreadPoolExecutor,
            "submit",
            tracking_submit,
        ),
    ):
        assert download_models.main() == 0

    assert set(submitted_labels) == {"embedding", "reranker"}
    assert len(submitted_labels) == 2
    out = capsys.readouterr().out
    assert "Models ready." in out


def test_both_successful_futures_complete(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        EMBEDDING_MODEL="org/embed-model",
        RERANKER_MODEL="org/rerank-model",
        HF_TOKEN=None,
    )
    prepare_embedding = MagicMock()
    prepare_reranker = MagicMock()

    with (
        patch.object(download_models, "get_settings", return_value=settings),
        patch.object(download_models, "_is_cached", return_value=False),
        patch.object(download_models, "_prepare_embedding", prepare_embedding),
        patch.object(download_models, "_prepare_reranker", prepare_reranker),
    ):
        assert download_models.main() == 0

    prepare_embedding.assert_called_once_with("org/embed-model", None, 8)
    prepare_reranker.assert_called_once_with("org/rerank-model", None, 8)
    out = capsys.readouterr().out
    assert "[embedding] downloading required files" in out
    assert "[reranker] downloading required files" in out
    assert "[embedding] ready" in out
    assert "[reranker] ready" in out
    assert "Models ready." in out


def test_cached_models_skip_remote_download(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        EMBEDDING_MODEL="org/embed-model",
        RERANKER_MODEL="org/rerank-model",
        HF_TOKEN=None,
    )

    def raise_if_called(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("heavy preparation must not run on cache hit")

    with (
        patch.object(download_models, "get_settings", return_value=settings),
        patch.object(download_models, "_is_cached", return_value=True) as is_cached,
        patch.object(
            download_models,
            "_prepare_embedding",
            side_effect=raise_if_called,
        ),
        patch.object(
            download_models,
            "_prepare_reranker",
            side_effect=raise_if_called,
        ),
        patch.object(download_models, "snapshot_download", side_effect=raise_if_called),
    ):
        assert download_models.main() == 0

    assert is_cached.call_count == 2
    out = capsys.readouterr().out
    assert "[embedding] checking org/embed-model" in out
    assert "[reranker] checking org/rerank-model" in out
    assert "[embedding] ready" in out
    assert "[reranker] ready" in out
    assert "Models ready." in out
    assert "downloading" not in out


def test_cache_miss_uses_targeted_allow_patterns() -> None:
    settings = Settings(
        EMBEDDING_MODEL="org/embed-model",
        RERANKER_MODEL="org/rerank-model",
        HF_TOKEN=None,
        HF_DOWNLOAD_MAX_WORKERS=4,
    )
    snapshot = MagicMock(return_value="/cache/path")

    with (
        patch.object(download_models, "get_settings", return_value=settings),
        patch.object(
            download_models,
            "_is_cached",
            return_value=False,
        ),
        patch.object(download_models, "snapshot_download", snapshot),
    ):
        assert download_models.main() == 0

    assert snapshot.call_count == 2
    snapshot.assert_has_calls(
        [
            call(
                repo_id="org/embed-model",
                allow_patterns=list(download_models.EMBEDDING_ALLOW_PATTERNS),
                token=None,
                max_workers=4,
            ),
            call(
                repo_id="org/rerank-model",
                allow_patterns=list(download_models.RERANKER_ALLOW_PATTERNS),
                token=None,
                max_workers=4,
            ),
        ],
        any_order=True,
    )


def test_configured_max_workers_reaches_snapshot_download() -> None:
    with patch.object(download_models, "snapshot_download") as snapshot:
        download_models._prepare_embedding("BAAI/bge-m3", None, max_workers=8)

    snapshot.assert_called_once_with(
        repo_id="BAAI/bge-m3",
        allow_patterns=list(download_models.EMBEDDING_ALLOW_PATTERNS),
        token=None,
        max_workers=8,
    )


def test_prepare_embedding_uses_targeted_snapshot() -> None:
    with patch.object(download_models, "snapshot_download") as snapshot:
        download_models._prepare_embedding("BAAI/bge-m3", None, 8)

    snapshot.assert_called_once_with(
        repo_id="BAAI/bge-m3",
        allow_patterns=list(download_models.EMBEDDING_ALLOW_PATTERNS),
        token=None,
        max_workers=8,
    )
    kwargs = snapshot.call_args.kwargs
    assert "force_download" not in kwargs
    patterns = kwargs["allow_patterns"]
    assert "onnx/**" not in patterns
    assert "pytorch_model.bin" in patterns


def test_prepare_reranker_uses_targeted_snapshot() -> None:
    with patch.object(download_models, "snapshot_download") as snapshot:
        download_models._prepare_reranker("BAAI/bge-reranker-v2-m3", None, 8)

    snapshot.assert_called_once_with(
        repo_id="BAAI/bge-reranker-v2-m3",
        allow_patterns=list(download_models.RERANKER_ALLOW_PATTERNS),
        token=None,
        max_workers=8,
    )
    assert "model.safetensors" in snapshot.call_args.kwargs["allow_patterns"]


def test_hf_token_absent_still_prepares_public_models() -> None:
    settings = Settings(HF_TOKEN=None)
    assert download_models._hf_token(settings) is None

    with patch.object(download_models, "snapshot_download") as snapshot:
        download_models._prepare_embedding("org/public-model", None, 8)

    snapshot.assert_called_once_with(
        repo_id="org/public-model",
        allow_patterns=list(download_models.EMBEDDING_ALLOW_PATTERNS),
        token=None,
        max_workers=8,
    )


def test_hf_token_passed_explicitly_to_hub_apis() -> None:
    settings = Settings(HF_TOKEN="hf_test_token_value")
    token = download_models._hf_token(settings)
    assert token == "hf_test_token_value"

    with patch.object(download_models, "snapshot_download") as snapshot:
        download_models._download_model(
            "org/model",
            token,
            download_models.EMBEDDING_ALLOW_PATTERNS,
            8,
        )

    snapshot.assert_called_once_with(
        repo_id="org/model",
        allow_patterns=list(download_models.EMBEDDING_ALLOW_PATTERNS),
        token="hf_test_token_value",
        max_workers=8,
    )


def test_cache_miss_main_passes_token_to_prepare() -> None:
    settings = Settings(
        EMBEDDING_MODEL="org/embed",
        RERANKER_MODEL="org/rerank",
        HF_TOKEN="hf_secret",
        HF_DOWNLOAD_MAX_WORKERS=8,
    )
    prepare_embedding = MagicMock()
    prepare_reranker = MagicMock()

    with (
        patch.object(download_models, "get_settings", return_value=settings),
        patch.object(download_models, "_is_cached", return_value=False),
        patch.object(download_models, "_prepare_embedding", prepare_embedding),
        patch.object(download_models, "_prepare_reranker", prepare_reranker),
    ):
        assert download_models.main() == 0

    prepare_embedding.assert_called_once_with("org/embed", "hf_secret", 8)
    prepare_reranker.assert_called_once_with("org/rerank", "hf_secret", 8)


def test_is_cached_uses_local_files_only_allow_patterns_and_token() -> None:
    patterns = download_models.EMBEDDING_ALLOW_PATTERNS
    with patch.object(
        download_models,
        "snapshot_download",
        return_value="/cache/path",
    ) as snapshot:
        assert download_models._is_cached("org/model", "hf_token", patterns) is True

    snapshot.assert_called_once_with(
        repo_id="org/model",
        local_files_only=True,
        allow_patterns=list(patterns),
        token="hf_token",
    )


def test_is_cached_returns_false_on_miss() -> None:
    with patch.object(
        download_models,
        "snapshot_download",
        side_effect=LocalEntryNotFoundError("missing"),
    ):
        assert (
            download_models._is_cached(
                "org/model",
                None,
                download_models.EMBEDDING_ALLOW_PATTERNS,
            )
            is False
        )


def test_embedding_failure_propagates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        EMBEDDING_MODEL="org/embed-model",
        RERANKER_MODEL="org/rerank-model",
        HF_TOKEN=None,
    )

    def fail_embedding(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("embedding download failed")

    with (
        patch.object(download_models, "get_settings", return_value=settings),
        patch.object(download_models, "_is_cached", return_value=False),
        patch.object(
            download_models,
            "_prepare_embedding",
            side_effect=fail_embedding,
        ),
        patch.object(download_models, "_prepare_reranker", MagicMock()),
    ):
        assert download_models.main() == 1

    err = capsys.readouterr().err
    assert "Model preparation failed: embedding / org/embed-model" in err


def test_reranker_failure_propagates(
    capsys: pytest.CaptureFixture[str],
) -> None:
    settings = Settings(
        EMBEDDING_MODEL="org/embed-model",
        RERANKER_MODEL="org/rerank-model",
        HF_TOKEN=None,
    )

    def fail_reranker(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("reranker download failed")

    with (
        patch.object(download_models, "get_settings", return_value=settings),
        patch.object(download_models, "_is_cached", return_value=False),
        patch.object(download_models, "_prepare_embedding", MagicMock()),
        patch.object(
            download_models,
            "_prepare_reranker",
            side_effect=fail_reranker,
        ),
    ):
        assert download_models.main() == 1

    err = capsys.readouterr().err
    assert "Model preparation failed: reranker / org/rerank-model" in err


def test_no_heavy_model_constructors_used() -> None:
    """Preparation must never construct SentenceTransformer or FlagReranker."""
    settings = Settings(
        EMBEDDING_MODEL="org/embed-model",
        RERANKER_MODEL="org/rerank-model",
        HF_TOKEN=None,
    )
    forbidden = MagicMock(side_effect=AssertionError("heavy constructor used"))

    with (
        patch.object(download_models, "get_settings", return_value=settings),
        patch.object(download_models, "_is_cached", return_value=False),
        patch.object(download_models, "snapshot_download", return_value="/cache"),
        patch.dict(
            "sys.modules",
            {
                "sentence_transformers": MagicMock(
                    SentenceTransformer=forbidden,
                ),
                "FlagEmbedding": MagicMock(FlagReranker=forbidden),
            },
        ),
    ):
        assert download_models.main() == 0

    forbidden.assert_not_called()
