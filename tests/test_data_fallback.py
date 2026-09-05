import pytest

from llmarcheval.config import TrainConfig
from llmarcheval.train.data import load_tokens


def test_research_mode_fails_loudly_without_fallback(monkeypatch):
    cfg = TrainConfig(dataset="tinystories", max_docs=8, allow_dataset_fallback=False)

    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("llmarcheval.train.data.load_tinystories", boom)
    with pytest.raises(RuntimeError, match="allow_dataset_fallback=False"):
        load_tokens(cfg)


def test_smoke_opt_in_fallback(monkeypatch, tmp_path):
    cfg = TrainConfig(dataset="tinystories", max_docs=8, allow_dataset_fallback=True)

    def boom(*_args, **_kwargs):
        raise RuntimeError("network down")

    monkeypatch.setattr("llmarcheval.train.data.load_tinystories", boom)
    tokens, source = load_tokens(cfg)
    assert tokens.size > 0
    assert source.startswith("fallback:")


def test_explicit_corpus_dataset_still_works():
    cfg = TrainConfig(dataset="smoke_corpus", allow_dataset_fallback=False)
    tokens, source = load_tokens(cfg)
    assert tokens.size > 0
    assert "smoke_corpus" in source
