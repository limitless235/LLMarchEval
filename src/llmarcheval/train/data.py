"""Token packing and dataset loaders."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from llmarcheval.config import TrainConfig
from llmarcheval.tokenizer import encode

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_CORPUS = REPO_ROOT / "data" / "smoke_corpus.txt"


class TokenBatcher:
    def __init__(self, tokens: np.ndarray, block_size: int, batch_size: int, seed: int = 1337) -> None:
        if tokens.size <= block_size + 1:
            raise ValueError("Not enough tokens for the configured block_size")
        self.tokens = tokens
        self.block_size = block_size
        self.batch_size = batch_size
        self.rng = np.random.default_rng(seed)

    def get_batch(self, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        max_start = self.tokens.size - self.block_size - 1
        starts = self.rng.integers(0, max_start, size=self.batch_size)
        x = np.stack([self.tokens[i : i + self.block_size] for i in starts])
        y = np.stack([self.tokens[i + 1 : i + 1 + self.block_size] for i in starts])
        return (
            torch.from_numpy(x.astype(np.int64)).to(device),
            torch.from_numpy(y.astype(np.int64)).to(device),
        )


def _tokenize_text(text: str) -> np.ndarray:
    ids = encode(text)
    if not ids:
        raise ValueError("Tokenizer produced no tokens")
    return np.array(ids, dtype=np.uint16)


def load_corpus_file(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8")
    return _tokenize_text(text)


def load_tinystories(max_docs: int) -> np.ndarray:
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets extra: pip install 'llmarcheval[data]'") from exc
    split = f"train[:{max_docs}]" if max_docs else "train[:2000]"
    dataset = load_dataset("roneneldan/TinyStories", split=split)
    text = "\n\n".join(item["text"] for item in dataset)
    return _tokenize_text(text)


def load_fineweb_edu(max_docs: int) -> np.ndarray:
    try:
        from datasets import load_dataset
    except ImportError as extra:
        raise RuntimeError("Install datasets extra: pip install 'llmarcheval[data]'") from extra
    stream = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name="sample-10BT",
        split="train",
        streaming=True,
    )
    chunks: list[str] = []
    limit = max_docs if max_docs else 200
    for i, row in enumerate(stream):
        chunks.append(row["text"])
        if i + 1 >= limit:
            break
    return _tokenize_text("\n\n".join(chunks))


def load_tokens(train: TrainConfig, allow_fallback: bool = True) -> tuple[np.ndarray, str]:
    name = train.dataset.lower()
    if name in {"synthetic", "corpus", "smoke_corpus"}:
        return load_corpus_file(DEFAULT_CORPUS), str(DEFAULT_CORPUS)
    if name == "tinystories":
        try:
            return load_tinystories(train.max_docs), "roneneldan/TinyStories"
        except Exception:
            if not allow_fallback:
                raise
            return load_corpus_file(DEFAULT_CORPUS), f"fallback:{DEFAULT_CORPUS}"
    if name in {"fineweb-edu", "fineweb_edu", "fineweb"}:
        try:
            return load_fineweb_edu(train.max_docs or 200), "HuggingFaceFW/fineweb-edu"
        except Exception:
            if not allow_fallback:
                raise
            try:
                return load_tinystories(train.max_docs or 2000), "fallback:tinystories"
            except Exception:
                return load_corpus_file(DEFAULT_CORPUS), f"fallback:{DEFAULT_CORPUS}"
    raise ValueError(f"Unknown dataset {train.dataset!r}")
