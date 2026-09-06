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

    def get_rng_state(self) -> dict:
        """Serialize the private NumPy Generator used for batch sampling."""
        return self.rng.bit_generator.state

    def set_rng_state(self, state: dict) -> None:
        self.rng.bit_generator.state = state

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


def load_tokens(train: TrainConfig, allow_fallback: bool | None = None) -> tuple[np.ndarray, str]:
    """Load and tokenize training data.

    Research / comparative runs must set ``allow_dataset_fallback=False`` (the
    TrainConfig default). Smoke development may opt in to falling back to the
    bundled corpus when TinyStories/FineWeb are unavailable.
    """
    if allow_fallback is None:
        allow_fallback = train.allow_dataset_fallback
    name = train.dataset.lower()
    if name in {"synthetic", "corpus", "smoke_corpus"}:
        return load_corpus_file(DEFAULT_CORPUS), str(DEFAULT_CORPUS)
    if name == "tinystories":
        try:
            return load_tinystories(train.max_docs or 2000), "roneneldan/TinyStories"
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "Failed to load TinyStories and allow_dataset_fallback=False. "
                    "Install the [data] extra / network access, or set "
                    "allow_dataset_fallback: true for smoke-only development."
                ) from exc
            return load_corpus_file(DEFAULT_CORPUS), f"fallback:{DEFAULT_CORPUS}"
    if name in {"fineweb-edu", "fineweb_edu", "fineweb"}:
        try:
            return load_fineweb_edu(train.max_docs or 200), "HuggingFaceFW/fineweb-edu"
        except Exception as exc:
            if not allow_fallback:
                raise RuntimeError(
                    "Failed to load FineWeb-Edu and allow_dataset_fallback=False. "
                    "Install the [data] extra / network access, or set "
                    "allow_dataset_fallback: true for smoke-only development."
                ) from exc
            try:
                return load_tinystories(train.max_docs or 2000), "fallback:tinystories"
            except Exception:
                return load_corpus_file(DEFAULT_CORPUS), f"fallback:{DEFAULT_CORPUS}"
    raise ValueError(f"Unknown dataset {train.dataset!r}")
