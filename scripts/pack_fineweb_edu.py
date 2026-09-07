#!/usr/bin/env python3
"""Pack a frozen FineWeb-Edu token corpus for the 1M/20M architecture matrix.

Stops at a TOKEN QUOTA (default 100M GPT-2 tokens), not max_docs.

Writes:
  <out-dir>/tokens.uint16.npy
  <out-dir>/manifest.json
  <out-dir>/tokens.uint16.npy.sha256

Does NOT start training. Prefer --dry-run first.

Example (after approval to download):
  python scripts/pack_fineweb_edu.py --target-tokens 100000000 \\
      --out-dir data/fineweb_edu_100m --seed 1337
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from llmarcheval.tokenizer import GPT2_VOCAB_SIZE, encode

DEFAULT_DATASET_ID = "HuggingFaceFW/fineweb-edu"
DEFAULT_CONFIG = "sample-10BT"
DOC_JOINER = "\n\n"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_save_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    tmp_path = Path(tmp_name)
    try:
        os.close(fd)
        with tmp_path.open("wb") as handle:
            np.save(handle, array)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, path)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def pack_fineweb_edu(
    *,
    target_tokens: int,
    out_dir: Path,
    seed: int,
    revision: str | None,
    dataset_id: str = DEFAULT_DATASET_ID,
    dataset_config: str = DEFAULT_CONFIG,
    dry_run: bool = False,
) -> dict:
    """Stream FineWeb-Edu until ``target_tokens`` packed GPT-2 ids are collected."""
    if target_tokens <= 0:
        raise ValueError("target_tokens must be positive")

    out_dir = Path(out_dir)
    tokens_path = out_dir / "tokens.uint16.npy"
    manifest_path = out_dir / "manifest.json"
    checksum_path = out_dir / "tokens.uint16.npy.sha256"

    plan = {
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_revision": revision,
        "target_packed_tokens": int(target_tokens),
        "seed": int(seed),
        "sampling": "streaming_take_until_token_quota",
        "out_dir": str(out_dir),
        "tokens_path": str(tokens_path),
        "dry_run": bool(dry_run),
    }
    print(json.dumps({"plan": plan}, indent=2), flush=True)
    if dry_run:
        return plan

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("Install datasets extra: pip install 'llmarcheval[data]'") from exc

    load_kwargs = {
        "path": dataset_id,
        "name": dataset_config,
        "split": "train",
        "streaming": True,
    }
    if revision:
        load_kwargs["revision"] = revision

    # Record resolved revision when the datasets API exposes it.
    stream = load_dataset(**load_kwargs)
    resolved_revision = revision
    try:
        info = getattr(stream, "info", None) or getattr(getattr(stream, "dataset", None), "info", None)
        if info is not None and getattr(info, "version", None) is not None:
            resolved_revision = resolved_revision or str(info.version)
    except Exception:
        pass

    # Seed is recorded for provenance. Streaming take-until-quota is sequential
    # under a pinned revision (order is HF shard order), not a reshuffle of the
    # full 10BT set (impractical on a 16 GB Mac).
    _ = seed

    chunks: list[np.ndarray] = []
    n_tokens = 0
    n_docs = 0
    joiner_ids = np.array(encode(DOC_JOINER), dtype=np.uint16)

    for row in stream:
        text = row.get("text") or ""
        if not text:
            continue
        ids = np.array(encode(text), dtype=np.uint16)
        if ids.size == 0:
            continue
        if chunks:
            # Match legacy load_fineweb_edu joining: docs separated by "\n\n".
            remain = target_tokens - n_tokens
            if remain <= 0:
                break
            take_join = joiner_ids[: min(len(joiner_ids), remain)]
            chunks.append(take_join)
            n_tokens += int(take_join.size)
            if n_tokens >= target_tokens:
                break
            remain = target_tokens - n_tokens
            if remain <= 0:
                break
            take = ids[:remain]
            chunks.append(take)
            n_tokens += int(take.size)
        else:
            take = ids[: target_tokens - n_tokens]
            chunks.append(take)
            n_tokens += int(take.size)
        n_docs += 1
        if n_docs % 500 == 0:
            print(f"packed docs={n_docs} tokens={n_tokens}", flush=True)
        if n_tokens >= target_tokens:
            break

    if n_tokens <= 0:
        raise RuntimeError("No tokens packed from FineWeb-Edu stream")

    packed = np.concatenate(chunks).astype(np.uint16, copy=False)
    if packed.size > target_tokens:
        packed = packed[:target_tokens]
    if int(packed.max(initial=0)) >= GPT2_VOCAB_SIZE:
        raise RuntimeError("Packed token id exceeds GPT-2 vocab size")

    split = max(int(packed.size * 0.9), 512 + 2)
    train_n = min(split, packed.size)
    val_n = packed.size - train_n

    _atomic_save_npy(tokens_path, packed)
    digest = _sha256_file(tokens_path)
    _atomic_write_bytes(checksum_path, f"{digest}  {tokens_path.name}\n".encode("utf-8"))

    manifest = {
        "dataset_id": dataset_id,
        "dataset_config": dataset_config,
        "dataset_revision": resolved_revision,
        "license": "ODC-By-1.0 (+ CommonCrawl Terms of Use)",
        "tokenizer": "tiktoken/gpt2",
        "tokenizer_vocab_size": GPT2_VOCAB_SIZE,
        "preprocessing": {
            "doc_joiner": DOC_JOINER,
            "stop_criterion": "token_quota",
            "target_packed_tokens": int(target_tokens),
            "sampling": "streaming_take_until_token_quota",
            "seed": int(seed),
            "docs_consumed": int(n_docs),
        },
        "actual_packed_tokens": int(packed.size),
        "train_tokens_after_90_10": int(train_n),
        "val_tokens_after_90_10": int(val_n),
        "split_rule": "prefix_90pct_train_suffix_10pct_val",
        "dtype": "uint16",
        "packed_path": str(tokens_path),
        "sha256": digest,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "notes": (
            "Frozen corpus for FineWeb-Edu 1M/20M local_16gb_w768 matrix; "
            "TinyStories 1M remains a separate historical pilot."
        ),
    }
    _atomic_write_bytes(manifest_path, (json.dumps(manifest, indent=2) + "\n").encode("utf-8"))
    print(json.dumps({"wrote": manifest}, indent=2), flush=True)
    return manifest


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Pack frozen FineWeb-Edu GPT-2 token corpus")
    parser.add_argument("--target-tokens", type=int, default=100_000_000)
    parser.add_argument("--out-dir", type=Path, default=Path("data/fineweb_edu_100m"))
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        help="Optional HF dataset revision/commit to pin (recorded in manifest)",
    )
    parser.add_argument("--dataset-id", type=str, default=DEFAULT_DATASET_ID)
    parser.add_argument("--dataset-config", type=str, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the packing plan without downloading or writing tokens",
    )
    args = parser.parse_args(argv)
    pack_fineweb_edu(
        target_tokens=args.target_tokens,
        out_dir=args.out_dir,
        seed=args.seed,
        revision=args.revision,
        dataset_id=args.dataset_id,
        dataset_config=args.dataset_config,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
