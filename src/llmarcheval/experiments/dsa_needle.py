"""DSA needle-in-context selection probe.

Records needle location, DSA selected-token indices (selection+masking over
dense scores), whether the needle span was selected, and a short model answer.
This is not a sparse-attention kernel benchmark.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Sequence

import torch

from llmarcheval.train.checkpoint import load_weights_into_model
from llmarcheval.experiments.common import (
    base_record,
    build_model,
    load_probe_experiment,
    parameter_block,
    resolve_device,
    resolve_dtype,
    set_seed,
    write_result,
)
from llmarcheval.tokenizer import decode, encode

NEEDLE = "The passcode is 93214."
PASSCODE = "93214"
QUERY = " What is the passcode?"
FILLER = "The river is quiet. Clouds drift over the hills. "


def _build_sequence(
    haystack_len: int,
    block_size: int,
    vocab_size: int,
    needle_pos: str = "middle",
) -> tuple[list[int], int, int]:
    filler = encode(FILLER * 200)
    needle_ids = encode(NEEDLE)
    query_ids = encode(QUERY)
    budget = max(haystack_len - len(needle_ids) - len(query_ids), 8)
    if needle_pos == "start":
        prefix_len = 0
    elif needle_pos == "end":
        prefix_len = budget
    else:
        prefix_len = budget // 2
    prefix = filler[:prefix_len]
    rest = filler[: max(budget - prefix_len, 0)]
    sequence = prefix + needle_ids + rest + query_ids
    needle_start = len(prefix)
    needle_end = needle_start + len(needle_ids)
    if len(sequence) > block_size:
        trim = len(sequence) - block_size
        sequence = sequence[trim:]
        needle_start = max(needle_start - trim, 0)
        needle_end = max(needle_end - trim, 0)
    sequence = [t % vocab_size for t in sequence]
    return sequence, needle_start, needle_end


@torch.no_grad()
def run_dsa_needle_experiment(
    *,
    variant: str = "v3_moe_mla_dsa",
    scale: str = "smoke",
    seed: int = 1337,
    device_name: str = "cpu",
    train_config: str = "configs/smoke.yaml",
    haystack_lens: Sequence[int] = (64, 128, 256),
    needle_positions: Sequence[str] = ("middle",),
    answer_tokens: int = 8,
    out_dir: str | Path = "results/experiments",
    write: bool = True,
    ckpt: str | Path | None = None,
) -> dict[str, Any]:
    set_seed(seed)
    exp = load_probe_experiment(variant, train_config=train_config, scale=scale)
    if not exp.model.use_dsa:
        raise ValueError(f"{variant} does not enable DSA")
    device = resolve_device(device_name)
    dtype = resolve_dtype("float32", device)
    model = build_model(exp.model, device, dtype)
    checkpoint_meta = None
    if ckpt is not None:
        checkpoint_meta = load_weights_into_model(model, ckpt, device)
    model.eval()
    params = parameter_block(model, exp.model)

    probes: list[dict[str, Any]] = []
    tokens_processed = 0
    t0 = time.perf_counter()
    for haystack in haystack_lens:
        for pos in needle_positions:
            seq, n0, n1 = _build_sequence(
                min(int(haystack), exp.model.block_size),
                exp.model.block_size,
                exp.model.vocab_size,
                needle_pos=pos,
            )
            idx = torch.tensor([seq], dtype=torch.long, device=device)
            out = model(idx)
            selected_indices: list[int] = []
            needle_selected = None
            if "dsa_keep" in out.stats:
                keep = out.stats["dsa_keep"][0]
                query_pos = keep.size(0) - 1
                selected_indices = keep[query_pos].nonzero(as_tuple=False).view(-1).tolist()
                if n1 > n0:
                    needle_selected = any(n0 <= i < n1 for i in selected_indices)
            gen = model.generate(idx, max_new_tokens=answer_tokens, temperature=0.0)
            answer = decode(gen[0, idx.size(1) :].tolist())
            probes.append(
                {
                    "haystack_len": int(haystack),
                    "seq_len": int(idx.size(1)),
                    "needle_location": {
                        "label": pos,
                        "start": int(n0),
                        "end": int(n1),
                        "span": [int(n0), int(n1)],
                    },
                    "needle_position_label": pos,
                    "needle_span": [int(n0), int(n1)],
                    "dsa_selected_token_indices": [int(i) for i in selected_indices],
                    "dsa_selected_count": len(selected_indices),
                    "needle_span_selected": needle_selected,
                    "model_answer": answer,
                    "retrieval_correct": PASSCODE in answer,
                    "passcode_in_answer": PASSCODE in answer,
                    "implementation_note": (
                        "DSA here is selection + masking over dense attention scores; "
                        "not a sparse-attention kernel benchmark."
                    ),
                }
            )
            tokens_processed += int(idx.numel())
    wall = time.perf_counter() - t0

    record = base_record(
        experiment="dsa_needle",
        variant=variant,
        seed=seed,
        exp=exp,
        device=device,
        dtype=dtype,
        params=params,
        context_length=exp.model.block_size,
        wall_clock_s=wall,
        tokens_processed=tokens_processed,
        training_steps=checkpoint_meta.get("step") if checkpoint_meta else None,
        metrics={
            "checkpoint": checkpoint_meta,
            "probes": probes,
            "n_probes": len(probes),
            "needle_selected_rate": sum(1 for p in probes if p["needle_span_selected"])
            / max(len(probes), 1),
            "retrieval_correct_rate": sum(1 for p in probes if p["retrieval_correct"])
            / max(len(probes), 1),
        },
        notes=[
            (
                "Fresh model initialization (no checkpoint)."
                if checkpoint_meta is None
                else f"Weights loaded from checkpoint {checkpoint_meta['name']}."
            ),

            "Offline synthetic haystacks; no network required.",
            "DSA implementation performs selection + masking on dense scores.",
            "Do not describe this as a sparse GEMM / production DSA benchmark.",
        ],
    )
    if write:
        path = write_result(record, out_dir)
        record["output_path"] = str(path)
    return record
