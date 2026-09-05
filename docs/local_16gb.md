# local_16gb research profile

Reproducible Mac / unified-memory profile (~16 GB) for comparing V0–V4
architectural mechanisms at **~37M active parameters**. This is not a pretrain
and not a GPU pilot.

## Dimensions (systematic midpoint)

Between smoke (`4 × 256`, ctx 256) and full (`12 × 768`, ctx 1024):

| Knob | local_16gb |
| --- | --- |
| `n_layer` / unique layers | 8 |
| `n_embd` | 384 |
| `n_head` | 6 (head_dim = 64) |
| `block_size` | 512 |
| V0/V1 `n_inner` | 1428 (shrunk so active params stay near V2–V4) |
| MLA | `kv_lora_rank=32`, `qk_rope_head_dim=16` |
| DSA | `index_topk=64` (≈1/8 of context, same rule as smoke/full) |
| V4 layout | prelude 1 / recurrent 6 / coda 1, `train_loops=2` |

All variants share the same tokenizer, TinyStories dataset, context length,
AdamW + cosine LR, batch size 1, grad accumulation 8, and training budget.

## Unified memory (read this)

On Apple Silicon, **unified memory is shared** by:

- model parameters
- gradients
- optimizer states (AdamW m/v)
- activations / attention scores
- OS and other applications

Therefore **16 GB is not 16 GB available to PyTorch**. Startup prints may include
**analytical memory estimates**; those are planning aids, **not measured peak
memory**. Do not report estimates as measurements.

## MPS / dtype policy

- Preferred device: **MPS**, with an explicit log + **CPU fallback** if MPS is missing.
- Supported dtype for this profile: **float32**.
- This repository does **not** enable bf16/fp16 autocast on MPS (no tested path).
- CUDA mixed precision remains unchanged for other profiles that opt into it.

## Commands

Short mechanical validation (bundled corpus, few steps — not a research run):

```bash
python scripts/validate_local_16gb.py
python scripts/validate_local_16gb.py --all-variants --max-iters 8
```

Short research-shaped run (requires TinyStories / `[data]` extra; no silent fallback):

```bash
python scripts/train.py --train-config configs/local_16gb.yaml --variant v0_dense --max-iters 50
```

## Out of scope (this milestone)

- Long training / 2–5B token pretraining
- New architecture mechanisms
- Agent/tool functionality
- Cybersecurity exploit evaluations
- GPU pilot
- Production MLA/DSA kernels
- MPS peak-memory helper (measured instrumentation comes later)
