# local_16gb_w768 opt-in scale profile

Isolated **~100M active** width scale-up that reuses the frozen `local_16gb`
training protocol (tokenizer, TinyStories, ctx 512, AdamW, FP32, B=1,
grad_accum=8). It does **not** replace [`configs/local_16gb.yaml`](../configs/local_16gb.yaml)
(~37M active) and is **not** a training campaign by itself.

## Dimensions

| Knob | local_16gb_w768 |
| --- | --- |
| `n_layer` / unique layers | 8 |
| `n_embd` | 768 |
| `n_head` | 12 (head_dim = 64) |
| `block_size` | 512 |
| V0/V1 `n_inner` | 2880 |
| V2/V3/V4 `n_inner` | 3200 |
| MLA | `kv_lora_rank=32`, `qk_rope_head_dim=16` (lab simplification) |
| DSA | `index_topk=64` (lab mask selection, not production sparse DSA) |
| V4 layout | prelude 1 / recurrent 6 / coda 1, `train_loops=2` (looped compute hypothesis) |

Active-parameter spread across V0–V4 is designed to stay ≤ 2%.

## Feasibility gates (opt-in)

These are mechanical smoke checks, **not scientific evidence**:

```bash
# Default: all five V0–V4 (opt-in feasibility gate only)
python scripts/sanity_local_16gb_w768.py
python scripts/microbench_local_16gb_w768.py --steps 20
```

Do not start a 1M/5M/10M token campaign from this doc alone.

## Memory labeling

Startup / accounting numbers are **analytical estimates** (model + grad + AdamW
state + attention-score sketch). Process RSS from the sanity scripts is
lightweight process telemetry — **not** measured peak unified memory.
