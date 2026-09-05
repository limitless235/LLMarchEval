# LLMarchEval — Frontier-SLM lab

A from-scratch PyTorch laboratory for testing whether architectural ideas from
GLM-5.3, Kimi K2.5, Fable 5, and GPT-6 Astra still do something at SLM scale.

The question this repo answers:

**Which mechanisms survive downscaling — and what measurable side-effects do they have?**

This is not a miniature GLM or Kimi, not a claim about Fable/Astra internals,
and not the article. Closed-model cells stay labeled in
[docs/architecture-matrix.md](docs/architecture-matrix.md).

## Variants

| ID | Mechanism | Inspired by (epistemic) |
| --- | --- | --- |
| V0 | Dense decoder Transformer + SwiGLU + RoPE | Baseline |
| V1 | Sparse MoE (8 experts, top-2, 1 shared) | GLM / Kimi — **known** |
| V2 | V1 + simplified MLA (compressed KV) | GLM / Kimi — **known** |
| V3 | V2 + DSA indexer (top-k + local window) | GLM — **known** |
| V4 | V3 + prelude / looped block / coda | Astra recurrent depth — **inferred**, not in the system card |

Fable's **known** public idea used here is an inference **effort dial**
(`low/medium/high/...` → recurrent loop count), not a guessed layer stack.

Fair comparison: match **active** parameters and training tokens, not total params.
Full-scale V0–V4 are kept within ±2% active parameters (V0/V1 FFN width is
shrunk slightly so MLA’s KV advantage is not erased by widening MLA).

## Accounting caveats

- `flops_per_token` is an analytical proxy (`2 * active_params * depth_scale`),
  not measured FLOPs. It ignores quadratic attention, DSA indexer cost, and
  MoE routing. MLA still runs dense QKᵀ in the current forward.
- `kv_bytes_per_token_*` are theoretical decode-cache layouts. Training does
  not implement a persistent KV cache.
- DSA here is selection + masking on dense scores — not a sparse GEMM kernel.

## Scope of this pass

Implemented and validated on tiny data (TinyStories or the bundled corpus).

Designed for a later single-GPU run: ~100–150M **active** params, FineWeb-Edu,
context 1024, 2–5B tokens. **Not run here.** Use the cost estimator first.

Out of scope: the article, vision, agent swarms, MuonClip, GRPO, 1M context,
tokenizer training, cyber-exploit evals.

## Setup

```bash
pip install -e ".[dev]"
# optional: TinyStories / FineWeb-Edu loaders
pip install -e ".[data]"
```

## Tests

```bash
pytest
```

## Smoke train (tiny)

Each variant should drop loss on the bundled corpus (TinyStories if installed):

```bash
python scripts/train.py --train-config configs/smoke.yaml --variant v0_dense
python scripts/train.py --train-config configs/smoke.yaml --all-variants
```

Eval probes (needle, routing entropy, loop consistency, effort dial):

```bash
python scripts/eval.py --variant v3_moe_mla_dsa --train-config configs/smoke.yaml
```

## Research experiment harness

Structured, reproducible architecture probes (MoE routing, MLA context, DSA
needle selection, recurrent loops). Methodology and limitations:
[`docs/research_experiments.md`](docs/research_experiments.md).

```bash
python scripts/run_experiment.py moe-routing --device cpu
python scripts/run_experiment.py mla-context --device cpu --contexts 256 512
python scripts/run_experiment.py dsa-needle --device cpu --haystack-lens 64 128
python scripts/run_experiment.py recurrent-loops --device cpu --loops 1 2 3 4
```

Outputs: `results/experiments/*.json` + `results/experiments/results.jsonl`.
Do not launch long training from these commands.


## Local 16GB research profile (Mac / unified memory)

Config: [`configs/local_16gb.yaml`](configs/local_16gb.yaml). Details:
[`docs/local_16gb.md`](docs/local_16gb.md).

- ~37M active params, context 512, batch 1, grad accum 8, **FP32**
- MPS if available (else CPU with an explicit log); **no silent dataset fallback**
- Same tokenizer/dataset/optim/LR/budget across V0–V4 for mechanism comparison
- Apple unified memory is shared with the OS — estimates ≠ measurements

```bash
# Short offline validation (not a research run):
python scripts/validate_local_16gb.py

# Short research-shaped run (needs TinyStories):
python scripts/train.py --train-config configs/local_16gb.yaml --variant v0_dense --max-iters 50
```


## Cost estimate (do this before any 150M pretrain)

Runs a few synthetic steps of the **full** shapes (or `--scale smoke` on CPU)
and writes projected device-hours for 2B / 3B / 5B tokens:

```bash
python scripts/estimate_cost.py --train-config configs/pilot.yaml --out results/pilot_cost.json
# CPU-friendly:
python scripts/estimate_cost.py --scale smoke --steps 2 --out results/pilot_cost_smoke.json
```

`configs/full_single_gpu.yaml` is the documented 3B-token recipe. Do not launch
it until the estimate looks acceptable.

## Knobs

All of these are YAML: `n_layer`, `n_embd`, `n_head`, `n_experts`, `n_active`,
`block_size`, `max_iters`, `tokens_budget`, MLA ranks, DSA `index_topk`,
recurrent `train_loops` / `eval_loops`.

Model files live in `configs/models/`. Scale blocks are `smoke`, `local_16gb`, and `full`.

## Architecture probes (not a security product)

- MoE: expert utilization and router entropy on clean vs repetitive text
- DSA: fraction of needle tokens the indexer drops
- Recurrent: output agreement vs loop count; effort dial on the same weights

No tool-calling agent, no sandbox escape, no exploit tasks.
