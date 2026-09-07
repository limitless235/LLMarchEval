# Research experiment harness

Offline, reproducible probes for four architecture investigations in this lab.
Each probe writes a structured JSON record (and appends JSONL) without requiring
network access unless a dataset is explicitly configured for a later training run.

This document describes **questions, methodology, metrics, and limitations**.
It does not draw conclusions.

## Shared output schema

Every run records (`schema_version` `1.0`):

| Field | Meaning |
| --- | --- |
| `git_commit` | `git rev-parse HEAD` at write time (or `unknown`) |
| `config` | Full model + train snapshot used for the probe |
| `variant` | Variant id(s) under test |
| `seed` | Fixed RNG seed |
| `parameter_counts` / `active_parameter_counts` | Total and active params |
| `context_length` | Model / probe context used |
| `batch_size` / `grad_accumulation` | From train config (probes are usually batch 1) |
| `dtype` / `device` | Resolved runtime types |
| `training_steps` / `training_tokens` | Null for pure probes; set if a short train is recorded |
| `wall_clock_s` / `tokens_per_sec` | Wall timing for the probe body |
| `loss_metrics` | Cross-entropy / NLL where measured |
| `metrics` | Experiment-specific payload |
| `accounting_proxy` | Analytical FLOP/KV proxies with explicit “theoretical” flags |
| `notes` | Caveats attached to the run |

Writers: `results/experiments/<stem>.json` and `results/experiments/results.jsonl`.

## Commands

```bash
# MoE routing regimes
python scripts/run_experiment.py moe-routing --device cpu --seed 1337

# MLA context sweep (V1 vs V2)
python scripts/run_experiment.py mla-context --device cpu --contexts 256 512 1024 2048

# DSA needle selection + retrieval accounting
python scripts/run_experiment.py dsa-needle --device cpu --haystack-lens 64 128 256

# Recurrent loop compute dial
python scripts/run_experiment.py recurrent-loops --device cpu --loops 1 2 3 4
```

Equivalent entry point: `llme-experiment <subcommand> ...`.

Default train config is `configs/smoke.yaml` (offline, cheap shapes). Use
`--train-config configs/local_16gb.yaml --scale local_16gb` for the validated
16GB research profile shapes. Do not launch long training from these probes.

---

## 1. MoE routing

**Research question.** How do expert assignments, utilization, and router entropy
vary across controlled text regimes (normal, repetitive, structured, reversed)
for MoE variants?

**Methodology.** Deterministic offline corpora; fixed seed; single forward per
regime on a fresh or eval-mode MoE model. No interpretation as security findings.

**Metrics.** Per-expert token counts, utilization %, routing entropy, top-k
assignment samples/histograms, per-regime NLL.

**Can establish.** Descriptive routing statistics under those corpora.

**Cannot establish.** Security properties, intentional backdoors, production MoE
load-balance quality at scale, or causal claims about GLM/Kimi internals.

---

## 2. MLA context scaling

**Research question.** How do forward time, memory proxies, and analytical KV
footprint change with context length for V1 (MHA+MoE) vs V2 (MLA+MoE)?

**Methodology.** Context lengths 256 / 512 / 1024 / 2048 (configurable). Rebuild
with `block_size = context_length` so longer windows are exercised. Short
synthetic forwards; no long train.

**Metrics.** Forward wall time, tokens/sec, loss finiteness, CUDA peak memory
when available, analytical KV bytes (theoretical decode layout), analytical
attention-score byte estimate.

**Known limitation (critical).** The current MLA training forward still
materializes dense K/V. Analytical KV figures are **not** measured decode-cache
sizes. Peak memory on CPU/MPS is typically unavailable in this harness.

**Can establish.** Relative wall-clock and analytical footprint trends under
this lab’s MLA implementation.

**Cannot establish.** Production MLA decode-cache savings, or that measured
activations equal theoretical KV layouts.

---

## 3. DSA needle selection / retrieval

**Research question.** For a controlled needle-in-context prompt, which token
indices does DSA select (via dense-score selection + masking), does the needle
span appear among selected keys for the final query position, and does a short
greedy continuation contain the passcode?

**Methodology.** Synthetic haystacks + fixed needle/query; no network. Records
needle location separately from selected indices and answer text.

**Metrics.** Needle location/span, DSA selected-token indices and count,
`needle_span_selected`, model answer, retrieval correctness.

**Naming.** This is **not** a sparse-attention kernel benchmark. DSA in this repo
selects and masks over dense attention scores.

**Can establish.** Accounting of indexer selection vs needle span on synthetic
prompts for this implementation.

**Cannot establish.** Production sparse attention throughput, long-context
retrieval SOTA, or GLM DSA equivalence.

---

## 4. Recurrent compute scaling

**Research question.** Treating loop count as an experimental compute dial, how
do loss, logit agreement, wall time, tokens/sec, and analytical FLOP proxy change
for V4 at loops = 1, 2, 3, 4?

**Methodology.** Fixed prompt; eval forward with explicit `loops=`; compare each
setting to the previous loop count (marginal loss / FLOP proxy / logit agreement).

**Metrics.** Per-loop NLL, wall time, tokens/sec, FLOP proxy, executed layers,
marginal deltas, cosine / argmax agreement vs previous loops.

**Epistemic note.** Recurrent depth here is a **hypothesis / compute dial**, not
evidence of proprietary Astra internals.

**Can establish.** Behavior of this lab’s looped-block implementation under a
fixed seed and prompt.

**Cannot establish.** Claims about closed-model internals, optimal effort dials,
or that extra loops improve quality after training (probes use untrained or
separately trained weights as provided).

---

## Reproducibility

- Fixed `--seed` (default 1337); `torch.manual_seed` (+ CUDA seed when present).
- Each experiment is an independent CLI subcommand.
- Offline by default; probes do not download datasets.
- Prefer `smoke` for unit tests and CI; use `local_16gb` only when intentionally
  measuring that profile’s shapes.

## What this harness deliberately excludes

- New model mechanisms or V0–V4 dimension changes
- Long training runs and GPU requirements
- Agent / tool / security product functionality
- Changes to the existing experimental methodology outside structured logging
- Conclusions in this document (reserved for later analysis)
