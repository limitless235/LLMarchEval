# Architecture matrix (known / inferred / unknown)

This file records what we actually know about the frontier models that
Frontier-SLM is sampling ideas from. It is a research artifact for the lab,
not an article.

Labels:

- **Known** — weights, papers, or official developer docs.
- **Inferred** — strongly suggested by behavior, reporting, or system cards,
  but not a confirmed internal architecture.
- **Unknown** — do not treat as fact; do not implement as “that model’s architecture.”

GLM-5.3 Flash (hybrid linear attention, mHC, vision) is out of scope.
The flagship open-weight comparison is the GLM-5.3 MoE + MLA + DSA text model
(same base as GLM-5.2; 5.3 gains are disclosed as post-training).

| Idea | GLM-5.3 | Kimi K2.5 | Fable 5 | GPT-6 Astra | Lab variant |
| --- | --- | --- | --- | --- | --- |
| Sparse MoE | Known. Same base as 5.2: ~744–753B total, ~40B active, 78 layers (first 3 dense), 256 routed experts, top-8 + 1 shared. | Known. 1T total / 32B active, 61 layers (1 dense), 384 experts, top-8 + 1 shared, SwiGLU. | Unknown | Unknown | V1 |
| MLA / KV compression | Known | Known | Unknown | Unknown | V2 |
| Dynamic sparse attention | Known (MLA + DSA indexer) | Not the GLM DSA recipe | Unknown | Unknown | V3 |
| Long context | Known (1M config) | Known (256K) | Known (1M API) | Known capability, unknown internals | Eval only; not 1M training |
| Test-time / adaptive compute | Known (always-on reasoning, effort) | Known (Thinking / Instant) | Known (adaptive thinking + effort dial) | Known (reasoning effort) | Inference knobs |
| Recurrent depth | Unknown | Unknown | Unknown | **Inferred** from press reporting. **Not** stated in the [Astra system card](https://deploymentsafety.openai.com/gpt-6-astra). | V4, labeled inferred |
| Post-training dominates architecture | Known (5.3 = 5.2 base + post-train) | Known (continual train + agent RL) | Inferred (wrapper + adaptive reasoning) | Known (RL reasoning) | Later SFT; not this pass |
| Safety routing / monitoring | Partial | External evals | Known (route a slice of traffic to Opus) | Known (CoT monitors, impossible-task / honeypot evals) | Light architecture probes only |

## What the lab implements

Mechanisms, not clones:

- MoE, MLA, and DSA are **simplified** stand-ins for ideas documented in GLM/Kimi.
- Recurrent depth tests the **Astra hypothesis** (looped block, extra test-time
  FLOPs without extra unique parameters). It is not Astra’s architecture.
- Fable’s public, known idea used here is an **effort dial** (more loops /
  more think tokens on the same weights), not a guessed layer stack.

## What this lab does not claim

- We do not know Fable or Astra internals (MoE, MLA, DSA, layer counts).
- We do not reproduce GLM-5.3 or Kimi K2.5.
- Architecture probes (routing entropy, indexer drop-rate, loop-count
  consistency) are measurements, not a security product and not exploit work.
