"""Parameter, FLOP, and KV-cache accounting.

Caveats (read before citing numbers in results or articles):

1. ``flops_per_token`` is an **analytical proxy**: ``2 * active_params * depth_scale``.
   It is not measured wall-clock FLOPs. It ignores quadratic attention cost, DSA
   indexer overhead, and MoE routing overhead. MLA variants report fewer active
   attention parameters, but the current forward still materializes K/V and runs
   dense QK^T, so real attention FLOPs are closer to MHA than the proxy suggests.

2. ``kv_bytes_per_token_*`` describe a **theoretical decode-time KV cache layout**
   (MHA: full K/V; MLA: latent + RoPE dims; DSA: optional indexer K). The training
   forward does **not** implement a persistent decode KV cache. Do not treat these
   bytes as measured peak memory.

3. DSA in this lab is **selection + masking** on dense attention scores. There is
   no sparse GEMM / production DSA kernel, so DSA does not currently provide
   kernel-level compute savings (and may be slower on CPU/GPU for that reason).

4. Optimizer / activation memory helpers below are **analytical estimates** for
   planning (e.g. local_16gb). They are not measured peak memory. On Apple
   unified memory, model + grads + optimizer + activations + OS share one pool.
"""

from __future__ import annotations

from typing import Any

import torch.nn as nn

from llmarcheval.config import ModelConfig


def total_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters())


def trainable_parameters(model: nn.Module) -> int:
    return sum(param.numel() for param in model.parameters() if param.requires_grad)


def _swiglu_params(n_embd: int, intermediate: int) -> int:
    return 3 * n_embd * intermediate


def _attn_params(config: ModelConfig) -> int:
    if config.use_mla:
        q_dim = config.n_head * (config.qk_nope_head_dim + config.qk_rope_head_dim)
        params = config.n_embd * q_dim
        params += config.n_embd * config.kv_lora_rank
        params += config.n_embd * config.n_head * config.qk_rope_head_dim
        params += config.kv_lora_rank * config.n_head * config.qk_nope_head_dim
        params += config.kv_lora_rank * config.n_head * config.v_head_dim
        params += config.n_head * config.v_head_dim * config.n_embd
        return params
    return 4 * config.n_embd * config.n_embd


def _ffn_total_params(config: ModelConfig) -> int:
    if not config.use_moe:
        assert config.n_inner is not None
        return _swiglu_params(config.n_embd, config.n_inner)
    assert config.moe_intermediate_size is not None
    expert = _swiglu_params(config.n_embd, config.moe_intermediate_size)
    routed = config.n_experts * expert
    shared = config.n_shared_experts * expert
    router = config.n_embd * config.n_experts
    return routed + shared + router


def _ffn_active_params(config: ModelConfig) -> int:
    if not config.use_moe:
        assert config.n_inner is not None
        return _swiglu_params(config.n_embd, config.n_inner)
    assert config.moe_intermediate_size is not None
    expert = _swiglu_params(config.n_embd, config.moe_intermediate_size)
    router = config.n_embd * config.n_experts
    return (config.n_shared_experts + config.n_active) * expert + router


def _layer_norm_params(config: ModelConfig) -> int:
    # Two pre-norm LayerNorms per block (weight + bias each).
    return 4 * config.n_embd


def _dsa_params(config: ModelConfig) -> int:
    if not config.use_dsa:
        return 0
    inner = config.index_n_heads * config.index_head_dim
    return 2 * config.n_embd * inner


def component_param_breakdown(config: ModelConfig) -> dict[str, int]:
    """Parameter counts by major component (closed-form, not measured)."""
    layers = config.unique_layers
    embed = config.vocab_size * config.n_embd
    lm_head = 0 if config.tie_weights else config.vocab_size * config.n_embd
    final_ln = 2 * config.n_embd
    attn = layers * _attn_params(config)
    ffn_total = layers * _ffn_total_params(config)
    ffn_active = layers * _ffn_active_params(config)
    layer_norms = layers * _layer_norm_params(config)
    dsa = layers * _dsa_params(config)
    return {
        "embedding": int(embed),
        "lm_head": int(lm_head),
        "attention": int(attn),
        "ffn_total": int(ffn_total),
        "ffn_active": int(ffn_active),
        "layer_norms": int(layer_norms + final_ln),
        "dsa_indexer": int(dsa),
    }


def estimate_training_memory(
    config: ModelConfig,
    *,
    batch_size: int = 1,
    bytes_per_param: int = 4,
) -> dict[str, Any]:
    """Analytical FP32 training memory sketch — not measured peak memory.

    AdamW stores two moment buffers (m, v) in FP32. Attention-score bytes cover
    one dense ``[B, H, T, T]`` tensor and ignore the rest of the activation graph.
    """
    est = estimate_from_config(config)
    total = est["total_params"]
    model_bytes = total * bytes_per_param
    grad_bytes = total * bytes_per_param
    adamw_state_bytes = total * 2 * bytes_per_param
    t = config.block_size
    attn_score_bytes = batch_size * config.n_head * t * t * bytes_per_param
    return {
        "bytes_per_param": bytes_per_param,
        "estimated_model_bytes": int(model_bytes),
        "estimated_grad_bytes": int(grad_bytes),
        "estimated_adamw_state_bytes": int(adamw_state_bytes),
        "estimated_optimizer_memory_bytes": int(adamw_state_bytes),
        "estimated_model_grad_optimizer_bytes": int(model_bytes + grad_bytes + adamw_state_bytes),
        "estimated_attention_score_bytes": int(attn_score_bytes),
        "memory_figures_are_estimates": True,
        "note": (
            "Estimates only. Unified memory is shared with OS/apps; "
            "do not treat these as measured peak memory."
        ),
    }


def estimate_from_config(config: ModelConfig, loops: int | None = None) -> dict[str, Any]:
    """Closed-form counts used for cost estimates and tests."""
    layers = config.unique_layers
    components = component_param_breakdown(config)
    total = (
        components["embedding"]
        + components["lm_head"]
        + components["attention"]
        + components["ffn_total"]
        + components["layer_norms"]
        + components["dsa_indexer"]
    )
    active = (
        components["embedding"]
        + components["lm_head"]
        + components["attention"]
        + components["ffn_active"]
        + components["layer_norms"]
        + components["dsa_indexer"]
    )

    loop_count = loops if loops is not None else config.eval_loops
    if config.use_recurrent:
        executed = config.n_prelude + config.n_recurrent * loop_count + config.n_coda
    else:
        executed = layers
    # Analytical proxy only — see module docstring.
    depth_scale = executed / max(layers, 1)
    flops_per_token = 2.0 * active * depth_scale

    kv_dims = kv_cache_dims_per_token(config)
    return {
        "total_params": int(total),
        "active_params": int(active),
        "components": components,
        "unique_layers": layers,
        "executed_layers": executed,
        "depth_scale": depth_scale,
        "flops_per_token": flops_per_token,
        "flops_per_token_is_proxy": True,
        "kv_dims_per_token": kv_dims,
        "kv_bytes_per_token_fp16": kv_dims * 2,
        "kv_bytes_per_token_fp32": kv_dims * 4,
        "kv_bytes_are_theoretical": True,
        "dsa_is_mask_only": bool(config.use_dsa),
        "dsa_index_topk": config.index_topk if config.use_dsa else None,
        "moe_experts": config.n_experts if config.use_moe else None,
        "moe_active": config.n_active if config.use_moe else None,
        "loops": loop_count if config.use_recurrent else 1,
    }


def kv_cache_dims_per_token(config: ModelConfig) -> int:
    """Theoretical cached dims/token across unique layers (loops do not add KV layers).

    Layout estimate for a future decode cache, not a measurement of the current
    training forward (which does not keep a persistent KV cache).
    """
    layers = config.unique_layers
    if config.use_mla:
        per_layer = config.kv_lora_rank + config.n_head * config.qk_rope_head_dim
    else:
        per_layer = 2 * config.n_head * config.head_dim
    extra_index = 0
    if config.use_dsa:
        extra_index = config.index_n_heads * config.index_head_dim  # indexer K
    return layers * (per_layer + extra_index)


def summarize_model(
    model: nn.Module,
    config: ModelConfig,
    loops: int | None = None,
    *,
    batch_size: int = 1,
) -> dict[str, Any]:
    estimate = estimate_from_config(config, loops=loops)
    estimate["measured_total_params"] = total_parameters(model)
    estimate["memory_estimate"] = estimate_training_memory(config, batch_size=batch_size)
    return estimate
