"""One-step smoke forward for every variant config (tiny data, real flags)."""

import torch

from llmarcheval.config import list_variants, load_model_config, variant_config_path
from llmarcheval.models.transformer import GPT


def test_each_variant_smoke_forward():
    for variant in list_variants():
        cfg = load_model_config(variant_config_path(variant), "smoke")
        # Shrink embedding-dominated models for a fast CPU unit test.
        cfg.vocab_size = 128
        cfg.n_inner = None
        cfg.moe_intermediate_size = None
        cfg.qk_nope_head_dim = None
        cfg.v_head_dim = None
        cfg.n_embd = 32
        cfg.n_head = 4
        cfg.block_size = 16
        cfg.__post_init__()
        model = GPT(cfg)
        idx = torch.randint(0, cfg.vocab_size, (2, 8))
        out = model(idx, idx, loops=2 if cfg.use_recurrent else None)
        assert out.logits.shape == (2, 8, 128), variant
        assert torch.isfinite(out.loss), variant
        if cfg.use_moe:
            assert out.stats["expert_tokens"].sum() > 0, variant
        if cfg.use_mla:
            assert out.stats["kv_latent_rank"] == cfg.kv_lora_rank, variant
        if cfg.use_dsa:
            assert "dsa_keep" in out.stats, variant
        if cfg.use_recurrent:
            assert out.stats["loops"] == 2, variant
