import torch

from llmarcheval.config import ModelConfig
from llmarcheval.models.accounting import estimate_from_config, kv_cache_dims_per_token
from llmarcheval.models.transformer import GPT


def _mla_cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=128,
        n_layer=2,
        n_embd=32,
        n_head=4,
        block_size=16,
        use_mla=True,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_mla_forward_and_latent_rank():
    cfg = _mla_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(idx)
    assert out.logits.shape[-1] == cfg.vocab_size
    assert out.stats["kv_latent_rank"] == cfg.kv_lora_rank


def test_mla_kv_smaller_than_mha():
    mla = _mla_cfg()
    mha = ModelConfig(vocab_size=128, n_layer=2, n_embd=32, n_head=4, block_size=16, use_mla=False)
    assert kv_cache_dims_per_token(mla) < kv_cache_dims_per_token(mha)


def test_mla_param_estimate():
    cfg = _mla_cfg()
    model = GPT(cfg)
    est = estimate_from_config(cfg)
    measured = sum(p.numel() for p in model.parameters())
    assert measured == est["total_params"]
