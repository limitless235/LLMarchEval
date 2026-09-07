import torch

from llmarcheval.config import ModelConfig
from llmarcheval.models.accounting import estimate_from_config, total_parameters
from llmarcheval.models.moe import MixtureOfExperts
from llmarcheval.models.transformer import GPT


def _moe_cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=128,
        n_layer=2,
        n_embd=32,
        n_head=4,
        block_size=16,
        use_moe=True,
        n_experts=4,
        n_active=2,
        n_shared_experts=1,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_moe_forward_and_aux_loss():
    cfg = _moe_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(idx, idx)
    assert out.aux_loss is not None and out.aux_loss.ndim == 0
    assert "expert_tokens" in out.stats
    assert out.stats["expert_tokens"].numel() == cfg.n_experts
    assert out.stats["expert_tokens"].sum() > 0


def test_router_topk():
    cfg = _moe_cfg()
    moe = MixtureOfExperts(cfg)
    x = torch.randn(2, 8, cfg.n_embd)
    _, aux, stats = moe(x)
    assert stats["router_indices"].shape[-1] == cfg.n_active
    assert aux.ndim == 0


def test_moe_active_params_less_than_total():
    cfg = _moe_cfg()
    est = estimate_from_config(cfg)
    model = GPT(cfg)
    assert total_parameters(model) == est["total_params"]
    assert est["active_params"] < est["total_params"]


def test_moe_gradients_flow():
    cfg = _moe_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(idx, idx)
    (out.loss + 0.01 * out.aux_loss).backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and torch.isfinite(g).all() and g.abs().sum() > 0 for g in grads)
