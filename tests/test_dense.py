import torch

from llmarcheval.config import ModelConfig
from llmarcheval.models.accounting import estimate_from_config, total_parameters
from llmarcheval.models.transformer import GPT


def _tiny(**kwargs) -> ModelConfig:
    base = dict(vocab_size=128, n_layer=2, n_embd=32, n_head=4, block_size=16, dropout=0.0)
    base.update(kwargs)
    return ModelConfig(**base)


def test_forward_shapes_and_loss():
    cfg = _tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (2, 8))
    targets = torch.randint(0, cfg.vocab_size, (2, 8))
    out = model(idx, targets)
    assert out.logits.shape == (2, 8, cfg.vocab_size)
    assert out.loss is not None and torch.isfinite(out.loss)


def test_causal_mask_ignores_future():
    cfg = _tiny()
    model = GPT(cfg)
    model.eval()
    x = torch.randint(0, cfg.vocab_size, (1, 8))
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % cfg.vocab_size
    with torch.no_grad():
        a = model(x).logits
        b = model(y).logits
    assert torch.allclose(a[0, :-1], b[0, :-1], atol=1e-5)
    assert not torch.allclose(a[0, -1], b[0, -1], atol=1e-5)


def test_param_estimate_matches_measured():
    cfg = _tiny()
    model = GPT(cfg)
    est = estimate_from_config(cfg)
    assert est["measured_total_params"] if False else total_parameters(model) == est["total_params"]


def test_generate_grows_sequence():
    cfg = _tiny()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 4))
    out = model.generate(idx, max_new_tokens=3, temperature=0.0)
    assert out.shape == (1, 7)
