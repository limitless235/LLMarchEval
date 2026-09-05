import torch

from llmarcheval.config import ModelConfig
from llmarcheval.eval.needle import needle_eval
from llmarcheval.models.attention import causal_mask
from llmarcheval.models.dsa import DSAIndexer
from llmarcheval.models.transformer import GPT


def _dsa_cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=128,
        n_layer=2,
        n_embd=32,
        n_head=4,
        block_size=32,
        use_mla=True,
        use_dsa=True,
        kv_lora_rank=8,
        qk_rope_head_dim=8,
        index_n_heads=2,
        index_head_dim=8,
        index_topk=4,
        dsa_local_window=2,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_dsa_keep_is_causal_and_sparse():
    cfg = _dsa_cfg()
    indexer = DSAIndexer(cfg)
    seq_len = 16
    x = torch.randn(2, seq_len, cfg.n_embd)
    causal = causal_mask(seq_len, x.device)
    keep, stats = indexer(x, causal)
    assert keep.shape == (2, seq_len, seq_len)
    future = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    assert not keep[0][future].any()
    assert stats["dsa_selected_mean"] <= seq_len
    # Sparse relative to dense causal (except tiny T).
    assert keep[0].float().mean() < 0.75


def test_dsa_model_forward_exposes_keep():
    cfg = _dsa_cfg()
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 16))
    out = model(idx)
    assert "dsa_keep" in out.stats
    assert out.stats["dsa_keep"].shape[-1] == 16


def test_needle_probe_runs():
    cfg = _dsa_cfg(vocab_size=50257, block_size=64)
    model = GPT(cfg)
    result = needle_eval(model, torch.device("cpu"), haystack_len=32)
    assert "dsa_needle_drop_frac" in result
    assert result["seq_len"] <= 64
