import torch

from llmarcheval.config import ModelConfig
from llmarcheval.eval.probes import effort_dial_probe, loop_consistency_probe
from llmarcheval.models.accounting import estimate_from_config, total_parameters
from llmarcheval.models.transformer import GPT


def _rec_cfg(**kwargs) -> ModelConfig:
    base = dict(
        vocab_size=128,
        n_embd=32,
        n_head=4,
        block_size=16,
        use_recurrent=True,
        n_prelude=1,
        n_recurrent=1,
        n_coda=1,
        train_loops=2,
        eval_loops=2,
    )
    base.update(kwargs)
    return ModelConfig(**base)


def test_recurrent_unique_layers_not_multiplied_by_loops():
    cfg = _rec_cfg()
    model = GPT(cfg)
    assert cfg.unique_layers == 3
    assert len(model.prelude) + len(model.recurrent) + len(model.coda) == 3
    est1 = estimate_from_config(cfg, loops=1)
    est4 = estimate_from_config(cfg, loops=4)
    assert est1["total_params"] == est4["total_params"]
    assert est4["executed_layers"] > est1["executed_layers"]
    assert est4["flops_per_token"] > est1["flops_per_token"]
    assert total_parameters(model) == est1["total_params"]


def test_recurrent_weight_reuse():
    cfg = _rec_cfg()
    model = GPT(cfg)
    ids = [id(p) for p in model.recurrent.parameters()]
    assert ids
    # Running more loops does not allocate new recurrent parameters.
    n_before = total_parameters(model)
    _ = model(torch.randint(0, cfg.vocab_size, (1, 8)), loops=4)
    assert total_parameters(model) == n_before


def test_more_loops_changes_logits():
    cfg = _rec_cfg()
    model = GPT(cfg)
    model.eval()
    idx = torch.randint(0, cfg.vocab_size, (1, 8))
    with torch.no_grad():
        a = model(idx, loops=1).logits
        b = model(idx, loops=3).logits
    assert not torch.allclose(a, b, atol=1e-5)


def test_effort_dial_maps_loops():
    cfg = _rec_cfg()
    assert cfg.loops_for_effort("low") == 1
    assert cfg.loops_for_effort("high") == 4
    model = GPT(cfg)
    probe = effort_dial_probe(model, torch.device("cpu"))
    assert probe["low"]["loops"] == 1
    assert probe["high"]["loops"] == 4
    consistency = loop_consistency_probe(model, torch.device("cpu"), loop_counts=(1, 2))
    assert "pairwise" in consistency
