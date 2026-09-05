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


def test_train_loops_changes_executed_depth():
    """Changing train_loops must change executed depth in training mode."""
    cfg = _rec_cfg(train_loops=1, eval_loops=3, n_prelude=1, n_recurrent=1, n_coda=1)
    model = GPT(cfg)
    idx = torch.randint(0, cfg.vocab_size, (1, 8))

    model.train()
    # Explicit train_loops path (matches trainer).
    out_train = model(idx, loops=cfg.train_loops)
    assert out_train.stats["loops"] == 1
    assert out_train.stats["n_blocks_executed"] == 1 + 1 * 1 + 1

    # Default training resolution uses train_loops when loops=None.
    out_train_default = model(idx)
    assert out_train_default.stats["loops"] == cfg.train_loops

    model.eval()
    out_eval = model(idx)
    assert out_eval.stats["loops"] == cfg.eval_loops
    assert out_eval.stats["n_blocks_executed"] == 1 + 1 * 3 + 1
    assert out_eval.stats["n_blocks_executed"] > out_train.stats["n_blocks_executed"]

    # Accounting tracks the same depth change.
    est1 = estimate_from_config(cfg, loops=1)
    est3 = estimate_from_config(cfg, loops=3)
    assert est3["executed_layers"] > est1["executed_layers"]


def test_loops_for_effort_uses_train_vs_eval_defaults():
    cfg = _rec_cfg(train_loops=2, eval_loops=5)
    assert cfg.loops_for_effort(training=True) == 2
    assert cfg.loops_for_effort(training=False) == 5
    assert cfg.loops_for_effort(effort="high", training=True) == 4
    assert cfg.loops_for_effort(loops=7, training=True) == 7
