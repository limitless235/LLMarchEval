from llmarcheval.config import list_variants, load_model_config, variant_config_path
from llmarcheval.models.accounting import estimate_from_config, kv_cache_dims_per_token
from llmarcheval.models.transformer import GPT


def test_smoke_active_params_in_range():
    v0 = load_model_config(variant_config_path("v0_dense"), "smoke")
    est = estimate_from_config(v0)
    # GPT-2 vocab dominates; smoke dense should be tens of millions, not 150M.
    assert 5_000_000 < est["total_params"] < 40_000_000


def test_full_dense_near_gpt2_small():
    v0 = load_model_config(variant_config_path("v0_dense"), "full")
    est = estimate_from_config(v0)
    assert 80_000_000 < est["active_params"] < 200_000_000


def test_full_variants_active_params_within_2_percent():
    """Fairness rule: match active params across V0–V4 within ±2%."""
    estimates = {
        variant: estimate_from_config(load_model_config(variant_config_path(variant), "full"))
        for variant in list_variants()
    }
    actives = {v: e["active_params"] for v, e in estimates.items()}
    lo = min(actives.values())
    hi = max(actives.values())
    assert (hi - lo) / lo <= 0.02, actives
    assert estimates["v1_moe"]["total_params"] > estimates["v0_dense"]["total_params"]
    for est in estimates.values():
        assert est["flops_per_token_is_proxy"] is True
        assert est["kv_bytes_are_theoretical"] is True


def test_accounting_matches_measured_params():
    for variant in list_variants():
        cfg = load_model_config(variant_config_path(variant), "smoke")
        cfg.vocab_size = 128
        # Force recomputation of derived widths after vocab change only.
        model = GPT(cfg)
        est = estimate_from_config(cfg)
        measured = sum(p.numel() for p in model.parameters())
        assert measured == est["total_params"], (variant, measured, est["total_params"])


def test_mla_reduces_kv():
    mha = load_model_config(variant_config_path("v1_moe"), "full")
    mla = load_model_config(variant_config_path("v2_moe_mla"), "full")
    assert kv_cache_dims_per_token(mla) < kv_cache_dims_per_token(mha)


def test_dsa_flag_marks_mask_only():
    v2 = estimate_from_config(load_model_config(variant_config_path("v2_moe_mla"), "full"))
    v3 = estimate_from_config(load_model_config(variant_config_path("v3_moe_mla_dsa"), "full"))
    assert v2["dsa_is_mask_only"] is False
    assert v3["dsa_is_mask_only"] is True
