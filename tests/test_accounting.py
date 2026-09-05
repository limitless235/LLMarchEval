from llmarcheval.config import load_model_config, variant_config_path
from llmarcheval.models.accounting import estimate_from_config, kv_cache_dims_per_token


def test_smoke_active_params_in_range():
    v0 = load_model_config(variant_config_path("v0_dense"), "smoke")
    est = estimate_from_config(v0)
    # GPT-2 vocab dominates; smoke dense should be tens of millions, not 150M.
    assert 5_000_000 < est["total_params"] < 40_000_000


def test_full_dense_near_gpt2_small():
    v0 = load_model_config(variant_config_path("v0_dense"), "full")
    est = estimate_from_config(v0)
    assert 80_000_000 < est["active_params"] < 200_000_000


def test_moe_full_active_close_to_dense():
    dense = estimate_from_config(load_model_config(variant_config_path("v0_dense"), "full"))
    moe = estimate_from_config(load_model_config(variant_config_path("v1_moe"), "full"))
    ratio = moe["active_params"] / dense["active_params"]
    assert 0.7 < ratio < 1.4
    assert moe["total_params"] > dense["total_params"]


def test_mla_reduces_kv():
    mha = load_model_config(variant_config_path("v1_moe"), "full")
    mla = load_model_config(variant_config_path("v2_moe_mla"), "full")
    assert kv_cache_dims_per_token(mla) < kv_cache_dims_per_token(mha)
