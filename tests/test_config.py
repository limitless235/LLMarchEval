from llmarcheval.config import (
    load_experiment,
    load_model_config,
    load_train_config,
    list_variants,
    variant_config_path,
)


def test_all_variants_present():
    assert list_variants() == [
        "v0_dense",
        "v1_moe",
        "v2_moe_mla",
        "v3_moe_mla_dsa",
        "v4_recurrent",
    ]


def test_smoke_and_full_scales_load():
    for variant in list_variants():
        smoke = load_model_config(variant_config_path(variant), "smoke")
        full = load_model_config(variant_config_path(variant), "full")
        assert smoke.n_embd == 256
        assert full.n_embd == 768
        assert smoke.block_size == 256
        assert full.block_size == 1024


def test_feature_flags():
    v0 = load_model_config(variant_config_path("v0_dense"), "smoke")
    assert not v0.use_moe and not v0.use_mla and not v0.use_dsa and not v0.use_recurrent
    v1 = load_model_config(variant_config_path("v1_moe"), "smoke")
    assert v1.use_moe and v1.n_experts == 8 and v1.n_active == 2 and v1.n_shared_experts == 1
    v2 = load_model_config(variant_config_path("v2_moe_mla"), "smoke")
    assert v2.use_moe and v2.use_mla
    v3 = load_model_config(variant_config_path("v3_moe_mla_dsa"), "smoke")
    assert v3.use_dsa and v3.use_mla
    v4 = load_model_config(variant_config_path("v4_recurrent"), "smoke")
    assert v4.use_recurrent
    assert v4.unique_layers == v4.n_prelude + v4.n_recurrent + v4.n_coda == 4


def test_load_experiment():
    exp = load_experiment(variant_config_path("v0_dense"), "configs/smoke.yaml")
    assert exp.train.dataset == "tinystories"
    assert exp.model.variant == "v0_dense"


def test_train_configs():
    smoke = load_train_config("configs/smoke.yaml")
    pilot = load_train_config("configs/pilot.yaml")
    full = load_train_config("configs/full_single_gpu.yaml")
    local = load_train_config("configs/local_16gb.yaml")
    assert smoke.scale == "smoke"
    assert pilot.scale == "full" and pilot.max_iters <= 50
    assert full.tokens_budget == 3_000_000_000
    assert local.scale == "local_16gb"
    assert local.allow_dataset_fallback is False
    assert local.dtype in {"float32", "fp32"}
