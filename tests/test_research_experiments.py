"""Unit tests for the research experiment harness (cheap CPU probes)."""

from __future__ import annotations

from pathlib import Path

import pytest

from llmarcheval.experiments.common import (
    REQUIRED_TOP_LEVEL_KEYS,
    SCHEMA_VERSION,
    assert_result_schema,
    write_result,
)
from llmarcheval.experiments.dsa_needle import run_dsa_needle_experiment
from llmarcheval.experiments.mla_context import DEFAULT_CONTEXTS, run_mla_context_experiment
from llmarcheval.experiments.moe_routing import TEXT_REGIMES, run_moe_routing_experiment
from llmarcheval.experiments.recurrent_loops import run_recurrent_loops_experiment


def test_output_schema_keys_and_writer(tmp_path: Path):
    record = run_moe_routing_experiment(
        seed=7,
        device_name="cpu",
        write=False,
        scale="smoke",
    )
    assert_result_schema(record)
    assert record["schema_version"] == SCHEMA_VERSION
    for key in REQUIRED_TOP_LEVEL_KEYS:
        assert key in record
    assert record["batch_size"] >= 1
    assert record["grad_accumulation"] >= 1
    assert "total" in record["parameter_counts"]
    assert record["active_parameter_counts"] == record["parameter_counts"]["active"]

    path = write_result(record, tmp_path, stem="schema_probe")
    assert path.exists()
    jsonl = tmp_path / "results.jsonl"
    assert jsonl.exists()
    assert jsonl.read_text(encoding="utf-8").strip()


def test_deterministic_seed_moe_routing():
    a = run_moe_routing_experiment(seed=1337, device_name="cpu", write=False)
    b = run_moe_routing_experiment(seed=1337, device_name="cpu", write=False)
    for regime in TEXT_REGIMES:
        assert a["metrics"]["regimes"][regime]["per_expert_token_counts"] == b["metrics"]["regimes"][
            regime
        ]["per_expert_token_counts"]
        assert a["metrics"]["regimes"][regime]["routing_entropy"] == pytest.approx(
            b["metrics"]["regimes"][regime]["routing_entropy"]
        )
        assert a["loss_metrics"]["per_regime_nll"][regime] == pytest.approx(
            b["loss_metrics"]["per_regime_nll"][regime]
        )


def test_moe_routing_metrics():
    record = run_moe_routing_experiment(seed=3, device_name="cpu", write=False)
    assert_result_schema(record)
    assert set(record["metrics"]["regimes"]) == set(TEXT_REGIMES)
    for regime, payload in record["metrics"]["regimes"].items():
        counts = payload["per_expert_token_counts"]
        util = payload["utilization_pct"]
        assert len(counts) == len(util)
        assert sum(counts) > 0
        assert abs(sum(util) - 100.0) < 1e-3
        assert payload["routing_entropy"] is not None
        assert payload["topk_assignments_sample"] is not None
        assert payload["topk_assignment_counts"] is not None
        assert regime in record["loss_metrics"]["per_regime_nll"]


def test_dsa_needle_selection_accounting():
    record = run_dsa_needle_experiment(
        seed=11,
        device_name="cpu",
        write=False,
        haystack_lens=(64,),
        needle_positions=("middle",),
        answer_tokens=4,
    )
    assert_result_schema(record)
    assert record["metrics"]["n_probes"] == 1
    probe = record["metrics"]["probes"][0]
    assert "needle_location" in probe
    assert probe["needle_location"]["span"] == probe["needle_span"]
    start, end = probe["needle_span"]
    assert end > start
    assert isinstance(probe["dsa_selected_token_indices"], list)
    assert probe["dsa_selected_count"] == len(probe["dsa_selected_token_indices"])
    assert probe["needle_span_selected"] is not None
    if probe["needle_span_selected"]:
        assert any(start <= i < end for i in probe["dsa_selected_token_indices"])
    assert "model_answer" in probe
    assert "retrieval_correct" in probe
    assert "sparse-attention" not in record["experiment"]
    assert any("dense" in n.lower() for n in record["notes"])


def test_recurrent_loop_sweep():
    loops = (1, 2, 3, 4)
    record = run_recurrent_loops_experiment(
        seed=5,
        device_name="cpu",
        write=False,
        loops=loops,
    )
    assert_result_schema(record)
    rows = record["metrics"]["rows"]
    assert [r["loops"] for r in rows] == list(loops)
    assert rows[0]["marginal_loss_vs_prev"] is None
    for prev, cur in zip(rows, rows[1:]):
        assert cur["marginal_loss_vs_prev"] == pytest.approx(cur["loss"] - prev["loss"])
        assert cur["marginal_flops_proxy_vs_prev"] is not None
        assert cur["logit_agreement_vs_prev"] is not None
        assert cur["flops_per_token_is_proxy"] is True
        assert cur["wall_clock_s"] > 0
        assert cur["tokens_per_sec"] > 0


def test_mla_context_sweep_configuration():
    assert tuple(DEFAULT_CONTEXTS) == (256, 512, 1024, 2048)
    contexts = (32, 64)
    record = run_mla_context_experiment(
        seed=9,
        device_name="cpu",
        write=False,
        contexts=contexts,
        variants=("v1_moe", "v2_moe_mla"),
        steps=1,
    )
    assert_result_schema(record)
    rows = record["metrics"]["rows"]
    assert len(rows) == 4
    assert record["metrics"]["contexts"] == list(contexts)
    by_key = {(r["variant"], r["context_length"]): r for r in rows}
    for variant in ("v1_moe", "v2_moe_mla"):
        for ctx in contexts:
            row = by_key[(variant, ctx)]
            assert row["context_length"] == ctx
            assert row["loss_finite"] is True
            assert row["analytical_kv_bytes_are_theoretical"] is True
            assert row["analytical_kv_bytes_fp32"] > 0
            assert row["use_mla"] is (variant == "v2_moe_mla")
            assert "peak_memory_bytes_measured" in row
    assert any("theoretical" in n.lower() or "decode" in n.lower() for n in record["notes"])
