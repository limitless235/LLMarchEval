"""Generic --ckpt loading across research experiment harnesses."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import torch

from llmarcheval.cli.experiment import main
from llmarcheval.experiments.common import build_model, load_probe_experiment, resolve_device, resolve_dtype, set_seed
from llmarcheval.experiments.dsa_needle import run_dsa_needle_experiment
from llmarcheval.experiments.mla_context import run_mla_context_experiment
from llmarcheval.experiments.recurrent_loops import run_recurrent_loops_experiment


def _ckpt_for(variant: str, tmp_path: Path, fill: float = 0.5) -> Path:
    set_seed(0)
    exp = load_probe_experiment(variant, train_config="configs/smoke.yaml", scale="smoke")
    device = resolve_device("cpu")
    model = build_model(exp.model, device, resolve_dtype("float32", device))
    with torch.no_grad():
        next(model.parameters()).fill_(fill)
    path = tmp_path / f"{variant}_ckpt.pt"
    torch.save({"model": model.state_dict(), "config": exp.to_dict(), "step": 42}, path)
    return path


def test_cli_ckpt_flags_for_all_probes():
    with patch("llmarcheval.cli.experiment.run_moe_routing_experiment", return_value={}) as moe:
        with patch("builtins.print"):
            main(["moe-routing", "--ckpt", "x.pt", "--no-write"])
        assert moe.call_args.kwargs["ckpt"] == "x.pt"
    with patch("llmarcheval.cli.experiment.run_mla_context_experiment", return_value={}) as mla:
        with patch("builtins.print"):
            main(["mla-context", "--ckpt", "v2_moe_mla=y.pt", "--no-write"])
        assert mla.call_args.kwargs["ckpts"] == {"v2_moe_mla": "y.pt"}
    with patch("llmarcheval.cli.experiment.run_dsa_needle_experiment", return_value={}) as dsa:
        with patch("builtins.print"):
            main(["dsa-needle", "--ckpt", "z.pt", "--no-write"])
        assert dsa.call_args.kwargs["ckpt"] == "z.pt"
    with patch("llmarcheval.cli.experiment.run_recurrent_loops_experiment", return_value={}) as rec:
        with patch("builtins.print"):
            main(["recurrent-loops", "--ckpt", "w.pt", "--no-write"])
        assert rec.call_args.kwargs["ckpt"] == "w.pt"


def test_dsa_and_recurrent_load_ckpt(tmp_path: Path):
    dsa_ckpt = _ckpt_for("v3_moe_mla_dsa", tmp_path)
    rec = run_dsa_needle_experiment(
        write=False, device_name="cpu", haystack_lens=(64,), answer_tokens=2, ckpt=dsa_ckpt
    )
    assert rec["metrics"]["checkpoint"]["step"] == 42
    assert rec["training_steps"] == 42

    rec_ckpt = _ckpt_for("v4_recurrent", tmp_path)
    out = run_recurrent_loops_experiment(
        write=False, device_name="cpu", loops=(1, 2), ckpt=rec_ckpt
    )
    assert out["metrics"]["checkpoint"]["step"] == 42
    assert out["training_steps"] == 42


def test_mla_accepts_ckpts_mapping(tmp_path: Path):
    ckpt = _ckpt_for("v2_moe_mla", tmp_path)
    record = run_mla_context_experiment(
        variants=["v2_moe_mla"],
        contexts=[64],
        steps=1,
        write=False,
        device_name="cpu",
        ckpts={"v2_moe_mla": ckpt},
    )
    assert "mla" in record.get("experiment", "")
