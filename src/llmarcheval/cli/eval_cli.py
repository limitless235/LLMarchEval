"""CLI: run eval probes on a checkpoint or a fresh init."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from llmarcheval.config import CONFIGS_DIR, load_experiment, variant_config_path
from llmarcheval.eval.needle import needle_eval
from llmarcheval.eval.perplexity import completion_probe
from llmarcheval.eval.probes import effort_dial_probe, loop_consistency_probe, routing_probe
from llmarcheval.models.transformer import GPT
from llmarcheval.train.trainer import auto_device


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Evaluate a Frontier-SLM variant")
    parser.add_argument("--train-config", type=Path, default=CONFIGS_DIR / "smoke.yaml")
    parser.add_argument("--variant", type=str, default="v0_dense")
    parser.add_argument("--ckpt", type=Path, default=None)
    parser.add_argument("--scale", type=str, default=None)
    args = parser.parse_args(argv)

    experiment = load_experiment(variant_config_path(args.variant), args.train_config, scale=args.scale)
    device = auto_device(experiment.train.device)
    model = GPT(experiment.model).to(device)
    if args.ckpt:
        blob = torch.load(args.ckpt, map_location=device)
        model.load_state_dict(blob["model"])

    payload = {
        "variant": experiment.model.variant,
        "completions": completion_probe(model, device),
        "needle": [
            needle_eval(model, device, haystack_len=length)
            for length in (64, 128, min(256, experiment.model.block_size))
        ],
        "routing": routing_probe(model, device),
        "loops": loop_consistency_probe(model, device),
        "effort_dial": effort_dial_probe(model, device),
    }
    out = Path(experiment.train.out_dir) / experiment.model.variant / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
