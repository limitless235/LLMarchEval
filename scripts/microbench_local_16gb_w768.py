#!/usr/bin/env python3
"""Opt-in 20-step feasibility microbenchmark for local_16gb_w768.

FEASIBILITY SMOKE DATA ONLY — not scientific evidence, not a research campaign.
Does not run automatically; invoke explicitly.

Reports per variant: tokens/sec, elapsed, loss, device, dtype, param counts.

Usage:
  python scripts/microbench_local_16gb_w768.py
  python scripts/microbench_local_16gb_w768.py --steps 20 --all-variants
"""

from __future__ import annotations

import argparse
import math
import resource
import time

import torch

from llmarcheval.config import CONFIGS_DIR, list_variants, load_experiment, variant_config_path
from llmarcheval.models.accounting import summarize_model
from llmarcheval.models.transformer import GPT
from llmarcheval.train.trainer import auto_device, auto_dtype


def _process_rss_bytes() -> int | None:
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        rss = int(usage.ru_maxrss)
        if rss <= 0:
            return None
        import sys

        if sys.platform == "darwin":
            return rss
        return rss * 1024
    except Exception:
        return None


def _run_variant(variant: str, steps: int, device: torch.device, dtype: torch.dtype) -> dict:
    exp = load_experiment(
        variant_config_path(variant),
        CONFIGS_DIR / "local_16gb_w768.yaml",
    )
    model = GPT(exp.model).to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)
    accounting = summarize_model(model, exp.model, batch_size=1)
    t = exp.model.block_size
    tokens_per_step = exp.train.batch_size * t  # microbench: one micro-batch per step

    last_loss = float("nan")
    t0 = time.perf_counter()
    for _ in range(steps):
        idx = torch.randint(0, exp.model.vocab_size, (1, t), device=device)
        targets = torch.randint(0, exp.model.vocab_size, (1, t), device=device)
        opt.zero_grad(set_to_none=True)
        out = model(idx, targets=targets)
        loss = out.loss
        assert loss is not None
        loss.backward()
        opt.step()
        last_loss = float(loss.detach().float().item())
        assert math.isfinite(last_loss)
    if device.type == "mps":
        torch.mps.synchronize()
    elapsed = time.perf_counter() - t0
    tokens = steps * tokens_per_step
    return {
        "variant": variant,
        "steps": steps,
        "tokens": tokens,
        "elapsed_s": elapsed,
        "tokens_per_sec": tokens / elapsed if elapsed > 0 else 0.0,
        "loss": last_loss,
        "device": str(device),
        "dtype": "float32",
        "total_params": accounting["measured_total_params"],
        "active_params": accounting["active_params"],
        "process_rss_bytes": _process_rss_bytes(),
        "label": "feasibility_smoke_data_not_scientific_evidence",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="local_16gb_w768 20-step feasibility microbench (opt-in; not science)"
    )
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--variant", type=str, default=None)
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--device", type=str, default="mps")
    args = parser.parse_args(argv)

    if args.variant is not None:
        variants = [args.variant]
    else:
        variants = list_variants()

    device = auto_device(args.device)
    dtype = auto_dtype("float32", device)
    print(
        "=== local_16gb_w768 feasibility microbench "
        "(NOT scientific evidence; NOT a training campaign) ===",
        flush=True,
    )
    print(f"device={device} dtype={dtype} steps={args.steps}", flush=True)

    for variant in variants:
        row = _run_variant(variant, args.steps, device, dtype)
        print(
            f"{row['variant']}: tok/s={row['tokens_per_sec']:.1f} "
            f"elapsed_s={row['elapsed_s']:.3f} loss={row['loss']:.4f} "
            f"device={row['device']} dtype={row['dtype']} "
            f"total={row['total_params']} active={row['active_params']} "
            f"process_rss_bytes={row['process_rss_bytes']}",
            flush=True,
        )

    print("=== end feasibility microbench ===", flush=True)


if __name__ == "__main__":
    main()
