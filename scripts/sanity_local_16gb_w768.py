#!/usr/bin/env python3
"""Opt-in feasibility smoke for local_16gb_w768 — NOT a research experiment.

By default runs one forward / backward / AdamW step at B=1, T=512, FP32 for
ALL V0–V4 variants. Prefer MPS when available; falls back to CPU with a log.

Process RSS via resource.getrusage is reported when available. That is
lightweight process telemetry — not Apple unified-memory peak and not a
substitute for the analytical memory formulas in accounting.py.

Usage:
  python scripts/sanity_local_16gb_w768.py
  python scripts/sanity_local_16gb_w768.py --variant v4_recurrent
  python scripts/sanity_local_16gb_w768.py --all-variants   # same as default
"""

from __future__ import annotations

import argparse
import math
import resource
import sys
import time

import torch

from llmarcheval.config import CONFIGS_DIR, list_variants, load_experiment, variant_config_path
from llmarcheval.models.accounting import summarize_model
from llmarcheval.models.transformer import GPT
from llmarcheval.train.trainer import auto_device, auto_dtype


def resolve_variants(variant: str | None, all_variants: bool) -> list[str]:
    """Select variants for the feasibility gate.

    Default (no --variant): all five V0–V4. A single --variant narrows the run.
    --all-variants is an explicit alias for the default.
    """
    if variant is not None and all_variants:
        raise SystemExit("Pass either --variant NAME or --all-variants, not both")
    if variant is not None:
        known = list_variants()
        if variant not in known:
            raise SystemExit(f"Unknown variant {variant!r}; expected one of {known}")
        return [variant]
    return list_variants()


def _process_rss_bytes() -> int | None:
    """Best-effort process RSS; not peak unified memory."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # ru_maxrss is KiB on Linux, bytes on macOS.
        rss = int(usage.ru_maxrss)
        if rss <= 0:
            return None
        if sys.platform == "darwin":
            return rss
        return rss * 1024
    except Exception:
        return None


def _one_step(variant: str, device: torch.device, dtype: torch.dtype) -> dict:
    exp = load_experiment(
        variant_config_path(variant),
        CONFIGS_DIR / "local_16gb_w768.yaml",
    )
    assert exp.train.scale == "local_16gb_w768"
    assert exp.model.n_embd == 768
    assert exp.model.block_size == 512
    assert exp.train.batch_size == 1
    assert dtype == torch.float32

    model = GPT(exp.model).to(device=device, dtype=dtype)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=exp.train.lr, weight_decay=exp.train.weight_decay)
    accounting = summarize_model(model, exp.model, batch_size=1)

    t = exp.model.block_size
    idx = torch.randint(0, exp.model.vocab_size, (1, t), device=device)
    targets = torch.randint(0, exp.model.vocab_size, (1, t), device=device)

    rss_before = _process_rss_bytes()
    t0 = time.perf_counter()
    opt.zero_grad(set_to_none=True)
    out = model(idx, targets=targets)
    loss = out.loss
    assert loss is not None
    loss.backward()
    grad_norm_sq = 0.0
    for p in model.parameters():
        if p.grad is not None:
            g = p.grad.detach()
            assert torch.isfinite(g).all(), "non-finite gradient"
            grad_norm_sq += float(g.float().pow(2).sum().item())
    opt.step()
    elapsed = time.perf_counter() - t0
    rss_after = _process_rss_bytes()

    loss_f = float(loss.detach().float().item())
    assert math.isfinite(loss_f), loss_f
    assert math.isfinite(grad_norm_sq) and grad_norm_sq > 0.0

    return {
        "variant": variant,
        "device": str(device),
        "dtype": "float32",
        "batch_size": 1,
        "block_size": t,
        "loss": loss_f,
        "grad_norm": math.sqrt(grad_norm_sq),
        "elapsed_s": elapsed,
        "total_params": accounting["measured_total_params"],
        "active_params": accounting["active_params"],
        "analytical_model_grad_optimizer_bytes": accounting["memory_estimate"][
            "estimated_model_grad_optimizer_bytes"
        ],
        "process_rss_bytes_before": rss_before,
        "process_rss_bytes_after": rss_after,
        "process_rss_is_not_peak_unified_memory": True,
        "label": "feasibility_smoke_not_scientific_evidence",
    }


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="local_16gb_w768 MPS/CPU feasibility smoke (not a training campaign)"
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Run a single variant (default: all five V0–V4)",
    )
    parser.add_argument(
        "--all-variants",
        action="store_true",
        help="Run all V0–V4 (default when --variant is omitted)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="mps",
        help="Preferred device string (mps falls back to CPU if unavailable)",
    )
    args = parser.parse_args(argv)

    variants = resolve_variants(args.variant, args.all_variants)
    device = auto_device(args.device)
    dtype = auto_dtype("float32", device)
    print(
        f"local_16gb_w768 feasibility smoke: device={device} dtype={dtype} "
        f"variants={variants} "
        f"(FEASIBILITY GATE ONLY — not scientific evidence)",
        flush=True,
    )

    failed: str | None = None
    try:
        for variant in variants:
            row = _one_step(variant, device, dtype)
            print(
                f"ok {row['variant']}: loss={row['loss']:.4f} "
                f"grad_norm={row['grad_norm']:.4f} elapsed_s={row['elapsed_s']:.3f} "
                f"total={row['total_params']} active={row['active_params']} "
                f"analytical_mgo_bytes={row['analytical_model_grad_optimizer_bytes']} "
                f"process_rss_after={row['process_rss_bytes_after']}",
                flush=True,
            )
    except Exception as exc:
        failed = f"{type(exc).__name__}: {exc}"
        print(f"FAIL local_16gb_w768 feasibility smoke: {failed}", flush=True)
        raise

    print(
        f"PASS local_16gb_w768 feasibility smoke ({len(variants)} variant(s))",
        flush=True,
    )


if __name__ == "__main__":
    main()
