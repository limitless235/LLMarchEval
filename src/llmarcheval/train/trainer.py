"""Single-device causal LM trainer."""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

import torch

from llmarcheval.config import ExperimentConfig
from llmarcheval.models.accounting import summarize_model
from llmarcheval.models.transformer import GPT
from llmarcheval.train.data import TokenBatcher, load_tokens


def auto_device(name: str) -> torch.device:
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(name)


def auto_dtype(name: str, device: torch.device) -> torch.dtype:
    if name == "auto":
        if device.type == "cuda" and torch.cuda.is_bf16_supported():
            return torch.bfloat16
        return torch.float32
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype {name}")
    return mapping[name]


def cosine_lr(step: int, warmup: int, max_steps: int, lr: float, min_lr: float) -> float:
    if step < warmup:
        return lr * (step + 1) / max(warmup, 1)
    if step >= max_steps:
        return min_lr
    progress = (step - warmup) / max(max_steps - warmup, 1)
    coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: GPT, batcher: TokenBatcher, device: torch.device, eval_iters: int, dtype: torch.dtype
) -> float:
    model.eval()
    losses = []
    amp = dtype in {torch.float16, torch.bfloat16} and device.type == "cuda"
    for _ in range(eval_iters):
        x, y = batcher.get_batch(device)
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
            out = model(x, y)
        assert out.loss is not None
        losses.append(out.loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def train(experiment: ExperimentConfig) -> dict:
    cfg = experiment.train
    torch.manual_seed(cfg.seed)
    device = auto_device(cfg.device)
    dtype = auto_dtype(cfg.dtype, device)

    tokens, source = load_tokens(cfg)
    n = tokens.size
    split = max(int(n * 0.9), experiment.model.block_size + 2)
    train_tokens, val_tokens = tokens[:split], tokens[split:]
    if val_tokens.size <= experiment.model.block_size + 1:
        val_tokens = train_tokens
    train_batcher = TokenBatcher(train_tokens, experiment.model.block_size, cfg.batch_size, cfg.seed)
    val_batcher = TokenBatcher(val_tokens, experiment.model.block_size, cfg.batch_size, cfg.seed + 1)

    model = GPT(experiment.model).to(device)
    if cfg.compile and hasattr(torch, "compile"):
        model = torch.compile(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
        betas=(0.9, 0.95),
    )

    out_dir = Path(cfg.out_dir) / experiment.model.variant
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = out_dir / "metrics.jsonl"
    accounting = summarize_model(model, experiment.model)
    (out_dir / "accounting.json").write_text(json.dumps(accounting, indent=2), encoding="utf-8")

    amp = dtype in {torch.float16, torch.bfloat16} and device.type == "cuda"
    use_fp16_scaler = dtype == torch.float16 and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=True) if use_fp16_scaler else None

    t0 = time.time()
    tokens_seen = 0
    first_loss = None
    last_loss = None
    history: list[dict] = []

    model.train()
    optimizer.zero_grad(set_to_none=True)
    max_iters = int(cfg.max_iters or 0)
    for step in range(max_iters):
        lr = cosine_lr(step, cfg.warmup_iters, max_iters, cfg.lr, cfg.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        step_loss = 0.0
        step_aux = 0.0
        for _ in range(cfg.grad_accum):
            x, y = train_batcher.get_batch(device)
            with torch.autocast(device_type=device.type, dtype=dtype, enabled=amp):
                out = model(x, y)
                assert out.loss is not None
                total = out.loss
                if experiment.model.use_moe:
                    total = total + experiment.model.router_aux_loss_coef * out.aux_loss
                total = total / cfg.grad_accum
            scaler.scale(total).backward() if scaler is not None else total.backward()
            step_loss += out.loss.item() / cfg.grad_accum
            step_aux += float(out.aux_loss.detach().cpu()) / cfg.grad_accum
            tokens_seen += x.numel()

        if cfg.grad_clip > 0:
            if scaler is not None:
                scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        if scaler is not None:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if first_loss is None:
            first_loss = step_loss
        last_loss = step_loss

        if step % cfg.log_interval == 0 or step == cfg.max_iters - 1:
            elapsed = time.time() - t0
            tok_s = tokens_seen / max(elapsed, 1e-6)
            row = {
                "step": step,
                "loss": step_loss,
                "aux_loss": step_aux,
                "lr": lr,
                "tok_s": tok_s,
                "tokens_seen": tokens_seen,
            }
            history.append(row)
            with metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row) + "\n")
            print(
                f"[{experiment.model.variant}] step {step:4d} loss {step_loss:.4f} "
                f"tok/s {tok_s:.1f}",
                flush=True,
            )

        if cfg.eval_interval and step > 0 and step % cfg.eval_interval == 0:
            val = estimate_loss(model, val_batcher, device, cfg.eval_iters, dtype)
            print(f"[{experiment.model.variant}] val loss {val:.4f}", flush=True)

        if cfg.ckpt_interval and step > 0 and step % cfg.ckpt_interval == 0:
            torch.save(
                {"model": model.state_dict(), "config": experiment.to_dict(), "step": step},
                out_dir / f"ckpt_{step}.pt",
            )

    torch.save(
        {"model": model.state_dict(), "config": experiment.to_dict(), "step": cfg.max_iters},
        out_dir / "ckpt_final.pt",
    )
    elapsed = time.time() - t0
    summary = {
        "variant": experiment.model.variant,
        "dataset_source": source,
        "device": str(device),
        "dtype": str(dtype),
        "first_loss": first_loss,
        "last_loss": last_loss,
        "loss_dropped": last_loss is not None and first_loss is not None and last_loss < first_loss,
        "tokens_seen": tokens_seen,
        "elapsed_s": elapsed,
        "tok_s": tokens_seen / max(elapsed, 1e-6),
        "accounting": accounting,
        "history": history,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary
