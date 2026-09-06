"""CLI: run architecture research experiment harnesses independently."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from llmarcheval.experiments.dsa_needle import run_dsa_needle_experiment
from llmarcheval.experiments.mla_context import DEFAULT_CONTEXTS, run_mla_context_experiment
from llmarcheval.experiments.moe_routing import run_moe_routing_experiment
from llmarcheval.experiments.recurrent_loops import run_recurrent_loops_experiment


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-config", type=str, default="configs/smoke.yaml")
    parser.add_argument("--scale", type=str, default="smoke")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out-dir", type=str, default="results/experiments")
    parser.add_argument("--no-write", action="store_true", help="Skip JSON/JSONL output")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Frontier-SLM research experiment harness (offline architecture probes)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    moe = sub.add_parser("moe-routing", help="MoE routing probe across text regimes")
    _add_common(moe)
    moe.add_argument("--variant", type=str, default="v1_moe")
    moe.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Optional trained checkpoint (expects keys: model, config, step)",
    )

    mla = sub.add_parser("mla-context", help="MLA vs MHA context scaling probe")
    _add_common(mla)
    mla.add_argument("--variants", nargs="+", default=["v1_moe", "v2_moe_mla"])
    mla.add_argument("--contexts", nargs="+", type=int, default=list(DEFAULT_CONTEXTS))
    mla.add_argument("--steps", type=int, default=2)

    dsa = sub.add_parser("dsa-needle", help="DSA needle selection/retrieval probe")
    _add_common(dsa)
    dsa.add_argument("--variant", type=str, default="v3_moe_mla_dsa")
    dsa.add_argument("--haystack-lens", nargs="+", type=int, default=[64, 128, 256])
    dsa.add_argument(
        "--needle-positions",
        nargs="+",
        default=["middle"],
        choices=["start", "middle", "end"],
    )
    dsa.add_argument("--answer-tokens", type=int, default=8)

    rec = sub.add_parser("recurrent-loops", help="Recurrent loop compute-dial sweep")
    _add_common(rec)
    rec.add_argument("--variant", type=str, default="v4_recurrent")
    rec.add_argument("--loops", nargs="+", type=int, default=[1, 2, 3, 4])

    args = parser.parse_args(argv)
    write = not args.no_write
    common = dict(
        scale=args.scale,
        seed=args.seed,
        device_name=args.device,
        train_config=args.train_config,
        out_dir=args.out_dir,
        write=write,
    )

    if args.command == "moe-routing":
        record = run_moe_routing_experiment(variant=args.variant, ckpt=args.ckpt, **common)
    elif args.command == "mla-context":
        record = run_mla_context_experiment(
            variants=args.variants,
            contexts=args.contexts,
            steps=args.steps,
            **common,
        )
    elif args.command == "dsa-needle":
        record = run_dsa_needle_experiment(
            variant=args.variant,
            haystack_lens=args.haystack_lens,
            needle_positions=args.needle_positions,
            answer_tokens=args.answer_tokens,
            **common,
        )
    elif args.command == "recurrent-loops":
        record = run_recurrent_loops_experiment(
            variant=args.variant,
            loops=args.loops,
            **common,
        )
    else:
        raise SystemExit(f"Unknown command {args.command}")

    print(json.dumps(record, indent=2))
    if write and "output_path" in record:
        print(f"\nWrote {record['output_path']}", flush=True)
        print(f"Appended {Path(args.out_dir) / 'results.jsonl'}", flush=True)


if __name__ == "__main__":
    main()
