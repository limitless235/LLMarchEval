# FineWeb-Edu frozen packed corpus (1M control + 20M primary)

This document describes the **planned** FineWeb-Edu experiment matrix on
`local_16gb_w768`. It does **not** replace the frozen TinyStories @ 1M pilot.

## Corpus

```bash
# Dry-run (no download):
python scripts/pack_fineweb_edu.py --dry-run --target-tokens 100000000

# After approval (downloads/streams FineWeb-Edu sample-10BT):
python scripts/pack_fineweb_edu.py --target-tokens 100000000 \
  --out-dir data/fineweb_edu_100m --seed 1337 --revision <HF_COMMIT>
```

Outputs under `data/fineweb_edu_100m/`:
- `tokens.uint16.npy` (~200 MB for 100M tokens)
- `manifest.json` (dataset id/config/revision, tokenizer, sha256, counts)
- `tokens.uint16.npy.sha256`

Train/val split remains the trainer’s existing **prefix 90% / suffix 10%** rule.

## Campaigns

| Config | Budget | Steps | tokens_seen | out_dir |
|---|---|---:|---:|---|
| `configs/fineweb_edu_1m.yaml` | 1M | 245 | 1,003,520 | `results/fineweb_edu_1m` |
| `configs/fineweb_edu_20m.yaml` | 20M | 4,883 | 20,000,768 | `results/fineweb_edu_20m` |

```bash
for v in v0_dense v1_moe v2_moe_mla v3_moe_mla_dsa v4_recurrent; do
  python scripts/campaign.py --train-config configs/fineweb_edu_1m.yaml \
    --variant "$v" --tokens-budget 1000000
done

for v in v0_dense v1_moe v2_moe_mla v3_moe_mla_dsa v4_recurrent; do
  python scripts/campaign.py --train-config configs/fineweb_edu_20m.yaml \
    --variant "$v" --tokens-budget 20000000
done
```

## Protocol notes

- Architecture/precision/B/T/optim unchanged from `local_16gb_w768`.
- `warmup_iters=20` kept (same recipe; warmup is a smaller fraction at 20M).
- Checkpoints: interval 1000, atomic writes, keep last 2 mid-ckpts + final.
- Post-hoc eval: `llme-eval-ckpt --ckpt … --train-config configs/fineweb_edu_20m.yaml --eval-iters 50`
  (or 100). Must use the packed train config so val data matches training.

## Explicitly not in this phase

- No 50M/100M Mac campaigns by default
- No FP16/BF16
- No TinyStories artifact modification
