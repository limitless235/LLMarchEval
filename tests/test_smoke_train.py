from llmarcheval.config import ModelConfig, TrainConfig
from llmarcheval.train.trainer import train


def test_v0_smoke_loss_drops(tmp_path):
    exp_model = ModelConfig(
        variant="v0_dense",
        vocab_size=50257,
        n_layer=2,
        n_embd=64,
        n_head=4,
        block_size=32,
    )
    from llmarcheval.config import ExperimentConfig

    exp = ExperimentConfig(
        model=exp_model,
        train=TrainConfig(
            scale="smoke",
            dataset="corpus",
            max_iters=25,
            batch_size=4,
            warmup_iters=2,
            log_interval=25,
            eval_interval=0,
            ckpt_interval=0,
            out_dir=str(tmp_path),
            device="cpu",
            dtype="float32",
            seed=0,
        ),
    )
    summary = train(exp)
    assert summary["loss_dropped"]
    assert summary["last_loss"] < summary["first_loss"]
