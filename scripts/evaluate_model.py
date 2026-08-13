import json

from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.evaluation import evaluate_checkpoint


def evaluate_model(config):
    """Evaluates one checkpoint and writes its metric summary."""
    summary = evaluate_checkpoint(
        config.DATA_ROOT,
        config.SPLIT_DIR,
        config.RUN_DIR,
        checkpoint=config.CHECKPOINT,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        device=config.DEVICE,
    )
    config.evaluation_path.parent.mkdir(parents=True, exist_ok=True)
    with config.evaluation_path.open("w") as stream:
        json.dump(summary, stream, indent=2)
    print(f"evaluation: {config.evaluation_path}")


if __name__ == "__main__":
    evaluate_model(RunConfig())
