from probabilistic_tsunami_surrogate.config import RunConfig
from probabilistic_tsunami_surrogate.training import train_model


def run_training(config):
    """Runs station-aware model training."""
    result = train_model(
        config.DATA_ROOT,
        config.SPLIT_DIR,
        config.RUN_DIR,
        max_epochs=config.MAX_EPOCHS,
        batch_size=config.BATCH_SIZE,
        learning_rate=config.LEARNING_RATE,
        micro_validation_size=config.MICRO_VALIDATION_SIZE,
        patience=config.PATIENCE,
        soft_kappa=config.SOFT_KAPPA,
        plateau_threshold=config.PLATEAU_THRESHOLD,
        plateau_cooldown=config.PLATEAU_COOLDOWN,
        num_workers=config.NUM_WORKERS,
        device=config.DEVICE,
    )
    print(f"final checkpoint: {result['final_checkpoint']}")
    print(f"best checkpoint: {result['best_checkpoint']}")


if __name__ == "__main__":
    run_training(RunConfig())
