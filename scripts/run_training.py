import argparse
import logging
import math
from pathlib import Path

import lightning as L
import torch
import wandb

from omegaconf import OmegaConf

from lightning.pytorch.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    LearningRateMonitor,
)
from lightning.pytorch.loggers import WandbLogger

from mm_argfallacy.datamodule import MMUSEDFallacyDataModule
from mm_argfallacy.lightning_module import DifferentialLRLightningModel

torch.set_float32_matmul_precision("high")
logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Train a model for MM-ArgFallacy2025")

    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file."
    )

    return parser.parse_args()


def generate_run_name(config: dict) -> str | None:
    name_parts = [
        config["data_module"]["dataset"]["task_name"],
        config["data_module"]["dataset"]["input_mode"].lower().replace("_", "-"),
        config["model"]["model_card"].split("/")[-1].lower(),
    ]

    if config.get("context_window"):
        name_parts.append(f"ctx-{config['context_window']}")

    return "-".join(name_parts)


def init_wandb(config: dict):
    if config.logger.wandb.enabled:
        run_name = generate_run_name(config)
        run = wandb.init(
            project=config.logger.wandb.project,
            config=OmegaConf.to_container(config, resolve=True),
            name=run_name,
        )
        wandb.define_metric("val_f1", summary="max")
        logger = WandbLogger(experiment=run)
    else:
        run = None
        logger = None

    return run, logger


def build_scheduler_params(config: dict, steps_per_epoch: int) -> dict:
    epochs = config.trainer.max_epochs
    warmup_ratio = config.scheduler.params.warmup_ratio
    total_steps = steps_per_epoch * epochs
    return {
        "num_warmup_steps": int(total_steps * warmup_ratio),
        "num_training_steps": total_steps,
    }


def build_callbacks(config: dict, save_dir: Path):
    early_stopping = EarlyStopping(**config.callbacks.early_stopping)

    checkpoint_cfg = {
        **config.callbacks.model_checkpoint,
        "filename": "best-{epoch}-{val_f1:.4f}",
        "dirpath": save_dir,
    }
    model_checkpoint = ModelCheckpoint(**checkpoint_cfg)

    return [
        early_stopping,
        model_checkpoint,
        LearningRateMonitor(logging_interval="step"),
    ]


def main():
    args = parse_args()

    base_cfg = OmegaConf.load(args.config)
    cli_cfg = OmegaConf.from_cli()
    config = OmegaConf.merge(base_cfg, cli_cfg)

    run, logger = init_wandb(config)

    L.seed_everything(config.seed, workers=True)

    datamodule = MMUSEDFallacyDataModule(
        batch_size=config.data_module.batch_size,
        dataset_cfg=config.data_module.dataset,
        collator_cfg=config.data_module.collator,
        seed=config.seed,
    )

    datamodule.setup()
    train_loader = datamodule.train_dataloader()
    val_loader = datamodule.val_dataloader()

    steps_per_epoch = math.ceil(
        len(train_loader) / config.trainer.accumulate_grad_batches
    )
    scheduler_hparams = build_scheduler_params(config, steps_per_epoch)

    model = DifferentialLRLightningModel(
        input_mode=config.data_module.dataset.input_mode,
        model_cfg=config.model,
        num_classes=config.model.num_classes,
        loss_name=config.loss_function.name,
        optimizer_name=config.optimizer.name,
        optimizer_hparams=config.optimizer.params,
        scheduler_name=config.scheduler.name if "scheduler" in config else None,
        scheduler_hparams=scheduler_hparams,
        loss_weights=config.loss_function.args.get("class_weights", None),
        scheduler_monitor=config.get("scheduler_monitor", "val_loss"),
        scheduler_interval=config.get("scheduler_interval", "step"),
        log_metrics=config.get("log_metrics", True),
        differential_lr=config.optimizer.get("differential_lr", None),
    )

    save_dir = (
        Path(__file__).parent.parent.resolve()
        / "checkpoints"
        / config.data_module.dataset.task_name
        / config.data_module.dataset.input_mode.lower().replace("_", "-")
        / wandb.run.id
        if run
        else "local"
    )
    save_dir.mkdir(parents=True, exist_ok=True)

    callbacks = build_callbacks(config, save_dir)

    trainer = L.Trainer(
        **config["trainer"],
        callbacks=callbacks,
        logger=logger,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    val_metrics = trainer.test(ckpt_path="best", dataloaders=val_loader)[0]

    logging.info(f"Validation metrics: {val_metrics}")

    if run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
