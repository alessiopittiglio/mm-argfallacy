import logging
from typing import Optional, List, Dict, Any

import torch
import torch.nn as nn
import lightning as L
from torch.nn.utils import clip_grad_norm_
from torchmetrics import MetricCollection
from torchmetrics.classification import MulticlassF1Score
from transformers import (
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
)

from mm_argfallacy.models import create_text_model, create_audio_model

logger = logging.getLogger(__name__)

OPTIMIZER_REGISTRY = {
    "adamw": torch.optim.AdamW,
}

SCHEDULER_REGISTRY = {
    "linear": get_linear_schedule_with_warmup,
    "cosine": get_cosine_schedule_with_warmup,
}


class BaseLightningModel(L.LightningModule):
    def __init__(
        self,
        input_mode: str,
        model_cfg: Dict[str, Any],
        num_classes: int,
        loss_name: str,
        optimizer_name: str,
        optimizer_hparams: Dict[str, Any],
        scheduler_name: Optional[str] = None,
        scheduler_hparams: Optional[Dict[str, Any]] = None,
        loss_weights: Optional[List[float]] = None,
        scheduler_monitor: str = "val_loss",
        scheduler_interval: str = "epoch",
        log_metrics: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.input_mode = input_mode
        self.num_classes = num_classes
        self.loss_name = loss_name
        self.loss_weights = loss_weights
        self.model_cfg = model_cfg

        self.model = self._build_model(self.input_mode, self.model_cfg)

        self.loss_fn = self._init_loss(self.loss_name.lower(), self.loss_weights)

        self.val_metrics = MetricCollection(
            {"f1": MulticlassF1Score(num_classes=self.num_classes, average="macro")}
        )
        self.test_metrics = MetricCollection(
            {"f1": MulticlassF1Score(num_classes=self.num_classes, average="macro")}
        )

    def _build_model(self, mode: str, cfg: Dict[str, Any]) -> nn.Module:
        if mode == "TEXT_ONLY":
            return create_text_model(cfg)
        elif mode == "AUDIO_ONLY":
            return create_audio_model(cfg)
        else:
            raise ValueError(f"Unsupported input mode: {mode}")

    def _init_loss(self, name: str, weights: Optional[List[float]]):
        if name == "cross_entropy":
            weight_tensor = (
                torch.tensor(weights, dtype=torch.float) if weights else None
            )
            return nn.CrossEntropyLoss(weight=weight_tensor)
        raise ValueError(f"Unsupported loss function: {name}")

    def forward(self, x):
        return self.model(x)

    def predict_step(self, batch, batch_idx):
        inputs, _ = batch
        logits = self(inputs)
        return torch.argmax(logits, dim=-1).detach().cpu().numpy()

    def _step(self, batch, stage: str):
        inputs, targets = batch
        logits = self(inputs)
        loss = self.loss_fn(logits, targets)

        self.log(f"{stage}_loss", loss, on_step=False, on_epoch=True, prog_bar=True)

        if stage == "val" and self.val_metrics:
            preds = torch.argmax(logits, dim=-1)
            self.val_metrics.update(preds, targets)

        if stage == "test" and self.test_metrics:
            preds = torch.argmax(logits, dim=-1)
            self.test_metrics.update(preds, targets)

        return loss

    def training_step(self, batch, batch_idx):
        return self._step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._step(batch, "test")

    def on_validation_epoch_end(self):
        if self.val_metrics:
            metrics = self.val_metrics.compute()
            for name, value in metrics.items():
                self.log(f"val_{name}", value, prog_bar=self.hparams.log_metrics)
            self.val_metrics.reset()

    def on_test_epoch_end(self):
        if self.test_metrics:
            metrics = self.test_metrics.compute()
            for name, value in metrics.items():
                self.log(f"test_{name}", value, prog_bar=self.hparams.log_metrics)
            self.test_metrics.reset()

    def on_before_optimizer_step(self, optimizer):
        grads = [p for p in self.parameters() if p.grad is not None]
        if not grads:
            norm = torch.tensor(0.0, device=self.device)
        else:
            norm = clip_grad_norm_(grads, max_norm=float("inf"))
        self.log("gradients/total_norm", norm, on_step=True, logger=True)

    def configure_optimizers(self):
        optimizer_cls = OPTIMIZER_REGISTRY.get(self.hparams.optimizer_name.lower())
        if not optimizer_cls:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")

        optimizer = optimizer_cls(
            self.model.parameters(), **self.hparams.optimizer_hparams
        )

        if not self.hparams.scheduler_name:
            return optimizer

        scheduler_cls = SCHEDULER_REGISTRY.get(self.hparams.scheduler_name)
        if not scheduler_cls:
            raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler_name}")

        scheduler = scheduler_cls(optimizer=optimizer, **self.hparams.scheduler_hparams)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.scheduler_monitor,
                "interval": self.hparams.scheduler_interval,
                "frequency": 1,
            },
        }


class DifferentialLRLightningModel(BaseLightningModel):
    """
    Lightning module that supports differential learning rates for different parameter
    groups.
    """

    def __init__(
        self,
        input_mode: str,
        model_cfg: Dict[str, Any],
        num_classes: int,
        loss_name: str,
        optimizer_name: str,
        optimizer_hparams: Dict[str, Any],
        scheduler_name: Optional[str] = None,
        scheduler_hparams: Optional[Dict[str, Any]] = None,
        loss_weights: Optional[List[float]] = None,
        scheduler_monitor: str = "val_loss",
        scheduler_interval: str = "epoch",
        log_metrics: bool = True,
        differential_lr: Optional[Dict[str, float]] = None,
    ):
        super().__init__(
            input_mode=input_mode,
            model_cfg=model_cfg,
            num_classes=num_classes,
            loss_name=loss_name,
            optimizer_name=optimizer_name,
            optimizer_hparams=optimizer_hparams,
            scheduler_name=scheduler_name,
            scheduler_hparams=scheduler_hparams,
            loss_weights=loss_weights,
            scheduler_monitor=scheduler_monitor,
            scheduler_interval=scheduler_interval,
            log_metrics=log_metrics,
        )
        self.differential_lr = differential_lr
        self.optimizer_hparams = optimizer_hparams

    def configure_optimizers(self):
        optimizer_cls = OPTIMIZER_REGISTRY.get(self.hparams.optimizer_name.lower())
        if not optimizer_cls:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")

        if self.differential_lr:
            optimizer = self._configure_differential_optimizer(optimizer_cls)
        else:
            optimizer = optimizer_cls(
                self.model.parameters(), **self.hparams.optimizer_hparams
            )

        if not self.hparams.scheduler_name:
            return optimizer

        scheduler_cls = SCHEDULER_REGISTRY.get(self.hparams.scheduler_name)
        if not scheduler_cls:
            raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler_name}")

        scheduler = scheduler_cls(optimizer=optimizer, **self.hparams.scheduler_hparams)

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": self.hparams.scheduler_monitor,
                "interval": self.hparams.scheduler_interval,
                "frequency": 1,
            },
        }

    def _configure_differential_optimizer(self, optimizer_cls):
        param_groups = []
        matched = set()

        for pattern, lr in self.differential_lr.items():
            group_params = [
                p
                for name, p in self.model.named_parameters()
                if pattern in name and name not in matched
            ]
            if group_params:
                param_groups.append({"params": group_params, "lr": lr})
                matched.update(
                    name for name, _ in self.model.named_parameters() if pattern in name
                )

        remaining_params = [
            p for name, p in self.model.named_parameters() if name not in matched
        ]
        if remaining_params:
            param_groups.append(
                {
                    "params": remaining_params,
                    "lr": self.optimizer_hparams.get("lr", 1e-3),
                }
            )

        logger.info(f"Configured {len(param_groups)} parameter groups.")
        for idx, group in enumerate(param_groups):
            print(
                f"  Group {idx}: lr={group['lr']},"
                f" num_params={len(group['params'])}"
            )

        opt_hparams = {k: v for k, v in self.optimizer_hparams.items() if k != "lr"}
        return optimizer_cls(param_groups, **opt_hparams)
