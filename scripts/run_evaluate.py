import argparse
import json
import logging
from pathlib import Path
from typing import List, Optional

import torch
import numpy as np
from omegaconf import OmegaConf
from sklearn.metrics import f1_score, classification_report
from tqdm import tqdm

from mm_argfallacy.datamodule import MMUSEDFallacyDataModule
from mm_argfallacy.lightning_module import DifferentialLRLightningModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate a trained model on the validation set."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the experiment config file used for training this checkpoint.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save evaluation metrics.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on (cuda or cpu).",
    )
    return parser.parse_args()


def evaluate(
    model: DifferentialLRLightningModel,
    dataloader: torch.utils.data.DataLoader,
    device: str,
    num_classes: int,
    class_names: Optional[List[str]] = None,
):
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating on validation set"):
            inputs, labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(inputs)

            all_preds.extend(torch.argmax(outputs, dim=-1).cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    f1_macro = f1_score(all_labels, all_preds, average="macro", zero_division=0)

    f1_per_class_values = f1_score(
        all_labels,
        all_preds,
        average=None,
        labels=np.arange(num_classes),
        zero_division=0,
    )

    if class_names and len(class_names) == num_classes:
        f1_per_class = {class_names[i]: f1 for i, f1 in enumerate(f1_per_class_values)}
        target_names = class_names
    else:
        f1_per_class = {f"class_{i}": f1 for i, f1 in enumerate(f1_per_class_values)}
        target_names = [f"class_{i}" for i in np.arange(num_classes)]

    report = classification_report(
        all_labels,
        all_preds,
        labels=np.arange(num_classes),
        target_names=target_names,
        zero_division=0,
        digits=4,
    )
    print("\nClassification Report:")
    print(report)

    report_dict = classification_report(
        all_labels,
        all_preds,
        labels=np.arange(num_classes),
        target_names=target_names,
        output_dict=True,
        zero_division=0,
    )

    return {
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "classification_report": report_dict,
        "num_samples": len(all_labels),
    }


def save_metrics(metrics: dict, checkpoint_name: str, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / f"{checkpoint_name}_eval_metrics.json"

    with metrics_path.open("w") as f:
        json.dump(metrics, f, indent=4)

    logger.info(f"Evaluation metrics saved to {metrics_path}")


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    logger.info(f"Using device: {args.device}")
    checkpoint_path = Path(args.checkpoint).resolve()
    output_dir = Path(args.output_dir).resolve()
    checkpoint_name = checkpoint_path.stem

    datamodule = MMUSEDFallacyDataModule(
        batch_size=config.data_module.batch_size,
        dataset_cfg=config.data_module.dataset,
        collator_cfg=config.data_module.collator,
        seed=config.seed,
    )
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    if not val_loader or len(val_loader.dataset) == 0:
        logger.error("Validation set is empty or not properly configured.")
        return

    try:
        model = DifferentialLRLightningModel.load_from_checkpoint(
            checkpoint_path,
            map_location=torch.device(args.device),
        )
    except Exception as e:
        logger.error(f"Failed to load model from checkpoint: {e}")
        return

    num_classes = config.model.num_classes
    class_names = ["AE", "AA", "AH", "FC", "SS", "S"]

    metrics = evaluate(model, val_loader, args.device, num_classes, class_names)
    metrics.update(
        {
            "checkpoint_folder": checkpoint_path.parent.name,
            "checkpoint_name": checkpoint_path.name,
            "config_file": str(args.config),
        }
    )

    save_metrics(metrics, checkpoint_name, output_dir)


if __name__ == "__main__":
    main()
