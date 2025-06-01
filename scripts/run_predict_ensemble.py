import argparse
import json
import logging
import zipfile
from pathlib import Path
from typing import List, Dict, Any, Optional

import torch
from omegaconf import OmegaConf
import pandas as pd
from tqdm import tqdm
import lightning as L

from mm_argfallacy.datamodule import MMUSEDFallacyDataModule
from mm_argfallacy.lightning_module import DifferentialLRLightningModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate ensemble predictions for MM-ArgFallacy2025 models."
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML configuration file.",
    )
    weight_group = parser.add_mutually_exclusive_group(required=True)
    weight_group.add_argument(
        "--weights_file",
        type=str,
        help="Path to JSON file with ensemble weights.",
    )
    weight_group.add_argument(
        "--weights",
        type=float,
        nargs="*",
        help="List of weights for ensembling.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save outputs.",
    )
    parser.add_argument(
        "--ensemble_name",
        type=str,
        default="ensemble_predictions",
        help="Subfolder name for saving predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for computation (cuda or cpu).",
    )

    return parser.parse_args()


def normalize(weights: List[float]) -> List[float]:  # Duplicated, ideally in utils
    s = sum(weights)
    if s == 0:
        return [1.0 / len(weights)] * len(weights) if weights else []
    return [w / s for w in weights]


def load_model(
    model_tag: str, checkpoint_path: str, device: str
) -> DifferentialLRLightningModel:
    logger.info(f"Loading model '{model_tag}' from checkpoint: {checkpoint_path}")
    model = DifferentialLRLightningModel.load_from_checkpoint(
        checkpoint_path, map_location=torch.device(device)
    )
    model.eval()
    model.to(device)
    return model


def get_test_logits(
    model_tag: str,
    model: DifferentialLRLightningModel,
    dataset_cfg: Dict[str, Any],
    collator_cfg: Dict[str, Any],
    batch_size: int,
    seed: Optional[int],
    device: str,
) -> torch.Tensor:
    logger.info(f"Preparing test dataloader for model '{model_tag}'")

    datamodule = MMUSEDFallacyDataModule(
        batch_size=batch_size,
        dataset_cfg=dataset_cfg,
        collator_cfg=collator_cfg,
        seed=seed,
        data_path="data/",
        split_key="mm-argfallacy-2025",
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    logits = []

    with torch.no_grad():
        for batch in tqdm(test_loader, desc=f"Predicting for '{model_tag}'"):
            inputs = {k: v.to(device) for k, v in batch[0].items()}
            output = model(inputs)
            logits.append(output.cpu())

    return torch.cat(logits)


def ensemble_predict(
    models_logits: List[torch.Tensor], weights: List[float]
) -> torch.Tensor:
    weights = normalize(weights)
    logger.info(f"Using normalized weights: {weights}")
    ensemble_logits = sum(w * l for w, l in zip(weights, models_logits))
    return torch.argmax(ensemble_logits, dim=1)


def save_predictions(
    predictions: torch.Tensor,
    output_dir: Path,
    config: OmegaConf,
    ensemble_name: str,
):
    task = config.dataset.task_name.upper()
    setting = config.dataset.input_mode.lower().replace("_", "-")

    csv_name = f"{task}_{setting}.csv"
    zip_name = f"{task}_{setting}.zip"
    out_dir = output_dir / ensemble_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / csv_name
    zip_path = out_dir / zip_name

    df = pd.DataFrame({"label": predictions.numpy()})
    df.to_csv(csv_path, index=False)

    logger.info(f"Saving predictions to {csv_path} and {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname=csv_name)


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    models_cfg = config.get("ensemble_models")

    if not models_cfg:
        logger.error("Missing 'ensemble_models' in config file.")
        return

    seed = config.get("seed")
    if seed is not None:
        L.seed_everything(seed, workers=True)

    if args.weights_file:
        with open(args.weights_file) as f:
            weights_data = json.load(f)
            weights = weights_data["optimized_weights"]
        logger.info(f"Loaded weights from file: {weights}")
    else:
        weights = args.weights
        logger.info(f"Using weights from CLI: {weights}")

    if len(weights) != len(models_cfg):
        raise ValueError(f"#weights ({len(weights)}) != #models ({len(models_cfg)})")

    logits_list = []
    model_keys = list(models_cfg.keys())

    for key in model_keys:
        spec = models_cfg[key]
        tag = f"{key}"
        model = load_model(tag, spec["checkpoint_path"], args.device)

        logits = get_test_logits(
            tag,
            model,
            spec["dataset_cfg"],
            spec["collator_cfg"],
            spec.get("batch_size", 8),
            seed,
            args.device,
        )
        logits_list.append(logits)

    predictions = ensemble_predict(logits_list, weights)
    save_predictions(predictions, Path(args.output_dir), config, args.ensemble_name)


if __name__ == "__main__":
    main()
