import argparse
import logging
import zipfile
from pathlib import Path

import pandas as pd
import torch
from omegaconf import OmegaConf
from tqdm import tqdm

from mm_argfallacy.datamodule import MMUSEDFallacyDataModule
from mm_argfallacy.lightning_module import DifferentialLRLightningModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate predictions for MM-ArgFallacy2025 challenge."
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.ckpt).",
    )
    parser.add_argument(
        "--config", type=str, required=True, help="Path to the config YAML file."
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/",
        help="Directory to save the predictions.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for prediction (cuda or cpu).",
    )
    return parser.parse_args()


def predict(
    model: DifferentialLRLightningModel,
    dataloader: torch.utils.data.DataLoader,
    device: str,
):
    model.eval()
    model.to(device)
    preds = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Predicting"):
            inputs = {k: v.to(device) for k, v in batch[0].items()}
            outputs = model(inputs)
            preds.extend(torch.argmax(outputs, dim=-1).cpu().tolist())

    return preds


def save_predictions(predictions, checkpoint_name, task_name, input_mode, output_dir):
    task = task_name.upper()
    setting = input_mode.lower().replace("_", "-")

    csv_name = f"{task}_{setting}.csv"
    zip_name = f"{task}_{setting}.zip"
    out_dir = output_dir / checkpoint_name
    out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = out_dir / csv_name
    zip_path = out_dir / zip_name

    pd.DataFrame({"label": predictions}).to_csv(csv_path, index=False)

    logger.info(f"Saving predictions to {csv_path} and {zip_path}...")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        zipf.write(csv_path, arcname=csv_name)


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)

    logger.info(f"Using device: {args.device}")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = Path(args.checkpoint)
    checkpoint_name = checkpoint_path.parent.name

    datamodule = MMUSEDFallacyDataModule(
        batch_size=config.data_module.batch_size,
        dataset_cfg=config.data_module.dataset,
        collator_cfg=config.data_module.collator,
        seed=config.seed,
        data_path="data/",
        split_key="mm-argfallacy-2025",
    )
    datamodule.setup(stage="test")
    test_loader = datamodule.test_dataloader()

    model = DifferentialLRLightningModel.load_from_checkpoint(
        checkpoint_path,
        map_location=torch.device(args.device),
    )

    predictions = predict(model, test_loader, args.device)

    save_predictions(
        predictions,
        checkpoint_name,
        config.data_module.dataset.task_name,
        config.data_module.dataset.input_mode,
        output_dir,
    )


if __name__ == "__main__":
    main()
