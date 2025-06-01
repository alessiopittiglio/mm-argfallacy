import argparse
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
from omegaconf import OmegaConf
from tqdm import tqdm
from sklearn.metrics import f1_score
from skopt import gp_minimize
from skopt.space import Real
import lightning as L

from mm_argfallacy.datamodule import MMUSEDFallacyDataModule
from mm_argfallacy.lightning_module import DifferentialLRLightningModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

torch.set_float32_matmul_precision("high")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Optimize ensemble weights for MM-ArgFallacy2025 models."
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to ensemble YAML configuration.",
    )
    parser.add_argument(
        "--output_weights_file",
        default="weights.json",
        help="Output path for weights.",
    )
    parser.add_argument(
        "--n_calls",
        type=int,
        default=20,
        help="Number of optimization steps.",
    )
    parser.add_argument(
        "--initial_weights",
        type=float,
        nargs="*",
        default=None,
        help="Initial guess for ensemble weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use.",
    )
    return parser.parse_args()


def calculate_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute macro F1 score from model logits and labels."""
    preds = logits if logits.ndim == 1 else torch.argmax(logits, dim=1)
    f1 = f1_score(labels.numpy(), preds.numpy(), average="macro", zero_division=0)
    return f1


def normalize(weights: List[float]) -> List[float]:
    total = sum(weights)
    if total == 0:
        return [1.0 / len(weights)] * len(weights)
    return [w / total for w in weights]


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


def get_model_logits_and_labels(
    model_tag: str,
    model: DifferentialLRLightningModel,
    dataset_cfg: Dict[str, Any],
    collator_cfg: Dict[str, Any],
    batch_size: int,
    seed: Optional[int],
    device: str,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logger.info(f"Preparing validation dataloader for model '{model_tag}'")

    datamodule = MMUSEDFallacyDataModule(
        batch_size=batch_size,
        dataset_cfg=dataset_cfg,
        collator_cfg=collator_cfg,
        seed=seed,
    )
    datamodule.setup()
    val_loader = datamodule.val_dataloader()

    if not val_loader or len(val_loader.dataset) == 0:
        raise ValueError(f"Validation data missing or empty for model '{model_tag}'")

    logits = []
    labels = []

    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Collecting logits for '{model_tag}'"):
            inputs, batch_labels = batch
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(inputs)
            logits.append(outputs.cpu())
            labels.append(batch_labels.cpu())

    return torch.cat(logits), torch.cat(labels)


def ensemble_objective(
    weights: List[float],
    model_logits_list: List[torch.Tensor],
    reference_labels: torch.Tensor,
) -> float:
    normalized = normalize(weights)

    combined_logits = sum(w * l for w, l in zip(normalized, model_logits_list))
    predictions = combined_logits.argmax(dim=1)

    f1 = f1_score(
        reference_labels.numpy(),
        predictions.numpy(),
        average="macro",
        zero_division=0,
    )
    return -f1  # negative for minimization


def run_ensemble_optimization(
    config: Dict[str, Any], args: argparse.Namespace
) -> Tuple[List[float], float, List[str], Dict[str, float]]:
    models_cfg = config["ensemble_models"]
    if not models_cfg:
        raise ValueError("The 'ensemble_models' section is empty.")

    seed = config.get("seed")
    if seed is not None:
        L.seed_everything(seed, workers=True)

    batch_size = config.get("val_batch_size", 8)

    logits_list = []
    label_list = []
    model_tags = list(models_cfg.keys())
    individual_f1s = {}

    logger.info("Loading models and gathering logits")
    for tag in model_tags:
        cfg = models_cfg[tag]
        model = load_model(tag, cfg["checkpoint_path"], args.device)

        logits, labels = get_model_logits_and_labels(
            tag,
            model,
            cfg["dataset_cfg"],
            cfg["collator_cfg"],
            batch_size,
            seed,
            args.device,
        )
        logits_list.append(logits)
        label_list.append(labels)

        f1 = calculate_f1(logits, labels)
        individual_f1s[tag] = f1
        logger.info(f"F1 Macro for '{tag}': {f1:.4f}")

    reference_labels = label_list[0]
    num_models = len(model_tags)
    space = [Real(0.0, 1.0, name=f"w_{i}") for i in range(len(model_tags))]

    logger.info(
        f"Starting Bayesian optimization for {args.n_calls} calls, "
        "optimizing {num_models} weights..."
    )

    optim_func = lambda w: ensemble_objective(
        w,
        logits_list,
        reference_labels,
    )

    initial = args.initial_weights
    if initial and len(initial) != num_models:
        logger.warning(
            f"Number of initial_weights ({len(initial)}) does not match "
            f"number of models ({num_models}). Ignoring initial_weights."
        )
        initial = None

    n_init = min(
        args.n_calls - 1 if args.n_calls > 1 else 1,
        max(1, int(num_models * 1.5)),
    )
    if args.n_calls <= n_init and args.n_calls > 0:
        n_init = args.n_calls - 1
    if n_init <= 0:
        n_init = 1

    result = gp_minimize(
        func=optim_func,
        dimensions=space,
        n_calls=args.n_calls,
        n_initial_points=n_init,
        x0=initial,
        random_state=seed,
        acq_func="EI",
    )

    best_weights = normalize(result.x)
    best_f1 = -result.fun

    logger.info(f"Optimization finished.")
    logger.info(f"Best ensemble F1: {best_f1:.4f}")
    for key, w in zip(model_tags, best_weights):
        logger.info(f"  {key}: weight={w:.4f}, individual F1={individual_f1s[key]:.4f}")

    return best_weights, best_f1, model_tags, individual_f1s


def main():
    args = parse_args()
    config = OmegaConf.load(args.config)
    models_cfg = config.get("ensemble_models")

    if not models_cfg:
        logger.error("Missing 'ensemble_models' in config file.")
        return

    logger.info(f"Using device: {args.device}")
    logger.info(f"Number of optimization calls: {args.n_calls}")
    if args.initial_weights:
        logger.info(f"Initial weights guess: {args.initial_weights}")

    weights, ensemble_f1, model_order, individual_f1s = run_ensemble_optimization(
        config, args
    )

    output_path = Path(args.output_weights_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    weights_data_to_save = {
        "ensemble_f1_macro_validation": ensemble_f1,
        "optimized_weights": weights,
        "model_order": model_order,
        "individual_model_f1_scores": individual_f1s,
        "config_file_used": Path(args.config).resolve().name,
        "n_optimization_calls": args.n_calls,
    }
    with open(output_path, "w") as f:
        json.dump(weights_data_to_save, f, indent=4)
    logger.info(f"Saved optimized weights to {output_path}")


if __name__ == "__main__":
    main()
