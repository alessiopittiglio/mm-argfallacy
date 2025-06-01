# Leveraging Context for Multimodal Fallacy Classification in Political Debates

Official repository for the paper *"Leveraging Context for Multimodal Fallacy Classification in Political Debates"*.

The work was developed as part of our submission to the [MM-ArgFallacy2025](https://nlp-unibo.github.io/mm-argfallacy/2025/) shared task, focusing on multimodal detection and classification of argumentative fallacies in political debates.

## Task overview

The MM-ArgFallacy2025 shared task comprises two subtasks:

-  **Argumentative Fallacy Detection (AFD)**
-  **Argumentative Fallacy Classification (AFC)**

In this work, we focus on the **AFC task**. The goal is to classify fallacies within political debates by using three input modalities: text-only, audio-only, or a combination of text and audio.

We leverage pre-trained Transformer-based models and explore several ways to integrate context information.

## Setup and installation

### 1. Clone the repository

```
git clone https://github.com/alessiopittiglio/mm-argfallacy.git
cd mm-argfallacy
pip install -e .
```

### 2. Create environment and install dependencies

We recommend using a virtual environment (e.g., Conda or venv):

```bash
# Using Conda
conda create -n mm_argfallacy python=3.10
conda activate mm_argfallacy

# Install dependencies
pip install -r requirements.txt
```

## Dataset

This project utilizes the **MM-USEDFallacy dataset**, provided by the shared task organizers. Please refer to the official shared task website for instructions on how to obtain and download the dataset.

Once downloaded, you may need to update the `data_path` in the configuration files (see below) to point to your local dataset directory.

## Configuration

All model architectures, training procedures, and hyperparameters are managed through YAML configuration files located in the `configs/` directory.

Key configuration files for reproducing paper results:

**Text-Only Model:**
- `config/text_only/context_pool_roberta_ctx_1_paper_result.yaml`: Configuration for the ContextPool-RoBERT model.
- `config/text_only/context_pool_roberta_ctx_5_paper_result.yaml`: Configuration for a single component of the Text-Only ensemble.

**Audio-Only Model:**
- `config/audio_only/hubert_base_finetuned_paper_results.yaml`: Configuration for our fine-tuned Hubert-based audio model.

You can create new `.yaml` files or modify existing ones to experiment with different settings. The `data_module.dataset.input_mode` field within each YAML file determines which type of model and data is used by the training script.

## Training models

To train a model, use the unified training script `scripts/run_training.py` with a specific configuration file:

```bash
python scripts/run_training.py --config path/to/model_config.yaml
```

Replace the `--config` path with the desired YAML configuration file.

## Evaluation and prediction

### Generating predictions (single model)

To generate predictions using a single trained model:

```bash
python ./scripts/run_predict.py --config path/to/model_config.yaml --checkpoint path/to/model_checkpoint.ckpt
```

### Running ensembles

Our submitted text-only results for Table 3 were generated using an ensemble of three models.

### 1. Optimize ensemble weights (Optional)

This script uses Bayesian optimization to find optimal weights for combining model predictions on a validation set.

```bash
python ./scripts/optimize_weights.py --config path/to/ensemble_config.yaml
```

### 2. Generate ensemble predictions

Uses the weights and component models defined in its config file to produce ensemble predictions.

```bash
python ./scripts/run_predict_ensemble.py --config path/to/ensemble_config.yaml --weights_file path/to/weights.json  
```

## Results

The following results (Macro F1-scores) were reported by the shared task organizers on the private test set. Our submitted models are highlighted.

| **Input**      | **Team**                  | **F1**     |
|----------------|---------------------------|----------  |
| Text-Only      | Team NUST                 | 0.4856     |
|                | Baseline BiLSTM           | 0.4721     |
|                | **Our team**              | **0.4444** |
| Audio-Only     | **Our team**              | **0.3559** |
|                | Team EvaAdriana           | 0.1858     |
|                | Team NUST                 | 0.1588     |
| Text-Audio     | Team NUST                 | 0.4611     |
|                | **Our team**              | **0.4403** |
|                | Baseline RoBERTa + WavLM  | 0.3816     |
