import logging
import torch.nn as nn
from transformers import AutoConfig

from mamkit.models.text import Transformer
from mm_argfallacy.models.text_models import (
    ContextPoolingTextModel,
    ContextAttentionTextModel,
)
from mm_argfallacy.models.audio_models import (
    TransformerAudioModel,
    ContextPoolingAudioModel,
)

logger = logging.getLogger(__name__)


def create_text_model(config: dict) -> nn.Module:
    """
    Build a text model based on the configuration.
    """
    model_type = config["model_type"]
    logger.info(
        f"Creating text model (Type: {model_type}, Card: {config['model_card']})"
    )

    model_constructors = {
        "transformer": _build_transformer_model,
        "concat": _build_transformer_model,
        "context_pooling": _build_context_pooling_text_model,
        "context_attention": _build_context_attention_text_model,
    }

    if model_type not in model_constructors:
        raise ValueError(f"Unknown model_type: {model_type}")

    model = model_constructors[model_type](config)
    return model


def create_audio_model(config: dict) -> nn.Module:
    """
    Build an audio model based on the configuration.
    """
    model_type = config["model_type"]
    logger.info(
        f"Creating audio model (Type: {model_type}, Card: {config['model_card']})"
    )

    model_constructors = {
        "transformer": _build_transformer_audio_model,
        "temporal_average": _build_context_pooling_audio_model,
    }

    if model_type not in model_constructors:
        raise ValueError(f"Unknown audio model_type: {model_type}")

    model = model_constructors[model_type](config)
    return model


def _build_context_attention_text_model(config: dict) -> nn.Module:
    return ContextAttentionTextModel(
        model_card=config["model_card"],
        is_transformer_trainable=config["is_transformer_trainable"],
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"],
        use_attentive_pooling=config.get("use_attentive_pooling", True),
        use_fusion_gate=config.get("use_fusion_gate", False),
        hidden_dims=config["head"]["hidden_layers"],
    )


def _build_context_pooling_text_model(config: dict) -> nn.Module:
    return ContextPoolingTextModel(
        model_card=config["model_card"],
        is_transformer_trainable=config["is_transformer_trainable"],
        dropout_rate=config["dropout_rate"],
        num_classes=config["num_classes"],
        hidden_dims=config["head"]["hidden_layers"],
    )


def _build_transformer_model(config: dict) -> nn.Module:
    head_module = lambda: _build_head(
        head_config=config.get("head"),
        model_card=config["model_card"],
        num_classes=config["num_classes"],
    )
    return Transformer(
        model_card=config["model_card"],
        is_transformer_trainable=config["is_transformer_trainable"],
        dropout_rate=config["dropout_rate"],
        head=head_module,
    )


def _build_transformer_audio_model(config: dict) -> nn.Module:
    return TransformerAudioModel(
        model_card=config["model_card"],
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"],
        num_layers_to_finetune=config["num_layers_to_finetune"],
        hidden_dims=config["head"]["hidden_layers"],
    )


def _build_context_pooling_audio_model(config: dict) -> nn.Module:
    return ContextPoolingAudioModel(
        model_card=config["model_card"],
        num_classes=config["num_classes"],
        dropout_rate=config["dropout_rate"],
        num_layers_to_finetune=config["num_layers_to_finetune"],
        hidden_dims=config["head"]["hidden_layers"],
    )


def _build_head(head_config: dict, model_card: str, num_classes: int) -> nn.Module:
    """
    Build an MLP classifier to use as a "head".
    """
    transformer_config = AutoConfig.from_pretrained(model_card)
    input_dim = transformer_config.hidden_size

    if not head_config:
        return nn.Linear(input_dim, num_classes)

    hidden_dims = head_config.get("hidden_layers", [])
    activation_name = head_config.get("activation", "ReLU")
    activation_fn = getattr(nn, activation_name, nn.ReLU)

    layers = []
    current_dim = input_dim

    for h_dim in hidden_dims:
        layers.append(nn.Linear(current_dim, h_dim))
        layers.append(activation_fn())
        current_dim = h_dim

    layers.append(nn.Linear(current_dim, num_classes))
    return nn.Sequential(*layers)
