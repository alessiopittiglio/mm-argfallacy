import logging
from typing import Dict, Tuple

import torch
import torch.nn as nn
from transformers import AutoModel

logger = logging.getLogger(__name__)


class BaseAudioModel(nn.Module):
    """
    Base class for audio models that wraps a transformer and provides utility methods
    for encoding, pooling, and fine-tuning control.
    """

    def __init__(
        self,
        model_card: str,
        dropout_rate: float = 0.0,
        num_layers_to_finetune: int = 0,
        use_proper_masking: bool = True,
    ):
        super().__init__()
        self.model_card = model_card
        self.transformer = AutoModel.from_pretrained(model_card)
        self.hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

        self._freeze_transformer_layers(num_layers_to_finetune)

        self.use_proper_masking = use_proper_masking

        if not self.use_proper_masking:
            logger.warning("Using basic mean pooling WITHOUT attention masking.")

    def _freeze_transformer_layers(self, num_layers_to_finetune: int):
        """
        Freezes or unfreezes transformer layers based on user configuration.
        """
        if hasattr(self.transformer, "encoder") and hasattr(
            self.transformer.encoder, "layers"
        ):
            transformer_layers = self.transformer.encoder.layers
        elif hasattr(self.transformer, "layers"):
            transformer_layers = self.transformer.layers

        if transformer_layers is None:
            logger.warning("Could not find transformer layers to freeze/unfreeze.")
            return

        if num_layers_to_finetune == 0:
            logger.info("Freezing all transformer layers.")
            for param in self.transformer.parameters():
                param.requires_grad = False

        elif num_layers_to_finetune < 0:
            logger.info("Unfreezing all transformer layers.")
            for param in self.transformer.parameters():
                param.requires_grad = True

        else:
            logger.info("Freezing all transformer layers.")
            for param in self.transformer.parameters():
                param.requires_grad = False

            num_total_layers = len(transformer_layers)
            num_to_unfreeze = min(num_layers_to_finetune, num_total_layers)
            logger.info(f"Unfreezing top {num_to_unfreeze}/{num_total_layers} layers.")

            for layer in transformer_layers[-num_to_unfreeze:]:
                for param in layer.parameters():
                    param.requires_grad = True

    def _build_head(
        self, input_dim: int, num_classes: int, hidden_dims: Tuple[int, ...]
    ) -> nn.Sequential:
        layers = []
        for dim in hidden_dims:
            layers.append(nn.Linear(input_dim, dim))
            layers.append(nn.ReLU())
            input_dim = dim
        layers.append(nn.Linear(input_dim, num_classes))
        return nn.Sequential(*layers)

    def _encode(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.transformer(
            input_values=input_values, attention_mask=attention_mask
        )
        return outputs.last_hidden_state

    def _mean_pool(self, embeddings: torch.Tensor) -> torch.Tensor:
        return embeddings.mean(dim=1)

    def _masked_mean_pool(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Mean pooling with attention mask.
        """
        mask_expanded = attention_mask.unsqueeze(-1)
        sum_embeddings = (embeddings * mask_expanded).sum(dim=1)
        token_counts = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        return sum_embeddings / token_counts

    def _get_pooled_output(
        self,
        embeddings: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Returns either a masked or unmasked pooled representation.
        """
        if not self.use_proper_masking:
            return self._mean_pool(embeddings)

        device = embeddings.device

        input_lengths = attention_mask.sum(dim=1)
        conv_strides = self.transformer.config.conv_stride
        total_stride = torch.prod(torch.tensor(conv_strides)).item()

        output_lengths = input_lengths // total_stride
        output_lengths = torch.clamp(output_lengths, max=embeddings.size(1))

        feat_range = torch.arange(embeddings.size(1), device=device).unsqueeze(0)
        pooled_mask = (feat_range < output_lengths.unsqueeze(1)).long()

        return self._masked_mean_pool(embeddings, pooled_mask)


class TransformerAudioModel(BaseAudioModel):
    """
    Standard Transformer classifier with optional masking.
    """

    def __init__(
        self,
        model_card: str,
        num_classes: int,
        dropout_rate: float = 0.0,
        num_layers_to_finetune: int = 0,
        hidden_dims: Tuple[int, ...] = (100, 50),
    ):
        super().__init__(model_card, dropout_rate, num_layers_to_finetune)

        self.classifier = self._build_head(self.hidden_size, num_classes, hidden_dims)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_values = batch["inputs"]
        input_mask = batch["input_mask"]

        embeddings = self._encode(input_values, input_mask)
        pooled = self._get_pooled_output(embeddings, input_mask)
        pooled = self.dropout(pooled)

        logits = self.classifier(pooled)
        return logits


class ContextPoolingAudioModel(BaseAudioModel):
    """
    Siamese Transformer-based model that uses pooled embeddings from a transformer to
    classify audio with context.
    """

    def __init__(
        self,
        model_card: str,
        num_classes: int,
        dropout_rate: float = 0.0,
        num_layers_to_finetune: int = 0,
        hidden_dims: Tuple[int, ...] = (100, 50),
    ):
        super().__init__(model_card, dropout_rate, num_layers_to_finetune)
        self.classifier = self._build_head(
            self.hidden_size * 2,
            num_classes,
            hidden_dims,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_values, input_mask = batch["inputs"], batch["input_mask"]
        context_values, context_mask = batch["context"], batch["context_mask"]

        input_embeddings = self._encode(input_values, input_mask)
        context_embeddings = self._encode(context_values, context_mask)

        input_pooled = self._get_pooled_output(input_embeddings, input_mask)
        context_pooled = self._get_pooled_output(context_embeddings, context_mask)

        combined_pooled = torch.cat([input_pooled, context_pooled], dim=-1)
        combined_pooled = self.dropout(combined_pooled)

        logits = self.classifier(combined_pooled)
        return logits
