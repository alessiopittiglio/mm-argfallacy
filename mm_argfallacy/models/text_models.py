from typing import Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel


class BaseTextModel(nn.Module):
    """
    Base class for text models with utility methods for encoding and pooling.
    """

    def __init__(
        self,
        model_card: str,
        is_transformer_trainable: bool,
        dropout_rate: float,
    ):
        super().__init__()

        if "deberta" in model_card:
            self.transformer = AutoModel.from_pretrained(model_card)
        else:
            self.transformer = AutoModel.from_pretrained(model_card, device_map="auto")

        if not is_transformer_trainable:
            for param in self.transformer.parameters():
                param.requires_grad = False

        self.hidden_size = self.transformer.config.hidden_size
        self.dropout = nn.Dropout(dropout_rate)

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
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.last_hidden_state

    def _mean_pool(
        self, embeddings: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        mask_expanded = attention_mask.unsqueeze(-1)
        sum_embeddings = (embeddings * mask_expanded).sum(dim=1)
        token_counts = torch.clamp(attention_mask.sum(dim=1, keepdim=True), min=1e-9)
        return sum_embeddings / token_counts


class ContextPoolingTextModel(BaseTextModel):
    """
    Siamese Transformer-based model that uses pooled embeddings from a transformer to
    classify text based on context.

    This model processes pairs of text inputs (main text and context), encodes them
    using a shared transformer model, pools the embeddings, and then combines them for
    classification.
    """

    def __init__(
        self,
        model_card: str,
        num_classes: int,
        dropout_rate: float = 0.0,
        is_transformer_trainable: bool = False,
        hidden_dims: Tuple[int, ...] = (100, 50),
    ):
        super().__init__(model_card, is_transformer_trainable, dropout_rate)
        self.classifier = self._build_head(
            self.hidden_size * 2,
            num_classes,
            hidden_dims,
        )

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids, input_mask = batch["inputs"], batch["input_mask"]
        context_ids, context_mask = batch["context"], batch["context_mask"]

        input_embeddings = self._encode(input_ids, input_mask)
        context_embeddings = self._encode(context_ids, context_mask)

        input_pooled = self._mean_pool(input_embeddings, input_mask)
        context_pooled = self._mean_pool(context_embeddings, context_mask)

        combined_pooled = torch.cat([input_pooled, context_pooled], dim=-1)
        combined_pooled = self.dropout(combined_pooled)

        logits = self.classifier(combined_pooled)
        return logits


class ContextAttentionTextModel(BaseTextModel):
    """
    Transformer-based model with cross-attention mechanism for integrating text and
    context.

    This model processes text and context inputs separately through a shared
    transformer, then applies cross-attention to integrate context information into the
    text representation.
    """

    def __init__(
        self,
        model_card: str,
        num_classes: int,
        dropout_rate=0.0,
        num_attention_heads: int = 8,
        is_transformer_trainable: bool = False,
        hidden_dims: Tuple[int, ...] = (100, 50),
        use_attentive_pooling: bool = True,
        use_fusion_gate: bool = True,
    ):
        super().__init__(model_card, is_transformer_trainable, dropout_rate)

        self.use_attentive_pooling = use_attentive_pooling
        self.use_fusion_gate = use_fusion_gate

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.hidden_size,
            num_heads=num_attention_heads,
            batch_first=True,
            dropout=dropout_rate,
        )

        self.norm = nn.LayerNorm(self.hidden_size)

        if use_fusion_gate:
            self.fusion_gate = nn.Sequential(
                nn.Linear(self.hidden_size * 2, self.hidden_size), nn.Sigmoid()
            )

        if use_attentive_pooling:
            self.att_pool = nn.Sequential(
                nn.Linear(self.hidden_size, self.hidden_size // 2),
                nn.Tanh(),
                nn.Linear(self.hidden_size // 2, 1),
            )

        self.classifier = self._build_head(self.hidden_size, num_classes, hidden_dims)

    def _apply_cross_attention(
        self, text: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor
    ) -> torch.Tensor:
        key_padding_mask = context_mask == 0
        attended, _ = self.cross_attention(
            query=text, key=context, value=context, key_padding_mask=key_padding_mask
        )
        return attended

    def _fuse(self, text: torch.Tensor, attended_context: torch.Tensor) -> torch.Tensor:
        if self.use_fusion_gate:
            gate_input = torch.cat([text, attended_context], dim=-1)
            gate = self.fusion_gate(gate_input)
            fused = text + gate * attended_context
        else:
            fused = text + attended_context
        return self.norm(fused)

    def _pool(self, embeddings: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if self.use_attentive_pooling:
            scores = self.att_pool(embeddings).squeeze(-1)
            scores = scores.masked_fill(mask == 0, -1e9)
            weights = F.softmax(scores, dim=1)
            pooled = (embeddings * weights.unsqueeze(-1)).sum(dim=1)
        else:
            pooled = self._mean_pool(embeddings, mask)
        return self.dropout(pooled)

    def forward(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        input_ids, input_mask = batch["inputs"], batch["input_mask"]
        context_ids, context_mask = batch["context"], batch["context_mask"]

        input_embeddings = self._encode(input_ids, input_mask)
        context_embeddings = self._encode(context_ids, context_mask)

        attended = self._apply_cross_attention(
            input_embeddings,
            context_embeddings,
            context_mask,
        )
        fused = self._fuse(input_embeddings, attended)
        pooled = self._pool(fused, input_mask)

        logits = self.classifier(pooled)
        return logits
