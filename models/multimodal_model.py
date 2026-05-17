import torch
import torch.nn as nn
from transformers import ViTModel, AutoModel


class MultimodalCXRModel(nn.Module):
    def __init__(
        self,
        image_model_name="google/vit-base-patch16-224-in21k",
        text_model_name="emilyalsentzer/Bio_ClinicalBERT",
        hidden_dim=768,
        num_labels=5,
        num_attention_heads=8,
        dropout=0.2,
    ):
        super().__init__()

        self.image_encoder = ViTModel.from_pretrained(image_model_name)
        self.text_encoder = AutoModel.from_pretrained(text_model_name)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_labels),
        )

    def forward(self, pixel_values, input_ids, attention_mask):
        # image outputs: [batch, num_patches + 1, hidden_dim]
        image_outputs = self.image_encoder(pixel_values=pixel_values)
        image_embeddings = image_outputs.last_hidden_state

        # remove image CLS token, keep patch embeddings
        image_patch_embeddings = image_embeddings[:, 1:, :]

        # text outputs: [batch, seq_len, hidden_dim]
        text_outputs = self.text_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask,
        )
        text_embeddings = text_outputs.last_hidden_state

        # text attends to image patches
        fused_embeddings, attention_weights = self.cross_attention(
            query=text_embeddings,
            key=image_patch_embeddings,
            value=image_patch_embeddings,
        )

        # masked mean pooling over text sequence
        mask = attention_mask.unsqueeze(-1).float()
        fused_embeddings = fused_embeddings * mask
        pooled = fused_embeddings.sum(dim=1) / mask.sum(dim=1).clamp(min=1e-6)

        logits = self.classifier(pooled)

        return {
            "logits": logits,
            "attention_weights": attention_weights,
        }