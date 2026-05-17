"""
Batch cross-attention visualization.

For each label:
1. Select positive samples.
2. Find a clinically relevant token that appears in Findings.
3. Extract token-to-image-patch cross-attention.
4. Save heatmap overlays for manual inspection.
"""

from pathlib import Path
import ast

import cv2
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image
from torchvision import transforms
from transformers import AutoTokenizer

from data_utils.iu_xray_dataset import IUXrayDataset
from models.multimodal_model import MultimodalCXRModel


LABEL_CONFIG = {
    "effusion": ["effusion", "pleural", "fluid"],
    "cardiomegaly": ["cardiomegaly", "enlarged", "cardiac", "heart"],
    "edema": ["edema", "interstitial", "vascular", "congestion"],
    "pneumonia": ["pneumonia", "opacity", "opacities", "airspace", "consolidation"],
    "pneumothorax": ["pneumothorax"],
}

DATA_DIR = Path(r"C:\multimodal_cxt\data\iu_xray")
METADATA_PATH = DATA_DIR / "iu_xray_labeled_metadata.csv"
CHECKPOINT_PATH = Path("checkpoints/unweighted_multimodal_cxr_model_epoch_1.pt")

OUTPUT_DIR = Path("outputs/cross_attention")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def find_token_positions(tokens, target_word):
    target_word = target_word.lower()

    for i in range(len(tokens)):
        rebuilt = ""

        for j in range(i, min(i + 10, len(tokens))):
            token = tokens[j]

            if token in ["[CLS]", "[SEP]", "[PAD]"]:
                break

            rebuilt += token.replace("##", "").lower()

            if rebuilt == target_word:
                return list(range(i, j + 1))

            if not target_word.startswith(rebuilt):
                break

    return None


def process_attention_heatmap(patch_attention, top_percent=10, power=2):
    heatmap = patch_attention.reshape(14, 14)

    threshold = np.percentile(heatmap, 100 - top_percent)
    heatmap = np.where(heatmap >= threshold, heatmap, 0)

    heatmap = cv2.resize(
        heatmap,
        (224, 224),
        interpolation=cv2.INTER_CUBIC,
    )

    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    heatmap = heatmap ** power

    return heatmap


def find_available_target_token(text, candidate_tokens):
    text = str(text).lower()

    if isinstance(candidate_tokens, str):
        candidate_tokens = [candidate_tokens]

    for token in candidate_tokens:
        if token.lower() in text:
            return token

    return None


def visualize_one_sample(
    model,
    dataset,
    df,
    tokenizer,
    idx,
    target_label,
    target_token,
    device,
):
    sample = dataset[idx]

    pixel_values = sample["pixel_values"].unsqueeze(0).to(device)
    input_ids = sample["input_ids"].unsqueeze(0).to(device)
    attention_mask = sample["attention_mask"].unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
        )

    attention_weights = outputs["attention_weights"].squeeze(0).cpu().numpy()
    tokens = tokenizer.convert_ids_to_tokens(sample["input_ids"])

    token_positions = find_token_positions(tokens, target_token)

    if token_positions is None:
        print(f"[SKIP] Token '{target_token}' not found after tokenization for index {idx}")
        return

    patch_attention = attention_weights[token_positions].mean(axis=0)

    heatmap = process_attention_heatmap(
        patch_attention,
        top_percent=10,
        power=2,
    )

    image_path = df.loc[idx, "image_path"]
    original_image = Image.open(image_path).convert("RGB").resize((224, 224))

    plt.figure(figsize=(6, 6))
    plt.imshow(original_image, cmap="gray")
    plt.imshow(heatmap, cmap="jet", alpha=0.45, extent=(0, 224, 224, 0))
    plt.axis("off")

    output_path = OUTPUT_DIR / f"{target_label}_token_{target_token}_sample_{idx}.png"
    plt.savefig(output_path, bbox_inches="tight", dpi=300)
    plt.close()

    print(f"[SAVED] {output_path}")
    print(f"  label={target_label}, token={target_token}, idx={idx}")
    print(f"  ground_truth={df.loc[idx, 'labels']}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(METADATA_PATH)
    df["labels"] = df["labels"].apply(ast.literal_eval)

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    dataset = IUXrayDataset(
        dataframe=df,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=128,
    )

    model = MultimodalCXRModel(num_labels=5).to(device)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.eval()

    samples_per_label = 3

    for target_label, candidate_tokens in LABEL_CONFIG.items():
        label_col = f"label_{target_label}"

        positive_df = df[df[label_col] == 1].copy()

        positive_df["available_token"] = positive_df["text"].apply(
            lambda x: find_available_target_token(x, candidate_tokens)
        )

        positive_df = positive_df[
            positive_df["available_token"].notna()
        ]

        if len(positive_df) == 0:
            print(f"[SKIP] No positive samples with visible tokens for {target_label}")
            continue

        selected_rows = positive_df.sample(
            n=min(samples_per_label, len(positive_df)),
            random_state=42,
        )

        for idx, row in selected_rows.iterrows():
            visualize_one_sample(
                model=model,
                dataset=dataset,
                df=df,
                tokenizer=tokenizer,
                idx=idx,
                target_label=target_label,
                target_token=row["available_token"],
                device=device,
            )


if __name__ == "__main__":
    main()
