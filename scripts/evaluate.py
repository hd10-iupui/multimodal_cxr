from pathlib import Path
import ast

import numpy as np
import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
)
from tqdm import tqdm

from data_utils.iu_xray_dataset import IUXrayDataset
from models.multimodal_model import MultimodalCXRModel


LABEL_NAMES = [
    "effusion",
    "cardiomegaly",
    "edema",
    "pneumonia",
    "pneumothorax",
]

DATA_DIR = Path(r"C:\multimodal_cxt\data\iu_xray")
METADATA_PATH = DATA_DIR / "iu_xray_labeled_metadata.csv"

CHECKPOINTS = {
    "weighted_epoch_1": Path("checkpoints/multimodal_cxr_model_epoch_1.pt"),
    "weighted_epoch_2": Path("checkpoints/multimodal_cxr_model_epoch_2.pt"),
    "weighted_epoch_3": Path("checkpoints/multimodal_cxr_model_epoch_3.pt"),
    "weighted_epoch_4": Path("checkpoints/multimodal_cxr_model_epoch_4.pt"),
    "weighted_epoch_5": Path("checkpoints/multimodal_cxr_model_epoch_5.pt"),
    "unweighted_epoch_1": Path("checkpoints/unweighted_multimodal_cxr_model_epoch_1.pt"),
    "unweighted_epoch_2": Path("checkpoints/unweighted_multimodal_cxr_model_epoch_2.pt"),
    "unweighted_epoch_3": Path("checkpoints/unweighted_multimodal_cxr_model_epoch_3.pt"),
    "unweighted_epoch_4": Path("checkpoints/unweighted_multimodal_cxr_model_epoch_4.pt"),
    "unweighted_epoch_5": Path("checkpoints/unweighted_multimodal_cxr_model_epoch_5.pt")
}


def evaluate_checkpoint(model, dataloader, device):
    model.eval()
    all_probs = []
    all_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].cpu()

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            probs = torch.sigmoid(outputs["logits"]).cpu()

            all_probs.append(probs)
            all_labels.append(labels)

    y_prob = torch.cat(all_probs, dim=0).numpy()
    y_true = torch.cat(all_labels, dim=0).numpy()

    return y_true, y_prob


def find_best_threshold(true_i, prob_i):
    thresholds = np.arange(0.01, 0.51, 0.01)

    best_result = {
        "best_threshold": 0.5,
        "best_f1": 0.0,
        "precision_at_best": 0.0,
        "recall_at_best": 0.0,
    }

    for threshold in thresholds:
        pred_i = (prob_i >= threshold).astype(int)

        precision = precision_score(true_i, pred_i, zero_division=0)
        recall = recall_score(true_i, pred_i, zero_division=0)
        f1 = f1_score(true_i, pred_i, zero_division=0)

        if f1 > best_result["best_f1"]:
            best_result = {
                "best_threshold": round(float(threshold), 2),
                "best_f1": float(f1),
                "precision_at_best": float(precision),
                "recall_at_best": float(recall),
            }

    return best_result


def compute_metrics(y_true, y_prob):
    rows = []

    for i, label_name in enumerate(LABEL_NAMES):
        true_i = y_true[:, i]
        prob_i = y_prob[:, i]

        if len(set(true_i)) < 2:
            auroc = None
            auprc = None
        else:
            auroc = roc_auc_score(true_i, prob_i)
            auprc = average_precision_score(true_i, prob_i)

        best = find_best_threshold(true_i, prob_i)

        rows.append({
            "label": label_name,
            "positive_count": int(true_i.sum()),
            "positive_ratio": float(true_i.mean()),
            "auroc": auroc,
            "auprc": auprc,
            "best_threshold": best["best_threshold"],
            "best_f1": best["best_f1"],
            "precision_at_best": best["precision_at_best"],
            "recall_at_best": best["recall_at_best"],
            "mean_prob": float(prob_i.mean()),
        })

    return pd.DataFrame(rows)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(METADATA_PATH)
    df["labels"] = df["labels"].apply(ast.literal_eval)

    _, val_df = train_test_split(
        df,
        test_size=0.2,
        random_state=42,
        shuffle=True,
    )

    tokenizer = AutoTokenizer.from_pretrained("emilyalsentzer/Bio_ClinicalBERT")

    image_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    val_dataset = IUXrayDataset(
        dataframe=val_df,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=128,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=8,
        shuffle=False,
        num_workers=0,
    )

    all_results = []

    for checkpoint_name, checkpoint_path in CHECKPOINTS.items():
        if not checkpoint_path.exists():
            print(f"\nSkipping missing checkpoint: {checkpoint_path}")
            continue

        print("\n==============================")
        print(f"Evaluating: {checkpoint_name}")
        print(f"Checkpoint: {checkpoint_path}")

        model = MultimodalCXRModel(num_labels=5).to(device)
        model.load_state_dict(
            torch.load(
                checkpoint_path,
                map_location=device,
                weights_only=True
            )
        )

        y_true, y_prob = evaluate_checkpoint(
            model=model,
            dataloader=val_loader,
            device=device,
        )

        metrics_df = compute_metrics(y_true, y_prob)
        metrics_df.insert(0, "checkpoint", checkpoint_name)

        print(metrics_df)
        all_results.append(metrics_df)

    if all_results:
        final_df = pd.concat(all_results, ignore_index=True)

        output_path = Path("evaluation_results_000.csv")
        final_df.to_csv(output_path, index=False)

        print(f"\nSaved evaluation results to: {output_path}")

if __name__ == "__main__":
    main()
