"""add weighted BCE"""

from pathlib import Path
import ast

import torch
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from data_utils.iu_xray_dataset import IUXrayDataset
from models.multimodal_model import MultimodalCXRModel


DATA_DIR = Path(r"C:\multimodal_cxt\data\iu_xray")
METADATA_PATH = DATA_DIR / "iu_xray_labeled_metadata.csv"


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)

    df = pd.read_csv(METADATA_PATH)
    df["labels"] = df["labels"].apply(ast.literal_eval)

    train_df, val_df = train_test_split(
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

    train_dataset = IUXrayDataset(
        dataframe=train_df,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=128,
    )

    val_dataset = IUXrayDataset(
        dataframe=val_df,
        tokenizer=tokenizer,
        image_transform=image_transform,
        max_length=128,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=4,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=4,
        shuffle=False,
        num_workers=0,
    )

    # label distribution /  detect class imbalance
    label_cols = [
        "label_effusion",
        "label_cardiomegaly",
        "label_edema",
        "label_pneumonia",
        "label_pneumothorax",
    ]

    label_counts = train_df[label_cols].sum()
    num_samples = len(train_df)

    # pos_weight. Rare positive --> punish positive errors
    pos_weights = []

    for col in label_cols:
        positive = label_counts[col]
        negative = num_samples - positive

        pos_weight = negative / max(positive, 1)  # in case a train split having no positive
        pos_weights.append(pos_weight)

    print("Positive counts:")
    print(label_counts)

    print("pos_weights:")
    print(pos_weights)

    pos_weights_tensor = torch.tensor(
        pos_weights,
        dtype=torch.float32
    ).to(device)

    criterion = torch.nn.BCEWithLogitsLoss(
        pos_weight=pos_weights_tensor
    )

    model = MultimodalCXRModel(num_labels=5).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-4,
    )

    num_epochs = 5

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")

        for batch in progress_bar:
            pixel_values = batch["pixel_values"].to(device)
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            optimizer.zero_grad()

            outputs = model(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
            )

            logits = outputs["logits"]
            loss = criterion(logits, labels)

            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            progress_bar.set_postfix(loss=loss.item())

        avg_train_loss = total_train_loss / len(train_loader)

        model.eval()
        total_val_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                pixel_values = batch["pixel_values"].to(device)
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"].to(device)

                outputs = model(
                    pixel_values=pixel_values,
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )

                logits = outputs["logits"]
                loss = criterion(logits, labels)

                total_val_loss += loss.item()

        avg_val_loss = total_val_loss / len(val_loader)

        print(f"Epoch {epoch + 1}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val loss:   {avg_val_loss:.4f}")

        save_path = Path("checkpoints")
        save_path.mkdir(exist_ok=True)

        torch.save(model.state_dict(), save_path / f"multimodal_cxr_model_epoch_{epoch+1}.pt")
        # print("Model saved to checkpoints/multimodal_cxr_model.pt")


if __name__ == "__main__":
    main()
