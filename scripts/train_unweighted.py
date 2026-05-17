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

LABEL_COLS = [
    "label_effusion",
    "label_cardiomegaly",
    "label_edema",
    "label_pneumonia",
    "label_pneumothorax",
]


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

    print("\nTraining label counts:")
    print(train_df[LABEL_COLS].sum())

    print("\nTraining positive ratio:")
    print(train_df[LABEL_COLS].mean())

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

    model = MultimodalCXRModel(num_labels=5).to(device)

    criterion = torch.nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,
        weight_decay=1e-4,
    )

    num_epochs = 5

    save_path = Path("checkpoints")
    save_path.mkdir(exist_ok=True)

    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0.0

        progress_bar = tqdm(
            train_loader,
            desc=f"Unweighted Epoch {epoch + 1}/{num_epochs}",
        )

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

        print(f"\nUnweighted Epoch {epoch + 1}")
        print(f"Train loss: {avg_train_loss:.4f}")
        print(f"Val loss:   {avg_val_loss:.4f}")

        checkpoint_path = save_path / f"unweighted_multimodal_cxr_model_epoch_{epoch + 1}.pt"
        torch.save(model.state_dict(), checkpoint_path)

        print(f"Model saved to {checkpoint_path}")


if __name__ == "__main__":
    main()
